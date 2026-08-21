import argparse
import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from guillemot.tools import (
    check_remote_topas_running,
    get_optimade_structures,
    list_available_data,
    print_structure,
    print_structures,
    plot_refinement_results,
    run_topas_refinement,
    run_topas_refinement_remote,
    save_topas_inp,
)
from guillemot.model import build_model
from guillemot.session import copy_into_session, current_session, start_session
from guillemot.tools.datalab import get_sample, get_samples, list_data_files
from pydantic_ai import Agent, BinaryContent, ImageUrl
from guillemot.utils import load_local_image

# Load environment variables
load_dotenv()

# Configure Logfire if token is available
try:
    import logfire

    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        logfire.configure(token=logfire_token)
except ImportError:
    pass  # Logfire is optional


@dataclass
class ConversationHistory:
    """Store conversation history for memory"""

    messages: List[Dict[str, Any]]

    def add_message(
        self,
        role: str,
        content: str,
        has_image: bool = False,
        timestamp: datetime | None = None,
    ):
        if timestamp is None:
            timestamp = datetime.now()
        self.messages.append(
            {
                "role": role,
                "content": content,
                "has_image": has_image,
                "timestamp": timestamp.isoformat(),
            }
        )

    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages for context"""
        return self.messages[-limit:]

    def get_formatted_history(self, limit: int = 5) -> str:
        """Format recent conversation history for the AI context"""
        recent = self.get_recent_messages(limit)
        formatted = []
        for msg in recent:
            content = msg["content"]
            if msg.get("has_image", False):
                content += " [included an image]"
            formatted.append(f"{msg['role']}: {content}")
        return "\n".join(formatted)


# Initialize conversation history
conversation_history = ConversationHistory(messages=[])


def is_local_image_path(text: str) -> bool:
    """Check if text contains a local image file path"""
    # Look for file:// URLs or local paths ending with image extensions
    file_pattern = r"(?:file://)?[^\s]+\.(jpg|jpeg|png|gif|bmp|webp)"
    return bool(re.search(file_pattern, text, re.IGNORECASE))


def extract_local_image_path(text: str) -> tuple[str, str]:
    """Extract local image path from text and return (text_without_path, image_path)"""
    file_pattern = r"(?:file://)?([^\s]+\.(jpg|jpeg|png|gif|bmp|webp))"
    match = re.search(file_pattern, text, re.IGNORECASE)
    if match:
        image_path = match.group(1)
        # Remove file:// prefix if present
        image_path = image_path.replace("file://", "")
        text_without_path = text.replace(match.group(), "").strip()
        return text_without_path, image_path
    return text, ""


# Set up the pydantic-ai agent
def create_agent() -> Agent:
    """Create and configure the pydantic-ai agent"""

    # Get model and API key from environment. `build_model` adds retries for rate
    # limiting (HTTP 429), which would otherwise abort the run mid-refinement.
    model_name = os.getenv("GUILLEMOT_AI_MODEL", "gemini-2.5-flash-lite")
    model = build_model(model_name)

    # One working directory per conversation, so this refinement's files stay together
    # and cannot be mixed up with an earlier one's.
    session = start_session()

    with open("examples/NaCoO2/example_refinement_NaCoO2.inp", "r") as f:
        topas_example = f.read()

    remote_host = os.getenv("GUILLEMOT_TOPAS_SSH_HOST")
    if remote_host:
        execution_prompt = f"""TOPAS is not installed locally: it runs on the remote machine
'{remote_host}' over SSH. Always use `run_topas_refinement_remote` to run refinements, never
`run_topas_refinement`. That tool refuses to start if TOPAS is already busy on the remote machine;
you can check that yourself with `check_remote_topas_running`. Each remote run gets its own run
directory which is copied back locally, so the paths in the result refer to the local copies.
TOPAS runs with that directory as its working directory, so every filename in the .inp file —
the data files it reads and the names given to the Out_* macros — should be a bare filename with
no directory part."""
    else:
        execution_prompt = """TOPAS runs on this machine, so use `run_topas_refinement` to run
refinements."""

    # Create the agent with tools
    agent = Agent(
        model,
        system_prompt=f"""You are an agent responsible for performing Rietveld refinements using
 the topas-academic program. You have access to a tool to write topas .inp files to a run directory,
 and a tool to run the refinement and get the results. You perform Rietveld refinements the way
 human researchers do: looking at an X-ray diffraction pattern, deciding which phases are most likely
to be present based on the pattern, then trying some basic refinements and looking at the results before
iterating to get the fit as good as possible.
You can also analyze and understand images that users share with you. Use this to look at images of Rietveld
refinements and plan your next refinement.
Give a summary of what you've done at the end, telling each refinement you did, explaining any errors you found,
and explaining why you made changes before the next refinement.

If the user does not provide a CIF, you can search in the Materials Project or Crystallography Open Database
via OPTIMADE. These searches will typically return a table of structures matching the query, which can be printed with `print_structures`. You can then print the most promising structures with `print_structure` and use the info to construct your TOPAS input.

Here is an example of a topas input file for refinement of a sample of NaCoO2: {topas_example}
    """,
        tools=[
            list_available_data,
            copy_into_session,
            save_topas_inp,
            # run_topas_refinement,
            run_topas_refinement_remote,
            check_remote_topas_running,
            get_optimade_structures,
            print_structure,
            print_structures,
            plot_refinement_results,
            get_samples,
            get_sample,
            list_data_files,
        ],
        instrument=True,
        retries=5,
    )

    return agent


async def chat_loop():
    """Main chat loop for the terminal application"""
    print("🪶 Guillemot chat framework")
    print("=" * 40)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'history' to see recent conversation history")
    print("Type 'clear' to clear conversation history")
    print("🖼️  Include image URLs or local file paths for image analysis")
    print("   Examples: 'Describe this image: https://example.com/image.jpg'")
    print("             'What's in this photo? /path/to/image.png'")
    print("=" * 40)

    agent = create_agent()
    session = current_session()
    if session is not None:
        print(f"📁 Session directory: {session.directory}")

    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()

            # Handle special commands
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "history":
                print("\n📜 Recent Conversation History:")
                print("-" * 30)
                recent_messages = conversation_history.get_recent_messages(10)
                for msg in recent_messages:
                    role_emoji = "💬" if msg["role"] == "user" else "🤖"
                    content = msg["content"]
                    if msg.get("has_image", False):
                        content += " 🖼️"
                    print(f"{role_emoji} {msg['role']}: {content}")
                continue

            if user_input.lower() == "clear":
                conversation_history.messages.clear()
                print("\n🧹 Conversation history cleared!")
                continue

            if not user_input:
                continue

            # Check for image content
            has_image = False
            message_parts = []

            # Check for local image path
            if is_local_image_path(user_input):
                text_without_path, image_path = extract_local_image_path(user_input)
                image_content = load_local_image(image_path)

                if image_content:
                    if text_without_path:
                        message_parts.append(text_without_path)
                    else:
                        message_parts.append("Please analyze this image:")
                    message_parts.append(image_content)
                    has_image = True
                    print(f"🖼️  Loaded local image: {image_path}")
                else:
                    print("❌ Failed to load image. Proceeding with text only.")
                    message_parts.append(user_input)

            else:
                # No image, just text
                message_parts.append(user_input)

            # Add user message to history
            conversation_history.add_message("user", user_input, has_image=has_image)

            # Prepare context with conversation history
            history_context = conversation_history.get_formatted_history(5)

            # Create the message to send to the agent
            if has_image:
                # For image messages, send the message parts directly
                agent_message = message_parts
            else:
                # For text-only messages, include conversation history
                full_prompt = f"""
                Recent conversation history:
                {history_context}

                Current user message: {user_input}
                """
                agent_message = full_prompt

            print("\n🤖 Assistant: ", end="", flush=True)

            # Run the agent
            result = await agent.run(agent_message)

            # Print the response
            response_text = result.output
            print(response_text)

            # Add assistant response to history
            conversation_history.add_message("assistant", response_text)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def build_task(
    pattern: str, elements: list[str] | None = None, notes: str | None = None
) -> str:
    """Write the instruction for a one-shot refinement started from the command line."""
    task = [
        f"Refine the diffraction pattern in the file `{pattern}`.",
        "Work through it end to end without asking me anything: copy the pattern into "
        "the session directory, write a TOPAS input file for it, run the refinement, "
        "and then tell me the Rwp and what you would try next.",
    ]
    if elements:
        task.insert(1, f"The sample is believed to contain: {', '.join(elements)}.")
    else:
        task.insert(
            1,
            "I have not told you the composition, so work out what you can from the "
            "filename; if you genuinely cannot, say so instead of guessing.",
        )
    if notes:
        task.append(f"Additional context from me: {notes}")
    return "\n".join(task)


async def run_once(
    pattern: str, elements: list[str] | None = None, notes: str | None = None
) -> str:
    """Run a single refinement from the command line and print the result."""
    agent = create_agent()
    session = current_session()

    print("🪶 Guillemot")
    print(f"📁 Session directory: {session.directory if session else 'run_dir'}")
    print(f"📈 Pattern: {pattern}")
    print(f"🧪 Elements: {', '.join(elements) if elements else 'not given'}")
    print("=" * 60)

    result = await agent.run(build_task(pattern, elements, notes))
    print(result.output)
    print("=" * 60)
    if session is not None:
        print(f"📁 Everything from this run is in {session.directory}")
    return result.output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="guillemot",
        description=(
            "Run TOPAS Rietveld refinements with an LLM agent. With no arguments, "
            "starts an interactive chat; give --pattern to run one refinement and exit."
        ),
    )
    parser.add_argument(
        "--pattern",
        help="path to the diffraction pattern to refine, e.g. examples/HL2-1/HL2-1_2.xy",
    )
    parser.add_argument(
        "--elements",
        help="comma-separated elements believed to be present, e.g. Ag,Cu,Pd",
    )
    parser.add_argument(
        "--notes", help="anything else the agent should know about the sample"
    )
    return parser.parse_args(argv)


async def main(args: argparse.Namespace) -> None:
    """Main entry point"""
    try:
        if args.pattern:
            elements = (
                [e.strip() for e in args.elements.split(",") if e.strip()]
                if args.elements
                else None
            )
            await run_once(args.pattern, elements, args.notes)
        else:
            await chat_loop()
    except Exception as e:
        print(f"❌ Failed to run guillemot: {e}")
        print("Check your .env file: GUILLEMOT_AI_MODEL and the matching API key.")


def launch():
    asyncio.run(main(parse_args()))
