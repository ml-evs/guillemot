# This file was edited with the assistance of an AI model and requires human review from the contributor.
"""The plot has to reach the model as an image.

Returning `BinaryContent` nested inside a pydantic model looks like it works — the
tool call succeeds and the trace shows an `output_image` — but pydantic-ai serialises
it to base64 text, so the model is billed for tens of thousands of tokens of
characters it cannot read and never sees a picture at all. Nothing else catches that,
hence this test.
"""

import pathlib

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from guillemot.tools.plotting import plot_refinement_results, render_refinement_plot

FIXTURE = pathlib.Path("examples/FeSb_19RBM/Runs/1")


def _agent_that_plots(save_path: pathlib.Path) -> Agent:
    def script(messages, info):
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "plot_refinement_results",
                        {
                            "output_file": str(FIXTURE / "output.txt"),
                            "save_path": str(save_path),
                            "hkl_file": str(FIXTURE / "hkl.txt"),
                        },
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    return Agent(FunctionModel(script), tools=[plot_refinement_results])


def _parts_of(messages):
    for message in messages:
        for part in message.parts:
            yield part


def test_plot_tool_sends_the_model_an_image(tmp_path):
    save_path = tmp_path / "fit.png"
    result = _agent_that_plots(save_path).run_sync("show me the fit")

    images = []
    for part in _parts_of(result.all_messages()):
        content = getattr(part, "content", None)
        for item in content if isinstance(content, list) else [content]:
            if isinstance(item, BinaryContent):
                images.append((type(part).__name__, item.media_type))

    assert images == [("UserPromptPart", "image/png")]
    assert save_path.is_file()


def test_plot_tool_sends_no_base64_text(tmp_path):
    """The bytes must not also appear as characters in the text stream."""
    result = _agent_that_plots(tmp_path / "fit.png").run_sync("show me the fit")

    text = ""
    for part in _parts_of(result.all_messages()):
        content = getattr(part, "content", None)
        if isinstance(content, str):
            text += content
        elif isinstance(content, list):
            text += "".join(item for item in content if isinstance(item, str))

    assert len(text) < 2000, "the image is being serialised into the text stream"


def test_render_returns_a_path_and_no_image(tmp_path):
    """The refinement tools use this: it draws the PNG without paying for an image."""
    save_path = tmp_path / "fit.png"
    assert render_refinement_plot(str(FIXTURE / "output.txt"), str(save_path)) == str(
        save_path
    )
    assert save_path.is_file()


def test_concurrent_plots_do_not_collide(tmp_path):
    """The agent runs tool calls in worker threads; pyplot's mathtext parser is not
    thread-safe, and three zoomed plots in one turn once ended a benchmark run."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from guillemot.tools.plotting import render_refinement_plot

    x = np.linspace(10, 80, 2000)
    y = 100 + 1000 * np.exp(-((x - 30) ** 2) / 0.05)
    output = tmp_path / "riet_01_output.txt"
    np.savetxt(output, np.column_stack([x, y, y * 0.97]))

    def render(i):
        return render_refinement_plot(
            output_file=str(output),
            save_path=str(tmp_path / f"zoom_{i}.png"),
            x_range=[20 + i % 5, 60],
        )

    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(render, range(24)))
    assert all(pathlib.Path(p).is_file() for p in paths)
