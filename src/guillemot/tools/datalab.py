from datalab_api import DatalabClient
from datalab_api._base import DatalabAPIError
import os
from pathlib import Path
from pydantic_ai import ModelRetry

DATALAB_INSTANCE = "demo.datalab-org.io"


def _explain(exc: DatalabAPIError) -> str:
    """Describe a datalab failure so the agent can tell the user about it.

    Returned rather than raised: an uncaught exception here aborts the whole agent
    run, so the user gets no reply at all. These are configuration problems (an
    expired API key, an unreachable instance) that retrying cannot fix.
    """
    hint = (
        " Their DEMO_DATALAB_API_KEY is missing or expired; a new one can be generated "
        "from their account page on the instance."
        if "401" in str(exc)
        else ""
    )
    return (
        f"Could not reach datalab at {DATALAB_INSTANCE}: {exc}. Tell the user.{hint} "
        "Do not retry. Files already on disk can still be used."
    )


def get_samples():
    """Get a list of all samples available in the Datalab
    and list whether they have XRD data."""
    try:
        client = DatalabClient(DATALAB_INSTANCE)
        return client.get_items("samples")
    except DatalabAPIError as exc:
        return _explain(exc)


def get_sample(sample_id: str | None):
    """Get a sample with the given ID and download its associated data files."""
    try:
        client = DatalabClient(DATALAB_INSTANCE)
        sample = client.get_item(sample_id)
    except DatalabAPIError as exc:
        return _explain(exc)

    orig_dir = Path(os.getcwd()).absolute()
    try:
        os.makedirs(f"./data/{sample_id}", exist_ok=True)
        os.chdir(f"./data/{sample_id}")
        files = client.get_item_files(sample_id)
        return sample, files

    finally:
        os.chdir(str(orig_dir))


def list_data_files(sample_id: str):
    """List data files that have been downloaded for a given sample."""
    return os.listdir(f"./data/{sample_id}")
