import pytest

from guillemot.session import end_session


@pytest.fixture(autouse=True)
def _clean_global_state():
    """Reset the module-level state the app sets up at startup."""
    end_session()
    yield
    end_session()
