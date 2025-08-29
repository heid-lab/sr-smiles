import pytest

from cgr_smiles.logger import logger


@pytest.fixture
def propagated_logger():
    """A fixture to temporarily enable propagation on the logger for testing."""
    # fixture to enable propagation for the test
    original_propagation = logger.propagate
    logger.propagate = True
    yield logger

    # restore the original setting after testing
    logger.propagate = original_propagation
