import pytest

from cgr_smiles.logger import set_verbose


@pytest.fixture(autouse=True)
def reset_logger_level():
    """A fixture to reset the logger's level to default after each test."""
    yield
    # this runs after each test to ensure a clean state
    set_verbose(verbose=False, debug=False)


def test_default_logging_level(propagated_logger, caplog):
    """Tests that by default, only WARNING and higher level messages are logged."""
    propagated_logger.debug("This is a debug message.")
    propagated_logger.info("This is an info message.")
    propagated_logger.warning("This is a warning message.")
    propagated_logger.error("This is an error message.")

    # assert that only the WARNING and ERROR messages were captured
    assert "This is a debug message." not in caplog.text
    assert "This is an info message." not in caplog.text
    assert "This is a warning message." in caplog.text
    assert "This is an error message." in caplog.text


def test_set_verbose_enables_info(propagated_logger, caplog):
    """Tests that verbose mode enables INFO but not DEBUG logs."""
    set_verbose(True)

    propagated_logger.debug("Debug should not appear")
    propagated_logger.info("Verbose info")
    propagated_logger.warning("Verbose warning")

    assert "Debug should not appear" not in caplog.text
    assert "Verbose info" in caplog.text
    assert "Verbose warning" in caplog.text


def test_set_debug_enables_debug(propagated_logger, caplog):
    """Tests that debug mode enables DEBUG and all higher logs."""
    set_verbose(debug=True)

    propagated_logger.debug("Verbose debug")
    propagated_logger.info("Verbose info")
    propagated_logger.warning("Verbose warning")

    assert "Verbose debug" in caplog.text
    assert "Verbose info" in caplog.text
    assert "Verbose warning" in caplog.text


def test_set_verbose_resets_to_warning(propagated_logger, caplog):
    """Tests that set_verbose(False) resets the level to WARNING."""
    set_verbose(debug=True)
    set_verbose(False)

    propagated_logger.info("This should not appear")
    propagated_logger.warning("This should appear")

    assert "This should not appear" not in caplog.text
    assert "This should appear" in caplog.text
