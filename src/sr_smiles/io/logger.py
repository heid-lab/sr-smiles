import logging

logger = logging.getLogger("sr_smiles")
logger.setLevel(logging.WARNING)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.NOTSET)
    formatter = logging.Formatter(
        fmt="[{asctime}] {levelname}: {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.propagate = False


def set_verbose(verbose: bool = True, debug: bool = False):
    """Allow users to control logging verbosity programmatically."""
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
