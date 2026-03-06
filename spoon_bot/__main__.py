"""Entry point for running spoon-bot as a module."""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets\.?")

from spoon_bot.cli import app

if __name__ == "__main__":
    app()
