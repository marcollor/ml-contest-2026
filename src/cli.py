"""Shared command-line help and validation behavior for local tools."""

import argparse
import sys


class HelpArgumentParser(argparse.ArgumentParser):
    """Print full help on invalid arguments and reject abbreviated options."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)

    def error(self, message):
        """Stop before application work and show the error plus complete help."""
        self._print_message(f"{self.prog}: error: {message}\n\n", sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)
