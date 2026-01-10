"""Command-line interface for lammps-utils."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from lammps_utils import __version__
from lammps_utils.io.convert._convert import data2gro, data2pdb
from lammps_utils.logging import get_child_logger

_logger = get_child_logger(__name__)


__all__ = ("main",)


def _data2gro_cli(
    args: argparse.Namespace,
) -> None:
    """
    Command-line handler for data2gro conversion.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing 'input' and optionally 'output'.
    """
    filepath_data: Path = args.input
    filepath_gro: Optional[Path] = getattr(args, "output", None)

    if filepath_gro is None:
        filepath_gro = filepath_data.with_suffix(".gro")

    data2gro(filepath_data, filepath_gro)
    _logger.info(f"Converting '{filepath_data}' to '{filepath_gro}'")


def _data2pdb_cli(
    args: argparse.Namespace,
) -> None:
    """
    Command-line handler for data2pdb conversion.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing 'input' and optionally 'output'.
    """
    filepath_data: Path = args.input
    filepath_pdb: Optional[Path] = getattr(args, "output", None)

    if filepath_pdb is None:
        filepath_pdb = filepath_data.with_suffix(".pdb")

    data2pdb(filepath_data, filepath_pdb)
    _logger.info(f"Converting '{filepath_data}' to '{filepath_pdb}'")


def main(cli_args: Sequence[str], prog: Optional[str] = None) -> None:
    """
    Main entry point for the lammps-utils command-line interface.

    Parameters
    ----------
    cli_args : Sequence[str]
        Command-line arguments (typically sys.argv[1:]).
    prog : Optional[str], optional
        Program name for the argument parser. Default is None.

    Notes
    -----
    This function sets up the argument parser with subcommands for data2gro
    and data2pdb conversions.
    """
    parser = argparse.ArgumentParser(prog=prog, description="LAMMPS utils CLI")
    # subcommand
    subparsers = parser.add_subparsers(dest="command")

    # data2gro
    data2gro_parser = subparsers.add_parser(
        "data2gro",
        help="Convert LAMMPS data file to GROMACS gro file",
    )
    data2gro_parser.add_argument(
        "input",
        type=Path,
        help="Input LAMMPS data file or file-like object",
    )
    data2gro_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GROMACS gro file or file-like object",
    )
    data2gro_parser.set_defaults(func=_data2gro_cli)

    # data2pdb
    data2pdb_parser = subparsers.add_parser(
        "data2pdb",
        help="Convert LAMMPS data file to pdb file",
    )
    data2pdb_parser.add_argument(
        "input",
        type=Path,
        help="Input LAMMPS data file or file-like object",
    )
    data2pdb_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output pdb file or file-like object",
    )
    data2pdb_parser.set_defaults(func=_data2pdb_cli)

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        help="show current version",
        version=f"%(prog)s: {__version__}",
    )
    args = parser.parse_args(cli_args)

    args.func(args)


def entrypoint() -> None:
    """
    Entry point function called by the console script.

    This function is typically called when the package is invoked as a command
    from the command line (e.g., `lammps-utils data2pdb input.data`).
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:], prog="lammps-utils")
