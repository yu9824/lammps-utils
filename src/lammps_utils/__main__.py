import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from lammps_utils import __version__
from lammps_utils.io._convert import data2gro
from lammps_utils.logging import get_child_logger

_logger = get_child_logger(__name__)


__all__ = ("main",)


def _data2gro_cli(
    args: argparse.Namespace,
) -> None:
    filepath_data: Path = args.input
    filepath_gro: Optional[Path] = getattr(args, "output", None)

    if filepath_gro is None:
        filepath_gro = filepath_data.with_suffix(".gro")

    data2gro(filepath_data, filepath_gro)
    _logger.info(f"Converting '{filepath_data}' to '{filepath_gro}'")


def main(cli_args: Sequence[str], prog: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description="LAMMPS utils CLI")
    # subcommand
    subparsers = parser.add_subparsers(dest="command")
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
    main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:], prog="lammps-utils")
