"""
Type definitions for lammps-utils.

This module provides type aliases and type definitions used throughout
the lammps-utils package.
"""

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


__all__ = ("CellBounds",)


CellBounds: TypeAlias = tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
]
"""Type alias for periodic cell bounds.

A tuple of three (lo, hi) tuples representing the lower and upper bounds
for each axis (x, y, z) in a periodic simulation cell.
"""
