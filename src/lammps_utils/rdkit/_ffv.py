from collections.abc import Sequence
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from scipy.spatial import KDTree

from lammps_utils.constants import MAP_VDW_RADIUS
from lammps_utils.rdkit._pbc import wrap_mol_positions_to_cell


def compute_ffv(
    mol: Chem.rdchem.Mol,
    confId: int = -1,
    cell_bounds: Optional[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = None,
    probe_radius: float = 1.4,
    grid_spacing: float = 1.0,
    n_jobs: Optional[int] = None,
) -> float:
    """
    Compute the fractional free volume (FFV) of a molecule.

    If `cell_bounds` is not provided, it is automatically determined from
    the conformer properties ("xlo", "xhi", etc.) assumed to be preassigned.

    Parameters
    ----------------
    mol : Chem.rdchem.Mol
        The input RDKit molecule.
    confId : int
        The conformer ID to use from the molecule.
    cell_bounds : tuple of tuple of float, optional
        The periodic cell boundaries as ((xlo, xhi), (ylo, yhi), (zlo, zhi)).
        If None, the bounds will be extracted from conformer properties.
    probe_radius : float
        The radius of the spherical probe for free volume determination.
    grid_spacing : float
        The spacing of the grid used to sample the cell volume.
    n_jobs : int, optional
        The number of parallel jobs to run. Use -1 to utilize all CPUs.

    Returns
    ----------------
    float
        The fractional free volume of the molecule.
    """
    conf = wrap_mol_positions_to_cell(
        mol, cell_bounds=cell_bounds
    ).GetConformer(confId)
    if cell_bounds is None:
        tup_tmp = tuple(
            (conf.GetDoubleProp(f"{axis}lo"), conf.GetDoubleProp(f"{axis}hi"))
            for axis in ("x", "y", "z")
        )
        assert len(tup_tmp) == 3
        cell_bounds = tup_tmp

    return calculate_ffv_parallel(
        conf.GetPositions(),
        vdw_radii=np.array(
            [
                MAP_VDW_RADIUS[atom.GetSymbol()]
                for atom in mol.GetAtoms()
                if isinstance(atom, Chem.Atom)
            ]
        ),
        cell_bounds=cell_bounds,
        probe_radius=probe_radius,
        grid_spacing=grid_spacing,
        n_jobs=n_jobs,
    )


def _check_free_single(
    grid_point: np.ndarray,
    candidate_indices: Sequence[int],
    positions: np.ndarray,
    effective_radii: np.ndarray,
) -> bool:
    """
    Determine whether a single grid point is located in the free volume.

    A grid point is considered "free" if it lies outside the effective van der Waals
    spheres (vdW radius + probe radius) of all nearby atoms.

    Parameters
    ----------------
    grid_point : np.ndarray
        A 3-element array representing the (x, y, z) coordinates of the grid point.
    candidate_indices : Sequence[int]
        Indices of atoms that are within the maximum effective radius of the grid point,
        obtained via spatial indexing (e.g., KDTree).
    positions : np.ndarray
        Array of atomic coordinates with shape (N_atoms, 3).
    effective_radii : np.ndarray
        Array of effective radii (vdW radius + probe radius) for each atom.

    Returns
    ----------------
    bool
        True if the grid point is in the free volume (i.e., not overlapping with
        any atom's effective radius), False otherwise.
    """
    if not candidate_indices:
        return True

    pos_j = positions[candidate_indices]
    radii_j = effective_radii[candidate_indices]
    diff = pos_j - grid_point
    dist2 = np.einsum("ij,ij->i", diff, diff)
    radius2 = radii_j**2

    return not np.any(dist2 < radius2, axis=None).item()


def calculate_ffv_parallel(
    positions: np.ndarray,
    vdw_radii: np.ndarray,
    cell_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ],
    probe_radius: float = 1.4,
    grid_spacing: float = 1.0,
    n_jobs: Optional[int] = None,
) -> float:
    """
    Compute the fractional free volume (FFV) using parallel processing.

    This function evaluates the free volume by placing a regular 3D grid
    inside the specified cell and checking for each grid point whether
    a probe sphere centered at that point does not overlap with any atoms.

    Parameters
    ----------------
    positions : np.ndarray
        Array of atomic coordinates with shape (N_atoms, 3).
    vdw_radii : np.ndarray
        Array of van der Waals radii for each atom.
    cell_bounds : tuple of tuple of float
        The periodic cell boundaries as ((xlo, xhi), (ylo, yhi), (zlo, zhi)).
    probe_radius : float
        The radius of the spherical probe.
    grid_spacing : float
        The spacing of the 3D grid in each dimension.
    n_jobs : int, optional
        Number of parallel jobs to use with joblib. -1 uses all available CPUs.

    Returns
    ----------------
    float
        The fractional free volume, defined as the fraction of grid points
        where the probe does not intersect any atom's van der Waals sphere.
    """
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = cell_bounds

    x = np.arange(xlo, xhi, grid_spacing, dtype=np.float32)
    y = np.arange(ylo, yhi, grid_spacing, dtype=np.float32)
    z = np.arange(zlo, zhi, grid_spacing, dtype=np.float32)
    grid = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T

    effective_radii = vdw_radii + probe_radius
    max_effective_radius = np.max(effective_radii).item()

    tree = KDTree(positions)
    candidate_indices_list: list[list[int]] = tree.query_ball_point(
        grid, r=max_effective_radius
    ).tolist()

    # Parallelize with joblib
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_check_free_single)(
            grid[idx], candidate_indices, positions, effective_radii
        )
        for idx, candidate_indices in enumerate(candidate_indices_list)
    )

    is_free = np.asarray(results, dtype=bool)
    ffv = np.sum(is_free, axis=None).item() / len(grid)
    return ffv
