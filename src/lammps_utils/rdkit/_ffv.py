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
    cell_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ],
    confId: int = -1,
    probe_radius: float = 1.4,
    grid_spacing: float = 1.0,
    n_jobs: Optional[int] = None,
) -> float:
    conf = wrap_mol_positions_to_cell(
        mol, cell_bounds=cell_bounds
    ).GetConformer(confId)
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
    単一グリッド点が自由かどうか判定する関数。
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
    joblibで並列化した自由体積分率計算。

    Parametersは従来版と同じです。
    n_jobsは並列ジョブ数（-1は全CPU利用）
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

    # joblibで並列化
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_check_free_single)(
            grid[idx], candidate_indices, positions, effective_radii
        )
        for idx, candidate_indices in enumerate(candidate_indices_list)
    )

    is_free = np.asarray(results, dtype=bool)
    ffv = np.sum(is_free, axis=None).item() / len(grid)
    return ffv
