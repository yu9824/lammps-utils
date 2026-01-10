"""Module for converting LAMMPS data and dump files to RDKit molecules."""

import io
import os
from collections.abc import Sequence
from typing import Literal, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem

from lammps_utils.chem.bond._bond import get_bond_order
from lammps_utils.constants import COLS_XYZ
from lammps_utils.graph.pbc._pbc import unwrap_positions_under_pbc
from lammps_utils.helpers import tqdm_joblib
from lammps_utils.io.dataframe._dataframe import load_data, load_dump
from lammps_utils.io.dataframe._pbc import unwrap_df_positions_under_pbc
from lammps_utils.types import CellBounds


def _set_conformer_cell_bounds(
    conf: Chem.Conformer,
    cell_bounds: CellBounds,
) -> None:
    """
    Set cell bounds properties on a conformer.

    Parameters
    ----------
    conf : Chem.Conformer
        The conformer to set properties on.
    cell_bounds : CellBounds
        Cell bounds for each axis (x, y, z). A tuple of three (lo, hi) tuples
        representing the lower and upper bounds for x, y, and z axes.

    Notes
    -----
    This function sets the cell bounds as double properties on the conformer:
    "xlo", "xhi", "ylo", "yhi", "zlo", "zhi".
    """
    for idx_axis, axis in enumerate(COLS_XYZ):
        conf.SetDoubleProp(f"{axis}lo", cell_bounds[idx_axis][0])
        conf.SetDoubleProp(f"{axis}hi", cell_bounds[idx_axis][1])


def _calculate_bond_distances(
    df_atoms: pd.DataFrame,
    df_bonds: pd.DataFrame,
) -> dict[int, float]:
    """
    Calculate mean bond distances for each bond type.

    Parameters
    ----------
    df_atoms : pd.DataFrame
        DataFrame containing atom positions. Must have columns 'x', 'y', 'z'.
    df_bonds : pd.DataFrame
        DataFrame containing bond information. Must have columns 'atom1', 'atom2',
        and 'type'.

    Returns
    -------
    dict[int, float]
        Dictionary mapping bond type to mean distance in angstroms.
    """
    distances_by_type: dict[int, list[float]] = {}
    for _, bond_row in df_bonds.iterrows():
        atom1_pos = df_atoms.loc[bond_row["atom1"], COLS_XYZ].values
        atom2_pos = df_atoms.loc[bond_row["atom2"], COLS_XYZ].values
        distance = np.linalg.norm(atom1_pos - atom2_pos)
        bond_type = bond_row["type"]
        if bond_type not in distances_by_type:
            distances_by_type[bond_type] = []
        distances_by_type[bond_type].append(distance)

    return {
        bond_type: np.mean(distances).item()
        for bond_type, distances in distances_by_type.items()
    }


def _get_conformer_positions(
    df_atoms: pd.DataFrame,
    make_molecule_whole: bool,
    cell_bounds: Optional[CellBounds] = None,
    graph: Optional[nx.Graph] = None,
) -> np.ndarray:
    """
    Compute 3D positions (conformer coordinates) from atomic dataframe.

    Optionally unwraps positions under periodic boundary conditions (PBC)
    to make the molecule whole.

    Parameters
    ----------
    df_atoms : pd.DataFrame
        DataFrame containing atom positions. Must have columns 'x', 'y', 'z'.
    make_molecule_whole : bool
        Whether to unwrap positions under PBC. If True, `cell_bounds` and
        `graph` must be provided.
    cell_bounds : CellBounds, optional
        Cell bounds for each axis. Required if `make_molecule_whole` is True.
    graph : nx.Graph, optional
        Molecular graph for unwrapping positions. Required if
        `make_molecule_whole` is True.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing 3D coordinates for N atoms.

    Raises
    ------
    ValueError
        If `make_molecule_whole` is True but `graph` or `cell_bounds` is None.
    """
    df_atoms.sort_index(inplace=True)
    positions = df_atoms.loc[:, COLS_XYZ].values

    if make_molecule_whole:
        if graph is None:
            raise ValueError(
                "graph is required when make_molecule_whole is True"
            )
        if cell_bounds is None:
            raise ValueError(
                "cell_bounds is required when make_molecule_whole is True"
            )

        cell_size = tuple(hi - lo for lo, hi in cell_bounds)
        positions = unwrap_positions_under_pbc(
            graph, positions=positions, cell_size=cell_size
        )
    return positions


def _mol_from_dataframe_data(
    df_atoms: pd.DataFrame,
    df_bonds: pd.DataFrame,
    cell_bounds: CellBounds,
    determine_bonds: bool = True,
    make_molecule_whole: bool = True,
) -> Chem.rdchem.Mol:
    """
    Construct an RDKit molecule from atomic and bond dataframes.

    This function creates an RDKit molecule with atoms, bonds, and a single
    conformer from pandas DataFrames containing atomic and bonding information.

    Parameters
    ----------
    df_atoms : pd.DataFrame
        DataFrame containing atomic information. Must have columns including
        'symbol' for atom symbols and 'x', 'y', 'z' for coordinates.
    df_bonds : pd.DataFrame
        DataFrame containing bond information. Must have columns 'atom1',
        'atom2', and 'type'.
    cell_bounds : CellBounds
        Cell bounds for each axis (x, y, z). A tuple of three (lo, hi) tuples.
    determine_bonds : bool, optional
        If True, determine bond orders based on interatomic distances.
        Default is True.
    make_molecule_whole : bool, optional
        If True, use unwrapped positions for bond determination. If False,
        unwrap positions temporarily for bond determination only.
        Default is True.

    Returns
    -------
    Chem.rdchem.Mol
        An RDKit molecule with atoms, bonds, and a single conformer containing
        3D coordinates. The conformer has cell bounds stored as properties
        ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi") and an "offset" property
        indicating the starting atom ID.

    Notes
    -----
    The molecule stores the original atom IDs as integer properties on each
    atom. Bond orders are determined by comparing interatomic distances to
    typical bond lengths if `determine_bonds` is True.
    """
    rwmol = Chem.RWMol()
    df_atoms.sort_index(inplace=True)
    offset = df_atoms.index[0].item()
    rwmol.SetIntProp("offset", offset)

    for atom_id, atom_row in df_atoms.iterrows():
        atom = Chem.Atom(atom_row["symbol"])
        atom.SetNoImplicit(True)
        atom.SetIntProp("id", atom_id)
        rwmol.AddAtom(atom)

    if determine_bonds:
        if make_molecule_whole:
            df_atoms_unwrapped = df_atoms
        else:
            df_atoms_unwrapped = unwrap_df_positions_under_pbc(
                df_atoms, df_bonds, cell_bounds
            )
        mean_distances = _calculate_bond_distances(
            df_atoms_unwrapped, df_bonds
        )
        dict_bond_type: dict[int, Chem.rdchem.BondType] = {}
        for bond_type, df_each_bond in df_bonds.groupby("type"):
            first_bond = df_each_bond.iloc[0]
            symbols = tuple(
                df_atoms_unwrapped.loc[
                    [first_bond["atom1"], first_bond["atom2"]], "symbol"
                ].tolist()
            )
            dict_bond_type[bond_type] = get_bond_order(
                symbols,
                mean_distances[bond_type],
            )
    else:
        dict_bond_type = {
            bond_type: Chem.rdchem.BondType.UNSPECIFIED
            for bond_type, _ in df_bonds.groupby("type")
        }

    for _, bond_row in df_bonds.iterrows():
        atom1_idx = (bond_row["atom1"] - offset).item()
        atom2_idx = (bond_row["atom2"] - offset).item()
        rwmol.AddBond(
            atom1_idx,
            atom2_idx,
            order=dict_bond_type[bond_row["type"]],
        )

    conf = Chem.Conformer(df_atoms.shape[0])
    positions = df_atoms.loc[:, COLS_XYZ].values
    conf.SetPositions(positions)
    _set_conformer_cell_bounds(conf, cell_bounds)

    rwmol.AddConformer(conf)
    return rwmol.GetMol()


def MolFromLAMMPSData(
    filepath_data_or_buffer: Union[os.PathLike, str, io.TextIOBase],
    make_molecule_whole: bool = True,
    determine_bonds: bool = True,
) -> Chem.rdchem.Mol:
    """
    Construct an RDKit Mol object from a LAMMPS data file or buffer.

    This function reads atomic and bonding information from a LAMMPS-style
    data file, reconstructs the molecular structure by inferring bond orders
    based on interatomic distances, and returns a corresponding RDKit Mol object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[os.PathLike, str, io.TextIOBase]
        Path to the LAMMPS data file, or a file-like buffer object
        containing the data.
    make_molecule_whole : bool, optional
        If True, unwrap the molecule under periodic boundary conditions
        to make it whole. Default is True.
    determine_bonds : bool, optional
        If True, determine bond orders based on interatomic distances.
        Default is True.

    Returns
    -------
    Chem.rdchem.Mol
        An RDKit Mol object with atoms and inferred bonds, including
        3D coordinates as a single conformer. The conformer stores cell
        bounds as properties.

    See Also
    --------
    MolFromLAMMPSDump : Create molecule from LAMMPS dump file with multiple conformers.
    """

    df_atoms, df_bonds, cell_bounds = load_data(
        filepath_data_or_buffer,
        make_molecule_whole=make_molecule_whole,
        return_bond_info=True,
        return_cell_bounds=True,
    )
    return _mol_from_dataframe_data(
        df_atoms,
        df_bonds,
        cell_bounds,
        determine_bonds=determine_bonds,
        make_molecule_whole=make_molecule_whole,
    )


def _mol_from_dataframe_dump(
    timestep_records: Sequence[tuple[int, pd.DataFrame, CellBounds]],
    mol_template: Chem.rdchem.Mol,
    n_jobs: Optional[int] = None,
    make_molecule_whole: bool = False,
    silent: bool = False,
) -> Chem.rdchem.Mol:
    """
    Construct an RDKit molecule with conformers from a sequence of trajectory records.

    Parameters
    ----------
    timestep_records : Sequence[tuple[int, pd.DataFrame, CellBounds]]
        Sequence of tuples for each frame, where each tuple contains:
        - Frame index or timestep (int)
        - Atom DataFrame (pd.DataFrame) with columns 'x', 'y', 'z'
        - Cell bounds (CellBounds) for the frame
    mol_template : Chem.rdchem.Mol
        RDKit molecule to use as a template. The atom order and bonds will be
        copied to the output molecule.
    n_jobs : int, optional
        Number of parallel jobs for processing frames. If None, uses a single
        process. Default is None.
    make_molecule_whole : bool, optional
        If True, unwrap the molecule coordinates in each frame according to
        periodic boundary conditions. Default is False.
    silent : bool, optional
        If True, suppress progress bar output. Default is False.

    Returns
    -------
    Chem.rdchem.Mol
        An RDKit molecule instance with one conformer per trajectory record.
        Each conformer stores:
        - Cell bounds as double properties: "xlo", "xhi", "ylo", "yhi", "zlo", "zhi"
        - Frame index as integer property: "frame"

    Notes
    -----
    The conformer coordinates are extracted from the atom DataFrame for each frame.
    If `make_molecule_whole` is True, atoms are unwrapped using the molecular graph
    and cell bounds to provide whole molecule coordinates per frame.
    """
    mol = Chem.Mol(mol_template)
    mol.RemoveAllConformers()
    n_atoms = mol.GetNumAtoms()

    graph = nx.from_numpy_array(Chem.GetAdjacencyMatrix(mol))
    assert isinstance(graph, nx.Graph)

    with tqdm_joblib(len(timestep_records), silent=silent, desc="Processing"):
        conformer_positions: tuple[np.ndarray, ...] = tuple(
            Parallel(n_jobs=n_jobs)(
                delayed(_get_conformer_positions)(
                    df_atoms,
                    make_molecule_whole=make_molecule_whole,
                    cell_bounds=cell_bounds if make_molecule_whole else None,
                    graph=graph if make_molecule_whole else None,
                )
                for _, df_atoms, cell_bounds in timestep_records
            )
        )

    assert len(conformer_positions) == len(timestep_records)
    for confId, (positions, (frame, _, cell_bounds)) in enumerate(
        zip(conformer_positions, timestep_records)
    ):
        conf = Chem.Conformer(n_atoms)
        conf.SetPositions(positions)
        conf.SetIntProp("frame", frame)
        conf.SetId(confId)
        _set_conformer_cell_bounds(conf, cell_bounds)
        mol.AddConformer(conf)
    return mol


def MolFromLAMMPSDump(
    filepath_dump: Union[os.PathLike, str],
    mol_template: Chem.rdchem.Mol,
    make_molecule_whole: bool = False,
    select: Optional[Union[int, slice, Sequence[int]]] = None,
    select_by: Literal["timestep", "index"] = "timestep",
    n_jobs: Optional[int] = None,
    silent: bool = False,
) -> Chem.rdchem.Mol:
    """
    Create an RDKit molecule with conformers from a LAMMPS dump file.

    This function loads atom coordinates from a LAMMPS trajectory file and assigns
    them to the provided molecular template as conformers. If specified, the molecule
    can be unwrapped under periodic boundary conditions to make it whole.

    Parameters
    ----------
    filepath_dump : Union[os.PathLike, str]
        Path to the LAMMPS dump file to load.
    mol_template : Chem.rdchem.Mol
        An RDKit molecule used as a template. The returned molecule will copy
        its atom and bond structure.
    make_molecule_whole : bool, optional
        If True, unwrap the molecule based on PBC to make it whole in each frame.
        Default is False.
    select : Optional[Union[int, slice, Sequence[int]]], optional
        Selection criteria for frames. If None, all frames are loaded.
        - If int: select a single frame
        - If slice: select a range of frames
        - If Sequence[int]: select specific frames
        Default is None.
    select_by : Literal["timestep", "index"], optional
        Whether to select by timestep value ("timestep") or index position ("index").
        Default is "timestep".
    n_jobs : int, optional
        Number of parallel jobs for loading the dump. If -1, uses all available CPUs.
        Default is None.
    silent : bool, optional
        If True, suppress progress bar output. Default is False.

    Returns
    -------
    Chem.rdchem.Mol
        An RDKit molecule with one conformer per frame in the LAMMPS dump file.
        Each conformer stores the simulation cell bounds as properties and the
        frame index as an integer property "frame".

    See Also
    --------
    MolFromLAMMPSData : Create molecule from LAMMPS data file with single conformer.
    """
    timestep_records = load_dump(
        filepath_dump,
        select=select,
        select_by=select_by,
        n_jobs=n_jobs,
        return_cell_bounds=True,
        silent=silent,
    )

    return _mol_from_dataframe_dump(
        timestep_records,
        mol_template=mol_template,
        n_jobs=n_jobs,
        make_molecule_whole=make_molecule_whole,
        silent=silent,
    )
