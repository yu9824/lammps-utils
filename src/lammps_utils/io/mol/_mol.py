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


def _get_conformer_positions(
    df_atoms: pd.DataFrame,
    make_molecule_whole: bool,
    cell_bounds: Optional[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = None,
    graph: Optional[nx.Graph] = None,
) -> np.ndarray:
    """
    Compute 3D positions (conformer coordinates) from atomic dataframe,
    with optional unwrapping under periodic boundary conditions (PBC).

    Parameters
    ----------
    df_atoms : pd.DataFrame
        DataFrame containing atom positions.
    make_molecule_whole : bool
        Whether to unwrap positions under PBC.
    cell_bounds : tuple, optional
        Cell bounds for each axis.
    graph : nx.Graph, optional
        Graph for unwrapping positions. Required if make_molecule_whole is True.

    Returns
    -------
    np.ndarray
        positions (3D coordinates) array.
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
    cell_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ],
    determine_bonds: bool = True,
    make_molecule_whole: bool = True,
) -> Chem.rdchem.Mol:
    rwmol = Chem.RWMol()
    df_atoms.sort_index(inplace=True)
    offset = df_atoms.index[0].item()
    rwmol.SetIntProp("offset", offset)

    for atom_id, _sr_atom in df_atoms.iterrows():
        atom = Chem.Atom(_sr_atom["symbol"])
        atom.SetNoImplicit(True)
        atom.SetIntProp("id", atom_id)
        rwmol.AddAtom(atom)

    if determine_bonds:
        if make_molecule_whole:
            _df_atoms_unwrapped = df_atoms
        else:
            _df_atoms_unwrapped = unwrap_df_positions_under_pbc(
                df_atoms, df_bonds, cell_bounds
            )
        dict_bond_type: dict[int, Chem.rdchem.BondType] = dict()
        for bond_type, df_each_bond in df_bonds.groupby("type"):
            distances = np.sqrt(
                np.sum(
                    np.square(
                        _df_atoms_unwrapped.loc[
                            df_each_bond.loc[:, "atom1"], COLS_XYZ
                        ].values
                        - _df_atoms_unwrapped.loc[
                            df_each_bond.loc[:, "atom2"], COLS_XYZ
                        ].values
                    ),
                    axis=1,
                )
            )
            symbols = tuple(
                _df_atoms_unwrapped.loc[
                    df_each_bond.iloc[0].loc[["atom1", "atom2"]], "symbol"
                ].tolist()
            )
            dict_bond_type[bond_type] = get_bond_order(
                symbols,
                np.mean(distances).item(),
            )
    else:
        dict_bond_type = {
            bond_type: Chem.rdchem.BondType.UNSPECIFIED
            for bond_type, _ in df_bonds.groupby("type")
        }

    for _, _sr_bond in df_bonds.iterrows():
        rwmol.AddBond(
            (_sr_bond.loc["atom1"] - offset).item(),
            (_sr_bond.loc["atom2"] - offset).item(),
            order=dict_bond_type[_sr_bond["type"]],
        )

    conf = Chem.Conformer(df_atoms.shape[0])
    positions = df_atoms.loc[:, COLS_XYZ].values
    conf.SetPositions(positions)
    for idx_axis, axis in enumerate(COLS_XYZ):
        conf.SetDoubleProp(f"{axis}lo", cell_bounds[idx_axis][0])
        conf.SetDoubleProp(f"{axis}hi", cell_bounds[idx_axis][1])

    rwmol.AddConformer(conf)
    return rwmol.GetMol()


def MolFromLAMMPSData(
    filepath_data_or_buffer: Union[os.PathLike, str, io.TextIOBase],
    make_molecule_whole: bool = True,
    determine_bonds: bool = True,
) -> Chem.rdchem.Mol:
    """
    Constructs an RDKit Mol object from a LAMMPS data file or buffer.

    This function reads atomic and bonding information from a LAMMPS-style
    data file, reconstructs the molecular structure by inferring bond orders
    based on interatomic distances, and returns a corresponding RDKit Mol object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[os.PathLike, str, io.TextIOBase]
        Path to the LAMMPS data file, or a file-like buffer object
        containing the data.

    Returns
    -------
    Chem.rdchem.Mol
        An RDKit Mol object with atoms and inferred bonds, including
        3D coordinates as a single conformer.
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
    timestep_records: Sequence[
        tuple[
            int,  # frame
            pd.DataFrame,
            tuple[  # cell_bounds
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
            ],
        ],
        ...,
    ],
    mol_template: Chem.rdchem.Mol,
    n_jobs: Optional[int] = None,
    make_molecule_whole: bool = False,
    silent: bool = False,
) -> Chem.rdchem.Mol:
    """
    Construct an RDKit molecule with conformers from a sequence of trajectory records.

    Parameters
    ----------
    timestep_records : Sequence[Tuple[int, pd.DataFrame, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]]
        List of tuples for each frame, where each tuple contains
        (frame index or timestep, atom DataFrame, cell bounds).
    mol_template : rdkit.Chem.rdchem.Mol
        RDKit molecule to use as a template. The atom order and bonds will be copied.
    n_jobs : int, optional
        Number of jobs to use for parallel processing. If None, uses a single process.
    make_molecule_whole : bool, optional
        If True, unwrap the molecule coordinates in each frame according to periodic boundary conditions.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        An RDKit molecule instance with one conformer per trajectory record. Each conformer
        stores its cell bounds information in double properties, and its corresponding frame
        or timestep as an integer property "frame".

    Notes
    -----
    The conformer coordinates are extracted from the atom DataFrame for each frame.
    If `make_molecule_whole` is True, atoms are unwrapped using the molecular graph
    and cell bounds to provide whole molecule coordinates per frame. Each conformer also
    has cell bounds attached as double properties: Xlo/Xhi, Ylo/Yhi, Zlo/Zhi.
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
        for index_axis, axis in enumerate(COLS_XYZ):
            conf.SetDoubleProp(f"{axis}lo", cell_bounds[index_axis][0])
            conf.SetDoubleProp(f"{axis}hi", cell_bounds[index_axis][1])
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
    ----------------
    filepath_dump : Union[os.PathLike, str]
        Path to the LAMMPS dump file to load.
    mol_template : Chem.rdchem.Mol
        An RDKit molecule used as a template. The returned molecule will copy
        its atom and bond structure.
    make_molecule_whole : bool
        If True, unwrap the molecule based on PBC to make it whole in each frame.
    select : Optional[Union[int, slice, Sequence[int]]], optional
        Selection criteria for frames. If None, all frames are loaded.
        - If int: select a single frame
        - If slice: select a range of frames
        - If Sequence[int]: select specific frames
    select_by : Literal["timestep", "index"], optional
        Whether to select by timestep value ("timestep") or index position ("index").
        Default is "timestep".
    n_jobs : int, optional
        Number of parallel jobs for loading the dump. -1 uses all available CPUs.

    Returns
    ----------------
    Chem.rdchem.Mol
        An RDKit molecule with one conformer per frame in the LAMMPS dump file.
        Each conformer stores the simulation cell bounds as properties.
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
