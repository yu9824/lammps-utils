import io
import os
from collections.abc import Sequence
from typing import Literal, Optional, Union

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem

from lammps_utils.chem.bond._bond import get_bond_order
from lammps_utils.constants import COLS_XYZ
from lammps_utils.graph.pbc._pbc import unwrap_positions_under_pbc
from lammps_utils.helpers import is_installed
from lammps_utils.io.dataframe._dataframe import load_data, load_dump
from lammps_utils.logging import get_child_logger

if is_installed("tqdm"):
    from tqdm.auto import tqdm
else:
    from lammps_utils.helpers import dummy_tqdm as tqdm

logger = get_child_logger(__name__)


def _prepare_conformer_data(
    frame: int,
    df_atoms,
    cell_bounds,
    confId: int,
    make_molecule_whole: bool,
    graph: Optional[nx.Graph] = None,
) -> tuple[np.ndarray, int, int, dict[str, tuple[float, float]]]:
    """
    Prepare conformer data from frame data.

    Parameters
    ----------
    frame : int
        Frame number.
    df_atoms
        DataFrame containing atom positions.
    cell_bounds
        Cell bounds for each axis.
    confId : int
        Conformer ID.
    make_molecule_whole : bool
        Whether to unwrap positions under PBC.
    graph : nx.Graph, optional
        Graph for unwrapping positions. Required if make_molecule_whole is True.

    Returns
    -------
    tuple[np.ndarray, int, int, dict[str, tuple[float, float]]]
        Tuple containing:
        - positions: Final positions array
        - frame: Frame number
        - confId: Conformer ID
        - cell_props: Dictionary with axis names as keys and (lo, hi) tuples as values
    """
    df_atoms.sort_index(inplace=True)
    positions = df_atoms.loc[:, COLS_XYZ].values

    cell_props = {}
    for idx_axis, axis in enumerate(COLS_XYZ):
        cell_props[axis] = (cell_bounds[idx_axis][0], cell_bounds[idx_axis][1])

    if make_molecule_whole:
        if graph is None:
            raise ValueError(
                "graph is required when make_molecule_whole is True"
            )
        cell_size = tuple(
            cell_props[axis][1] - cell_props[axis][0] for axis in COLS_XYZ
        )
        positions = unwrap_positions_under_pbc(
            graph, positions=positions, cell_size=cell_size
        )

    return positions, frame, confId, cell_props


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
            _df_atoms_unwrapped = load_data(
                filepath_data_or_buffer, make_molecule_whole=True
            ).sort_index()
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


def MolFromLAMMPSDump(
    filepath_dump: Union[os.PathLike, str],
    mol_template: Chem.rdchem.Mol,
    make_molecule_whole: bool = False,
    select: Optional[Union[int, slice, Sequence[int]]] = None,
    select_by: Literal["timestep", "index"] = "timestep",
    n_jobs: Optional[int] = None,
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
    )

    # logging
    logger.info(
        "Successfully loaded the dump file. Converting to 'rdkit.Chem.rdchem.Mol'."
    )

    mol = Chem.Mol(mol_template)
    mol.RemoveAllConformers()
    n_atoms = mol.GetNumAtoms()

    graph = nx.from_numpy_array(Chem.GetAdjacencyMatrix(mol))
    assert isinstance(graph, nx.Graph)

    conformer_data_results = Parallel(n_jobs=n_jobs)(
        delayed(_prepare_conformer_data)(
            frame,
            df_atoms,
            cell_bounds,
            confId,
            make_molecule_whole=make_molecule_whole,
            graph=graph if make_molecule_whole else None,
        )
        for confId, (frame, df_atoms, cell_bounds) in enumerate(
            timestep_records
        )
    )

    for positions, frame, confId, cell_props in tqdm(
        conformer_data_results, desc="AddConformer"
    ):
        conf = Chem.Conformer(n_atoms)
        conf.SetPositions(positions)
        conf.SetIntProp("frame", frame)
        conf.SetId(confId)
        for axis in COLS_XYZ:
            conf.SetDoubleProp(f"{axis}lo", cell_props[axis][0])
            conf.SetDoubleProp(f"{axis}hi", cell_props[axis][1])
        mol.AddConformer(conf)
    return mol
