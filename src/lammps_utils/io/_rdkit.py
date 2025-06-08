import io
import os
from typing import Optional, Union

import networkx as nx
import numpy as np
from rdkit import Chem

from lammps_utils.graph._pbc import unwrap_molecule_under_pbc
from lammps_utils.io._load import load_data, load_dump
from lammps_utils.rdkit._bond import get_bond_order

COLS_XYZ = ["x", "y", "z"]


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
    return Chem.RemoveHs(
        rwmol.GetMol(),
        implicitOnly=True,
        updateExplicitCount=True,
        sanitize=True,
    )


def MolFromLAMMPSDump(
    filepath_dump: Union[os.PathLike, str],
    mol_template: Chem.rdchem.Mol,
    make_molecule_whole: bool = False,
    n_jobs: Optional[int] = None,
) -> Chem.rdchem.Mol:
    tup_results = load_dump(
        filepath_dump, n_jobs=n_jobs, return_cell_bounds=True
    )
    mol = Chem.Mol(mol_template)
    mol.RemoveAllConformers()
    n_atoms = mol.GetNumAtoms()

    graph = nx.from_numpy_array(Chem.GetAdjacencyMatrix(mol))
    assert isinstance(graph, nx.Graph)
    for confId, (frame, df_atoms, cell_bounds) in enumerate(tup_results):
        df_atoms.sort_index(inplace=True)
        conf = Chem.Conformer(n_atoms)
        conf.SetPositions(df_atoms.loc[:, COLS_XYZ].values)
        conf.SetIntProp("frame", frame)
        conf.SetId(confId)
        for idx_axis, axis in enumerate(COLS_XYZ):
            conf.SetDoubleProp(f"{axis}lo", cell_bounds[idx_axis][0])
            conf.SetDoubleProp(f"{axis}hi", cell_bounds[idx_axis][1])

        if make_molecule_whole:
            cell_size = tuple(
                conf.GetDoubleProp(f"{axis}hi")
                - conf.GetDoubleProp(f"{axis}lo")
                for axis in COLS_XYZ
            )

            conf.SetPositions(
                unwrap_molecule_under_pbc(
                    graph, positions=conf.GetPositions(), cell_size=cell_size
                )
            )
        mol.AddConformer(conf)
    return mol
