"""Module for generating and minimizing molecular conformers."""

from array import array
from typing import Literal, Optional, Union

from rdkit import Chem
from rdkit.Chem import (
    rdDistGeom,
)
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties,
    MMFFHasAllMoleculeParams,
    UFFGetMoleculeForceField,
)
from rdkit.Chem.rdmolops import AddHs
from rdkit.ForceField.rdForceField import ForceField


def generate_minimized_conformer(
    smiles_or_mol: Union[str, Chem.Mol],
    forcefield: Literal["MMFF", "UFF"] = "MMFF",
    max_iters: int = 500,
    max_attempts: int = 1000,
    num_confs: int = 10,
    seed: Optional[int] = None,
) -> Chem.Mol:
    """
    Generate multiple conformers from a SMILES string, minimize them,
    and return the molecule with the lowest energy conformer.

    Parameters
    ----------
    smiles_or_mol : Union[str, Chem.Mol]
        SMILES string or RDKit Mol object to generate conformers from.
    forcefield : Literal["MMFF", "UFF"], optional
        Force field used for geometry optimization. Default is "MMFF".
    max_iters : int, optional
        Maximum number of iterations for geometry minimization.
        Default is 500.
    max_attempts : int, optional
        Maximum number of attempts for 3D conformer generation.
        Default is 1000.
    num_confs : int, optional
        Number of conformers to generate and minimize. Default is 10.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Chem.Mol
        RDKit Mol object with a single conformer of minimized lowest energy.

    Raises
    ------
    ValueError
        If the SMILES is invalid or forcefield is not supported.
    RuntimeError
        If conformer generation or minimization fails.
    """
    mol, conf_ids = generate_conformers_from_smiles(
        smiles_or_mol,
        max_attempts=max_attempts,
        num_confs=num_confs,
        seed=seed,
    )

    min_energy = float("inf")
    best_conf_id = -1
    energies = array("f", [0.0 for _ in range(len(conf_ids))])
    for conf_id in conf_ids:
        energy = minimize_conformer(
            mol, forcefield=forcefield, max_iters=max_iters, conf_id=conf_id
        )
        energies[conf_id] = energy
        if energy < min_energy:
            min_energy = energy
            best_conf_id = conf_id

    # 最小エネルギーの conformer だけ残す
    all_conf_ids = array("I", [conf.GetId() for conf in mol.GetConformers()])
    for cid in all_conf_ids:
        if cid != best_conf_id:
            mol.RemoveConformer(cid)

    return mol


def generate_conformers_from_smiles(
    smiles_or_mol: Union[str, Chem.Mol],
    max_attempts: int = 1000,
    num_confs: int = 10,
    seed: Optional[int] = None,
) -> tuple[Chem.Mol, tuple[int, ...]]:
    """
    Generate multiple conformers from a SMILES string or RDKit Mol object.

    Parameters
    ----------
    smiles_or_mol : Union[str, Chem.Mol]
        SMILES string or RDKit Mol object to generate conformers from.
    max_attempts : int, optional
        Maximum number of attempts for 3D conformer generation.
        Default is 1000.
    num_confs : int, optional
        Number of conformers to generate. Default is 10.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    tuple[Chem.Mol, tuple[int, ...]]
        A tuple containing:
        - The RDKit Mol object with generated conformers
        - A tuple of conformer IDs

    Raises
    ------
    ValueError
        If the input SMILES is invalid or cannot be converted to a Mol object.
    TypeError
        If the input is neither a SMILES string nor an RDKit Mol object.
    RuntimeError
        If conformer generation fails after the maximum number of attempts.
    """
    if isinstance(smiles_or_mol, str):
        smiles = smiles_or_mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = AddHs(mol)
    elif isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
        if Chem.NeedsHs(mol):
            mol = AddHs(mol)
    else:
        raise TypeError(
            "Input must be a SMILES string or an RDKit Mol object."
        )

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed if seed is not None else -1
    if hasattr(params, "maxIterations"):
        params.maxIterations = max_attempts
    elif hasattr(params, "maxAttempts"):
        params.MaxAttempts = max_attempts

    conf_ids = rdDistGeom.EmbedMultipleConfs(
        mol, numConfs=num_confs, params=params
    )
    if not conf_ids:
        raise RuntimeError("Failed to generate conformers.")
    return mol, tuple(conf_ids)


def minimize_conformer(
    mol: Chem.Mol,
    forcefield: Literal["MMFF", "UFF"] = "MMFF",
    max_iters: int = 500,
    conf_id: int = -1,
) -> float:
    """Minimize the energy of a conformer using the specified force field.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit Mol object with a conformer to minimize.
    forcefield : Literal["MMFF", "UFF"], optional
        Force field to use for minimization. Supported values are "MMFF" and "UFF".
        Default is "MMFF".
    max_iters : int, optional
        Maximum number of iterations for minimization. Default is 500.
    conf_id : int, optional
        Conformer ID to minimize. Default is -1 (minimizes the last conformer).

    Returns
    -------
    float
        The minimized energy of the conformer.

    Raises
    ------
    RuntimeError
        If the force field parameters are missing or minimization fails.
    ValueError
        If an unsupported force field is specified.
    """
    ff: ForceField
    if forcefield == "MMFF":
        if not MMFFHasAllMoleculeParams(mol):
            raise RuntimeError("MMFF parameters missing.")
        ff = MMFFGetMoleculeForceField(
            mol, MMFFGetMoleculeProperties(mol), confId=conf_id
        )
    elif forcefield == "UFF":
        ff = UFFGetMoleculeForceField(mol, confId=conf_id)
    else:
        raise ValueError(f"Unsupported force field: {forcefield}")

    ff.Initialize()
    ff.Minimize(maxIts=max_iters)
    return ff.CalcEnergy()


if __name__ == "__main__":
    print(generate_minimized_conformer("CCO"))
