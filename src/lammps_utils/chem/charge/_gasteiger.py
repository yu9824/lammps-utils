from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges


def compute_gasteiger_charges(mol: Chem.Mol) -> tuple[float, ...]:
    """
    Compute Gasteiger charges for a molecule.
    """
    ComputeGasteigerCharges(mol)
    return tuple(
        atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()
    )


if __name__ == "__main__":
    charges = compute_gasteiger_charges(
        Chem.MolFromSmiles("[3H]CC(c1ccccc1)[3H]")
    )
    print(charges)
