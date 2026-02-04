import py3Dmol
from rdkit import Chem


def view_3d(mol: Chem.rdchem.Mol) -> py3Dmol.view:
    """
    Render a 3D visualization of a molecule using py3Dmol.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit molecule object with 3D coordinates.

    Returns
    -------
    py3Dmol.view
        py3Dmol view object for interactive 3D visualization.
        Call `.show()` in a Jupyter notebook to display.
    """
    view = py3Dmol.view(width="100%")
    view.addModel(Chem.MolToMolBlock(mol), "sdf", {"keepH": True})
    view.setStyle({"stick": {"radius": 0.25}, "sphere": {"scale": 0.35}})
    return view
