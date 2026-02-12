import py3Dmol
from rdkit import Chem


def view_3d(
    mol: Chem.rdchem.Mol, show_atom_index: bool = False
) -> py3Dmol.view:
    """
    Render a 3D visualization of a molecule using py3Dmol.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit molecule object with 3D coordinates.
    show_atom_index : bool, optional
        Whether to show the atom index on the molecule. Default is False.

    Returns
    -------
    py3Dmol.view
        py3Dmol view object for interactive 3D visualization.
        Call `.show()` in a Jupyter notebook to display.
    """
    view = py3Dmol.view(width="100%")
    view.addModel(Chem.MolToMolBlock(mol), "sdf", {"keepH": True})

    view.setStyle({"stick": {"radius": 0.25}, "sphere": {"scale": 0.35}})

    if show_atom_index:
        view.addPropertyLabels(
            "index",  # 0-based atom index
            "",
            {
                "fontSize": 12,
                "fontColor": "black",
                "backgroundColor": "white",
                "showBackground": True,
                "backgroundOpacity": 0.6,
            },
        )

    view.zoomTo()
    return view
