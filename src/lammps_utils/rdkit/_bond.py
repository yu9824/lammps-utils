from rdkit import Chem

from lammps_utils.logging import get_child_logger

_logger = get_child_logger(__name__)


def get_bond_order(
    atom_symbols: tuple[str, str], bond_length: float
) -> Chem.rdchem.BondType:
    """
    Estimate bond order based on atom symbols and bond length.

    Parameters
    ----------
    atom_symbols : tuple of str
        A tuple containing the atomic symbols of the two bonded atoms (e.g., ("C", "O")).
    bond_length : float
        The bond length in angstroms.

    Returns
    -------
    rdkit.Chem.rdchem.BondType
        The estimated bond type (SINGLE, DOUBLE, TRIPLE, or AROMATIC).
    """
    SINGLE_BOND_ELEMENTS = {"H", "F", "Cl", "Br", "I"}
    ALL_ELEMENTS_USED = {"C", "N", "O", "P", "S", "H", "F", "Cl", "Br", "I"}

    symbol1, symbol2 = sorted(atom_symbols)

    if symbol1 in SINGLE_BOND_ELEMENTS or symbol2 in SINGLE_BOND_ELEMENTS:
        return Chem.rdchem.BondType.SINGLE

    if (symbol1, symbol2) == ("C", "C"):
        # C–C bond
        if bond_length >= 1.41:  # sp3–sp3, sp3–sp2, sp3–sp
            return Chem.rdchem.BondType.SINGLE
        elif 1.39 <= bond_length < 1.41:  # aromatic (e.g., ca–ca)
            return Chem.rdchem.BondType.AROMATIC
        elif 1.33 <= bond_length < 1.39:  # sp2–sp2
            return Chem.rdchem.BondType.DOUBLE
        elif 1.20 <= bond_length < 1.33:  # sp–sp2 or sp–sp3
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.20, sp–sp
            return Chem.rdchem.BondType.TRIPLE

    elif (symbol1, symbol2) == ("C", "N"):
        # C–N bond
        if bond_length >= 1.34:  # sp3–sp3, sp3–sp2, sp3–sp
            return Chem.rdchem.BondType.SINGLE
        elif 1.29 <= bond_length < 1.34:  # sp2–sp2
            return Chem.rdchem.BondType.DOUBLE
        else:  # <1.29, sp–sp
            return Chem.rdchem.BondType.TRIPLE

    elif (symbol1, symbol2) == ("C", "O"):
        # C–O bond
        if bond_length >= 1.22:  # sp3–sp3, sp3–sp2, sp3–sp
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.22, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("N", "N"):
        # N–N bond
        if bond_length >= 1.30:  # sp3–sp3, sp3–sp2, sp3–sp
            return Chem.rdchem.BondType.SINGLE
        elif 1.16 < bond_length < 1.30:  # sp2–sp2
            return Chem.rdchem.BondType.DOUBLE
        else:  # ≤1.16, sp–sp
            return Chem.rdchem.BondType.TRIPLE

    elif (symbol1, symbol2) == ("N", "O"):
        # N–O bond
        if bond_length >= 1.25:  # sp3–sp3, sp3–sp2, sp3–sp
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.25, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("O", "O"):
        # O–O bond
        if bond_length >= 1.44:  # sp3–sp3
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.44, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("C", "P"):
        # C–P bond
        if bond_length >= 1.70:  # sp3–sp3, sp3–sp2
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.70, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("N", "P"):
        # N–P bond
        if bond_length >= 1.65:  # sp3–sp3, sp3–sp2
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.65, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("O", "P"):
        # O–P bond
        if bond_length >= 1.53:  # sp3–sp3, sp3–sp2
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.53, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("P", "P"):
        # P–P bond
        if bond_length >= 1.80:  # sp3–sp3
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.80, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("C", "S"):
        # C–S bond
        if bond_length >= 1.64:  # sp3–sp3
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.64, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("N", "S"):
        # N–S bond
        if bond_length >= 1.58:  # sp3–sp3
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.58, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("O", "S"):
        # O–S bond
        if bond_length >= 1.56:  # sp3–sp3
            return Chem.rdchem.BondType.SINGLE
        else:  # <1.56, sp2–sp2
            return Chem.rdchem.BondType.DOUBLE

    elif (symbol1, symbol2) == ("S", "S"):
        # S–S bond (usually single)
        return Chem.rdchem.BondType.SINGLE

    elif symbol1 in ALL_ELEMENTS_USED and symbol2 in ALL_ELEMENTS_USED:
        _logger.warning(
            f"{symbol1}-{symbol2} is not in {ALL_ELEMENTS_USED}. "
            "Use BondType.UNSPECIFIED."
        )
        return Chem.rdchem.BondType.UNSPECIFIED
    else:
        # Default fallback
        return Chem.rdchem.BondType.SINGLE
