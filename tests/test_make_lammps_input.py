from lammps_utils.chem.charge import compute_gasteiger_charges
from lammps_utils.chem.conformer import generate_minimized_conformer
from lammps_utils.chem.polymer import attach_terminal_groups, polymerize_linear
from lammps_utils.io.moltemplate import write_lammps_input


def test_make_lammps_input():
    mol_mono = generate_minimized_conformer(
        "[3H]CC(c1ccccc1)[3H]", seed=1, num_confs=1
    )

    mol_polymer = polymerize_linear((mol_mono,), ratio=(1.0,), seed=1)

    mol_ter = generate_minimized_conformer("[3H]C", seed=1, num_confs=1)

    mol_polymer = attach_terminal_groups(mol_polymer, mol_ter)

    charges = compute_gasteiger_charges(mol_polymer)

    write_lammps_input(
        [(mol_polymer, 10)],
        charges=charges,
        density=0.3,
        seed=1,
        work_in_cwd=False,
    )
