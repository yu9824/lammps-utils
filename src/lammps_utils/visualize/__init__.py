from lammps_utils.helpers import is_installed

if is_installed("py3Dmol"):
    from lammps_utils.visualize._view_3d import view_3d

    __all__ = ["view_3d"]
else:
    raise ImportError("py3Dmol is not installed")
