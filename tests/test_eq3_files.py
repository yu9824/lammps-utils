"""Tests for loading eq3_last.data and eq3_last.dump files."""

from pathlib import Path

import pytest

from lammps_utils.io.dataframe import load_data, load_dump


@pytest.fixture
def data_file():
    """Path to eq3_last.data file."""
    return Path(__file__).parent.parent / "examples" / "eq3_last.data"


@pytest.fixture
def dump_file():
    """Path to eq3_last.dump file."""
    return Path(__file__).parent.parent / "examples" / "eq3_last.dump"


def test_load_data_basic(data_file):
    """Test basic loading of eq3_last.data file."""
    df = load_data(data_file)

    # Check that DataFrame is returned
    assert df is not None
    assert len(df) > 0

    # Check expected number of atoms (336 from the file header)
    assert len(df) == 336

    # Check that essential columns exist
    assert "id" == df.index.name
    assert "x" in df.columns
    assert "y" in df.columns
    assert "z" in df.columns


def test_load_data_with_bond_info(data_file):
    """Test loading data file with bond information."""
    df, bonds = load_data(data_file, return_bond_info=True)

    # Check that DataFrame is returned
    assert df is not None
    assert len(df) == 336

    # Check that bonds DataFrame is returned
    assert bonds is not None
    assert len(bonds) > 0

    # Check bond DataFrame columns
    assert "id" == bonds.index.name
    assert "type" in bonds.columns
    assert "atom1" in bonds.columns
    assert "atom2" in bonds.columns


def test_load_data_with_cell_bounds(data_file):
    """Test loading data file with cell bounds."""
    df, cell_bounds = load_data(data_file, return_cell_bounds=True)

    # Check that DataFrame is returned
    assert df is not None
    assert len(df) == 336

    # Check cell bounds structure
    assert cell_bounds is not None
    assert len(cell_bounds) == 3  # x, y, z
    for axis_bounds in cell_bounds:
        assert len(axis_bounds) == 2  # min, max


def test_load_data_with_all_options(data_file):
    """Test loading data file with all options enabled."""
    df, bonds, cell_bounds = load_data(
        data_file, return_bond_info=True, return_cell_bounds=True
    )

    # Check that all returns are valid
    assert df is not None
    assert len(df) == 336
    assert bonds is not None
    assert len(bonds) > 0
    assert cell_bounds is not None
    assert len(cell_bounds) == 3


def test_load_dump_basic(dump_file):
    """Test basic loading of eq3_last.dump file."""
    timesteps = load_dump(dump_file)

    # Check that timesteps are returned
    assert timesteps is not None
    assert len(timesteps) > 0

    # Check structure: each element should be (timestep, DataFrame)
    for timestep, df in timesteps:
        assert isinstance(timestep, int)
        assert df is not None
        assert len(df) > 0

        # Check expected number of atoms (336 from the file header)
        assert len(df) == 336

        # Check that essential columns exist
        assert "id" == df.index.name
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns


def test_load_dump_with_cell_bounds(dump_file):
    """Test loading dump file with cell bounds."""
    timesteps = load_dump(dump_file, return_cell_bounds=True)

    # Check that timesteps are returned
    assert timesteps is not None
    assert len(timesteps) > 0

    # Check structure: each element should be (timestep, DataFrame, cell_bounds)
    for timestep, df, cell_bounds in timesteps:
        assert isinstance(timestep, int)
        assert df is not None
        assert len(df) == 336
        assert cell_bounds is not None
        assert len(cell_bounds) == 3


def test_load_dump_select_timestep(dump_file):
    """Test loading specific timestep from dump file."""
    # Select the first timestep (5000000 from the file)
    timesteps = load_dump(dump_file, select=5000000, select_by="timestep")

    assert timesteps is not None
    assert len(timesteps) == 1

    timestep, df = timesteps[0]
    assert timestep == 5000000
    assert len(df) == 336


def test_data_dump_consistency(data_file, dump_file):
    """Test consistency between data and dump files."""
    # Load data file
    df_data = load_data(data_file)

    # Load dump file
    timesteps = load_dump(dump_file)
    assert len(timesteps) > 0
    _, df_dump = timesteps[0]

    # Both should have the same number of atoms
    assert len(df_data) == len(df_dump) == 336

    # Both should have id column
    assert "id" == df_data.index.name
    assert "id" == df_dump.index.name

    # Check that atom IDs match (sorted for comparison)
    ids_data = sorted(df_data.index.values)
    ids_dump = sorted(df_dump.index.values)
    assert ids_data == ids_dump
