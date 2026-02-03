from ._builder import attach_terminal_groups, polymerize_linear
from ._cell import calculate_box_length, generate_amorphous_cell
from ._connect import (
    connect_mols,
    detect_head_and_tail,
    get_head_and_tail,
    has_asterisk,
    has_tritium,
    replace_to_tritium_marker,
)
from ._polymer import find_main_chains

__all__ = (
    "attach_terminal_groups",
    "calculate_box_length",
    "connect_mols",
    "detect_head_and_tail",
    "get_head_and_tail",
    "has_tritium",
    "has_asterisk",
    "replace_to_tritium_marker",
    "find_main_chains",
    "generate_amorphous_cell",
    "polymerize_linear",
)
