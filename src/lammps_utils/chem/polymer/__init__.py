from ._builder import attach_terminal_groups, polymerize_linear
from ._cell import calculate_box_length, generate_amorphous_cell
from ._connect import (
    connect_mols,
    get_head_and_tail_from_props,
    has_asterisk,
    has_tritium,
    infer_head_and_tail,
    replace_to_tritium_marker,
    resolve_head_and_tail,
)
from ._polymer import find_main_chains

__all__ = (
    "attach_terminal_groups",
    "calculate_box_length",
    "connect_mols",
    "get_head_and_tail_from_props",
    "infer_head_and_tail",
    "resolve_head_and_tail",
    "has_tritium",
    "has_asterisk",
    "replace_to_tritium_marker",
    "find_main_chains",
    "generate_amorphous_cell",
    "polymerize_linear",
)
