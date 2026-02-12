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
from ._errors import TacticityError
from ._handedness import compute_main_chain_handedness, improper_handedness
from ._polymer import find_main_chains

__all__ = (
    "attach_terminal_groups",
    "calculate_box_length",
    "TacticityError",
    "connect_mols",
    "get_head_and_tail_from_props",
    "infer_head_and_tail",
    "compute_main_chain_handedness",
    "improper_handedness",
    "resolve_head_and_tail",
    "generate_amorphous_cell",
    "polymerize_linear",
    "has_tritium",
    "has_asterisk",
    "find_main_chains",
    "replace_to_tritium_marker",
)
