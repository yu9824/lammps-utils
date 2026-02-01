"""This package contains helper functions and classes that are used throughout the project.

The functions and classes in this package are designed to be reusable and modular,
making it easy to incorporate them into different parts of the project. These helpers
are intended to simplify common tasks, such as checking if a package is installed,
iterating with a dummy progress bar, and verifying function arguments.

"""

from ._helpers import (
    check_encoding,
    dummy_tqdm,
    in_directory,
    is_argument,
    is_installed,
    run_executable,
    tqdm_joblib,
)

# _helpers.pyだと、_が入っているのでドキュメント化されない。
# ドキュメント化したい場合は、モジュールメソッドとして登録するため、__all__に入れる。
__all__ = (
    "check_encoding",
    "dummy_tqdm",
    "in_directory",
    "is_argument",
    "is_installed",
    "tqdm_joblib",
    "run_executable",
)
