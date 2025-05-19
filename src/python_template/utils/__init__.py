"""This package contains utility functions and classes that are used throughout the project.

The functions and classes in this package are designed to be reusable and modular,
making it easy to incorporate them into different parts of the project. These utilities
are intended to simplify common tasks, such as checking if a package is installed,
iterating with a dummy progress bar, and verifying function arguments.

"""

from ._utils import dummy_tqdm, is_argument, is_installed

# _utils.pyだと、_が入っているのでドキュメント化されない。
# ドキュメント化したい場合は、モジュールメソッドとして登録するため、__all__に入れる。
__all__ = ("dummy_tqdm", "is_argument", "is_installed")
