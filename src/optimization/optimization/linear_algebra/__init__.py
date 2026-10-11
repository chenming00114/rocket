"""Define linear algebraic structures for optimization problems."""

from optimization.linear_algebra.condensed_matrix import IndexSelectedMatrix, Matrix
from optimization.linear_algebra.slice_utils import SliceType, resolve_slice

__all__ = [
    "IndexSelectedMatrix",
    "Matrix",
    "SliceType",
    "resolve_slice",
]
