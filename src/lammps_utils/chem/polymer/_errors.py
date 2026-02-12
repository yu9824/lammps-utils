"""Custom exceptions for the polymer module."""

from __future__ import annotations


class TacticityError(ValueError):
    """
    Raised when tacticity (stereoregularity) is invalid or validation fails.

    This includes:
    - Unsupported combination (e.g. syndiotactic with forcefield minimization,
      or tacticity with random_walk).
    - Invalid tacticity value.
    - Post-build validation failure: main-chain handedness does not match
      the expected pattern for the requested tacticity (isotactic, syndiotactic,
      or atactic).
    """

    def __init__(
        self,
        message: str,
        *,
        expected_tacticity: str | None = None,
        handedness: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.expected_tacticity = expected_tacticity
        self.handedness = handedness
