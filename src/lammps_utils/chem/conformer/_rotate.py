import numpy as np


def rotate_around_axis(
    points: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rotate points around an axis passing through the origin
    using Rodrigues' rotation formula.

    Parameters
    ----------
    points : (N, 3) np.ndarray
        Coordinates to be rotated. Each row is a 3D vector.
    axis : (3,) np.ndarray
        Rotation axis direction (need not be normalized).
        The axis is assumed to pass through the origin.
    angle : float
        Rotation angle in radians (right-hand rule).

    Returns
    -------
    (N, 3) np.ndarray
        Rotated points.
    """
    points = np.asarray(points, dtype=float)
    axis = np.asarray(axis, dtype=float)
    angle = ensure_float(angle)

    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("Rotation axis must be a non-zero vector.")
    u = axis / norm  # unit axis

    cos_t = np.cos(angle)
    sin_t = np.sin(angle)

    # Rodrigues' rotation formula (vector form)
    ux_p = np.cross(u, points)  # u × p
    u_dot_p = np.dot(points, u)  # u · p

    rotated = (
        points * cos_t + ux_p * sin_t + np.outer(u_dot_p, u) * (1.0 - cos_t)
    )

    return rotated


def rotate_around_bond(
    coords: np.ndarray,
    idx_center: int,
    idx_target: int,
    angle: float,
) -> np.ndarray:
    """
    Rotate coordinates around an axis defined by idx_center -> idx_target
    using Rodrigues' rotation formula.

    Parameters
    ----------
    coords : (N, 3) np.ndarray
        Cartesian coordinates.
    idx_center : int
        Index of the atom through which the rotation axis passes.
    idx_target : int
        Index of the atom defining the axis direction.
    angle : float
        Rotation angle in radians (right-hand rule).

    Returns
    -------
    (N, 3) np.ndarray
        Rotated coordinates.
    """
    coords = np.asarray(coords, dtype=float)

    # --- 1. shift origin to idx_center ---
    center = coords[idx_center]
    rel = coords - center

    # --- 2. define rotation axis direction ---
    axis = coords[idx_target] - center
    if np.linalg.norm(axis) == 0.0:
        raise ValueError(
            "idx_center and idx_target define a zero-length axis."
        )

    # --- 3. rotate around axis through origin ---
    rel_rot = rotate_around_axis(rel, axis, angle)

    # --- 4. restore origin ---
    coords_rot = rel_rot + center

    # 数値誤差対策（理論上は不要）
    assert np.allclose(coords_rot[idx_center], coords[idx_center])
    # coords_rot[idx_center] = coords[idx_center]

    return coords_rot


def ensure_float(x) -> float:
    """
    Convert the input to a float if possible.

    Parameters
    ----------
    x : any
        The value to convert to float.

    Returns
    -------
    float
        The converted float value.

    Raises
    ------
    TypeError
        If `x` is of type `bool` or cannot be converted to float.
    ValueError
        If `x` cannot be converted to float due to a value error.

    Notes
    -----
    Boolean values are not accepted and will raise a TypeError.
    """
    try:
        return float(x)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Cannot convert {x!r} to float") from e
