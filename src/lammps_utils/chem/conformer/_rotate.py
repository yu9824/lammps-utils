import numpy as np


def rotate_around_axis(
    points: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """
    Rotate a set of 3D points around a specified axis by a given angle.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) representing N points in 3D space.
    axis : np.ndarray
        Array of shape (3,) representing the rotation axis (does not need to be normalized).
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) of rotated points.

    Notes
    -----
    The rotation is performed using Rodrigues' rotation formula.
    The axis will be normalized internally.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[1, 0, 0]])
    >>> axis = np.array([0, 0, 1])
    >>> angle = np.pi / 2
    >>> rotate_around_axis(pts, axis, angle)
    array([[0., 1., 0.]])
    """
    axis = axis / np.linalg.norm(axis)
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)

    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )

    R = cos_t * np.eye(3) + sin_t * K + (1 - cos_t) * np.outer(axis, axis)

    return points @ R.T


def rotate_around_bond(
    coords: np.ndarray,
    idx_center: int,
    idx_target: int,
    angle: float,
) -> np.ndarray:
    """
    Rotate a set of 3D coordinates around a bond defined by two atom indices.

    Parameters
    ----------
    coords : np.ndarray
        An array of shape (N, 3) representing the Cartesian coordinates
        of N atoms.
    idx_center : int
        The index of the atom to be used as the center of rotation (the pivot point).
    idx_target : int
        The index of the atom that, together with the center, defines the rotation axis.
        The axis is from the center (idx_center) to the target (idx_target).
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    np.ndarray
        The rotated coordinates of shape (N, 3).

    Notes
    -----
    All coordinates are rotated around the axis defined by the two indices,
    with the origin shifted to `idx_center` before rotation and restored after.
    This function rotates all points; to rotate only part of the structure,
    pass only those coordinates.
    """
    # 回転中心（target_idx2）
    center = coords[idx_center]
    coords -= center

    # 回転軸ベクトル（idx2 → target_idx2）
    axis = coords[idx_target]
    axis /= np.linalg.norm(axis)

    # 回転
    coords = rotate_around_axis(coords, axis, angle)

    # 戻す
    coords += center
    return coords
