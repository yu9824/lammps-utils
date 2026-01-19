import numpy as np


def rotate_around_axis(
    points: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """
    points: (N, 3)
    axis: (3,) 正規化済み
    angle: rad
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
