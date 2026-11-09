"""GT projection using nuPlan Box3D corners, near-plane edge clipping and Shapely.

Matrices use column-vector transforms; boxes follow NAVSIM x,y,z,l,w,h,yaw.
"""
import sys
import numpy as np
from common import IMAGE_SIZE, PATCH_SIZE, ROOT

sys.path.insert(0, str(ROOT / "nuplan-devkit"))
from nuplan.database.utils.boxes.box3d import Box3D
from pyquaternion import Quaternion
from shapely.geometry import box as rectangle
from shapely.ops import unary_union

EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def box_corners(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
        raise ValueError("Invalid NAVSIM box")
    return Box3D(center=box[:3], size=box[[4, 3, 5]],
                 orientation=Quaternion(axis=[0, 0, 1], radians=box[6])).corners().T


def clip_edges(corners, near=0.1):
    if near <= 0:
        raise ValueError("near must be positive")
    segments = []
    for i, j in EDGES:
        a, b = corners[i].copy(), corners[j].copy()
        if a[2] < near and b[2] < near:
            continue
        if a[2] < near or b[2] < near:
            intersection = a + (near - a[2]) / (b[2] - a[2]) * (b - a)
            intersection[2] = near
            if a[2] < near:
                a = intersection
            else:
                b = intersection
        segments.append(np.stack([a, b]))
    return np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)


def project_box(box, intrinsics, lidar_to_camera, near=0.1):
    transform = np.asarray(lidar_to_camera, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    corners = box_corners(box) @ transform[:3, :3].T + transform[:3, 3]
    edges = clip_edges(corners, near)
    if not len(edges):
        return None
    points = edges.reshape(-1, 3) @ intrinsics.T
    pixels = points[:, :2] / points[:, 2:3]
    if not np.isfinite(pixels).all():
        raise ValueError("Non-finite projection")
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    low, high = np.maximum(low, 0), np.minimum(high, IMAGE_SIZE)
    if np.any(high <= low):
        return None
    return {"bbox": [float(low[0]), float(low[1]), float(high[0]), float(high[1])],
            "segments": pixels.reshape(-1, 2, 2).tolist(),
            "near_clipped": bool(np.any(corners[:, 2] < near))}


def patch_distribution(boxes):
    """Exact union area per patch: overlaps counted once, including partial patches."""
    width, height = IMAGE_SIZE
    result = np.zeros((height // PATCH_SIZE, width // PATCH_SIZE), dtype=np.float32)
    if not boxes:
        return result.reshape(-1), False
    union = unary_union([rectangle(*b) for b in boxes]).intersection(rectangle(0, 0, width, height))
    if union.is_empty or union.area <= 0:
        return result.reshape(-1), False
    x0, y0, x1, y1 = union.bounds
    for row in range(max(0, int(y0 // PATCH_SIZE)), min(result.shape[0], int(np.ceil(y1 / PATCH_SIZE)))):
        for col in range(max(0, int(x0 // PATCH_SIZE)), min(result.shape[1], int(np.ceil(x1 / PATCH_SIZE)))):
            cell = rectangle(col * PATCH_SIZE, row * PATCH_SIZE, (col + 1) * PATCH_SIZE, (row + 1) * PATCH_SIZE)
            result[row, col] = union.intersection(cell).area / PATCH_SIZE**2
    total = float(result.sum())
    if total == 0:
        return result.reshape(-1), False
    result /= total
    return result.reshape(-1), True
