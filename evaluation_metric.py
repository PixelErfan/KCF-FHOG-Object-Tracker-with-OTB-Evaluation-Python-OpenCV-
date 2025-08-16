import numpy as np

def region_to_bbox(region, center=True):
    n = len(region)
    if n == 4:
        return _rect(region, center)
    elif n == 8:
        return _poly(region, center)
    else:
        raise ValueError("Region must have 4 or 8 values.")

def _rect(region, center):
    region = np.asarray(region, dtype=float)
    if center:
        x, y, w, h = region
        cx = x + w / 2.0
        cy = y + h / 2.0
        return np.array([cx, cy, w, h], dtype=float)
    else:
        return region.astype(float)

def _poly(region, center):
    region = np.asarray(region, dtype=float)
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / (A2 + 1e-12))
    w = s * (x2 - x1) + 1.0
    h = s * (y2 - y1) + 1.0
    if center:
        return np.array([cx, cy, w, h], dtype=float)
    else:
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=float)

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0] + boxA[2] / 2.0, boxA[1] + boxA[3] / 2.0), dtype=float)
    b = np.array((boxB[0] + boxB[2] / 2.0, boxB[1] + boxB[3] / 2.0), dtype=float)
    return float(np.linalg.norm(a - b))

def overlap_ratio(rect1, rect2):
    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])
    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect + 1e-12
    iou = intersect / union
    return np.clip(iou, 0.0, 1.0)

def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0.02, 1.02, 0.02)
    success = np.zeros(len(thresholds_overlap), dtype=float)
    iou = np.ones(len(gt_bb), dtype=float) * (-1.0)
    mask = np.sum(gt_bb > 0, axis=1) == 4
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i, thr in enumerate(thresholds_overlap):
        success[i] = np.sum(iou >= thr) / float(n_frame)
    return success

def compile_results(gt, bboxes):
    l = int(np.size(bboxes, 0))
    gt4 = np.zeros((l, 4), dtype=float)
    new_distances = np.zeros(l, dtype=float)
    distance_thresholds = np.linspace(1, 50, 50)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])

    precision_curve = np.array([np.mean(new_distances < t) for t in distance_thresholds], dtype=float)
    dp_20 = float(precision_curve[19])  # threshold = 20 px
    average_cle = float(new_distances.mean())

    success_curve = success_overlap(gt4, bboxes, l)
    auc = float(np.mean(success_curve))

    return precision_curve, success_curve, average_cle, auc, dp_20