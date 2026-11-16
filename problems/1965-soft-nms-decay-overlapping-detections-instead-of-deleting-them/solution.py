import numpy as np

def soft_nms(boxes, scores, method="gaussian", iou_threshold=0.3, sigma=0.5, score_threshold=0.001):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).copy()   # work on a copy, never the caller's scores
    if len(scores) == 0:
        return []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    remaining = list(range(len(scores)))
    keep = []
    while remaining:
        # highest current score; on a tie, the lowest index
        best = max(remaining, key=lambda i: (scores[i], -i))
        keep.append((int(best), float(scores[best])))
        remaining.remove(best)
        if not remaining:
            break
        rest = np.array(remaining)
        # IoU of the selected box with every remaining box
        iw = np.maximum(0.0, np.minimum(boxes[best, 2], boxes[rest, 2]) - np.maximum(boxes[best, 0], boxes[rest, 0]))
        ih = np.maximum(0.0, np.minimum(boxes[best, 3], boxes[rest, 3]) - np.maximum(boxes[best, 1], boxes[rest, 1]))
        inter = iw * ih
        union = areas[best] + areas[rest] - inter
        o = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
        # decay scores
        if method == "linear":
            w = np.where(o > iou_threshold, 1.0 - o, 1.0)
        elif method == "gaussian":
            w = np.exp(-(o ** 2) / sigma)
        elif method == "hard":
            w = np.where(o > iou_threshold, 0.0, 1.0)
        else:
            raise ValueError(f"unknown method: {method}")
        scores[rest] *= w
        # drop boxes whose score fell below the threshold
        remaining = [i for i in remaining if scores[i] >= score_threshold]
    return keep
