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


"""

import numpy as np
# NumPy is used for array and IoU calculations.


def nms(boxes, scores, iou_threshold=0.5):
    # Perform hard Non-Maximum Suppression.

    boxes = np.asarray(boxes, dtype=float)
    # Convert the boxes into a NumPy array.

    scores = np.asarray(scores, dtype=float)
    # Convert the confidence scores into a NumPy array.

    order = np.argsort(scores)[::-1]
    # Get box indices sorted from highest score to lowest.

    keep = []
    # Store the indices of boxes that survive NMS.

    while len(order) > 0:
        # Continue while there are boxes left.

        i = order[0]
        # Select the box with the highest score.

        keep.append(i)
        # Keep the selected box.

        if len(order) == 1:
            break
        # Stop if there are no other boxes to compare.

        remaining = order[1:]
        # Get all boxes that are still remaining.

        x1 = np.maximum(boxes[i, 0], boxes[remaining, 0])
        # Find the left side of the intersection.

        y1 = np.maximum(boxes[i, 1], boxes[remaining, 1])
        # Find the top side of the intersection.

        x2 = np.minimum(boxes[i, 2], boxes[remaining, 2])
        # Find the right side of the intersection.

        y2 = np.minimum(boxes[i, 3], boxes[remaining, 3])
        # Find the bottom side of the intersection.

        w = np.maximum(0, x2 - x1)
        # Calculate intersection width.

        h = np.maximum(0, y2 - y1)
        # Calculate intersection height.

        intersection = w * h
        # Calculate intersection area.

        area_i = (
            boxes[i, 2] - boxes[i, 0]
        ) * (
            boxes[i, 3] - boxes[i, 1]
        )
        # Calculate the area of the selected box.

        area_remaining = (
            boxes[remaining, 2] - boxes[remaining, 0]
        ) * (
            boxes[remaining, 3] - boxes[remaining, 1]
        )
        # Calculate the areas of the remaining boxes.

        union = area_i + area_remaining - intersection
        # Calculate the union area.

        iou = intersection / union
        # Calculate IoU between the selected box and remaining boxes.

        order = remaining[iou <= iou_threshold]
        # Remove boxes whose IoU is greater than the threshold.

    return keep
    # Return the indices of boxes that survived.

"""

