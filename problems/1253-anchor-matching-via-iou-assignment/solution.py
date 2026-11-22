import numpy as np


def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign each anchor a training label via IoU matching.

    labels:
        1  -> positive
        0  -> negative
        -1 -> ignore

    matched_gt:
        index of matched GT box
        -1 if there is no match
    """

    # Convert input to NumPy arrays
    anchors = np.asarray(anchors)
    gt_boxes = np.asarray(gt_boxes)

    N = len(anchors)
    M = len(gt_boxes)

    # ---------------------------------------------------------
    # Edge case 1: no anchors
    # ---------------------------------------------------------
    if N == 0:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int)
        )

    # ---------------------------------------------------------
    # Edge case 2: no GT boxes
    # ---------------------------------------------------------
    if M == 0:
        return (
            np.zeros(N, dtype=int),
            np.full(N, -1, dtype=int)
        )

    # ---------------------------------------------------------
    # Calculate pairwise IoU
    #
    # anchors:  (N, 4)
    # gt_boxes: (M, 4)
    #
    # iou:      (N, M)
    #
    # iou[i, j] = IoU between anchor i and GT j
    # ---------------------------------------------------------

    # Intersection top-left
    inter_x1 = np.maximum(
        anchors[:, None, 0],
        gt_boxes[None, :, 0]
    )

    inter_y1 = np.maximum(
        anchors[:, None, 1],
        gt_boxes[None, :, 1]
    )

    # Intersection bottom-right
    inter_x2 = np.minimum(
        anchors[:, None, 2],
        gt_boxes[None, :, 2]
    )

    inter_y2 = np.minimum(
        anchors[:, None, 3],
        gt_boxes[None, :, 3]
    )

    # Intersection width and height
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)

    # Intersection area
    inter_area = inter_w * inter_h

    # Anchor areas
    anchor_area = (
        np.maximum(0, anchors[:, 2] - anchors[:, 0]) *
        np.maximum(0, anchors[:, 3] - anchors[:, 1])
    )

    # GT areas
    gt_area = (
        np.maximum(0, gt_boxes[:, 2] - gt_boxes[:, 0]) *
        np.maximum(0, gt_boxes[:, 3] - gt_boxes[:, 1])
    )

    # Union area
    union_area = (
        anchor_area[:, None] +
        gt_area[None, :] -
        inter_area
    )

    # IoU
    iou = np.divide(
        inter_area,
        union_area,
        out=np.zeros_like(inter_area, dtype=float),
        where=union_area > 0
    )

    # ---------------------------------------------------------
    # For every anchor:
    # find its best GT
    # ---------------------------------------------------------

    best_gt = np.argmax(iou, axis=1)
    best_iou = np.max(iou, axis=1)

    # ---------------------------------------------------------
    # Initially mark everything as ignore
    # ---------------------------------------------------------

    labels = np.full(N, -1, dtype=int)

    # IoU >= 0.5 → positive
    labels[best_iou >= pos_threshold] = 1

    # IoU < 0.4 → negative
    labels[best_iou < neg_threshold] = 0

    # Anchors in the ignore range remain -1
    #
    # Example:
    # IoU = 0.7 → 1
    # IoU = 0.3 → 0
    # IoU = 0.45 → -1

    # ---------------------------------------------------------
    # matched_gt
    # ---------------------------------------------------------

    matched_gt = np.full(N, -1, dtype=int)

    # Only positive anchors initially get their best GT
    positive = labels == 1
    matched_gt[positive] = best_gt[positive]

    # ---------------------------------------------------------
    # Force a match for EVERY GT
    #
    # For each GT j:
    # find the anchor having the highest IoU with that GT.
    #
    # Even if IoU < pos_threshold, force it to positive.
    # ---------------------------------------------------------

    for j in range(M):

        # Find anchor with highest IoU for GT j
        anchor_idx = np.argmax(iou[:, j])

        # Force this anchor to positive
        labels[anchor_idx] = 1

        # Match it with GT j
        matched_gt[anchor_idx] = j

    return labels, matched_gt