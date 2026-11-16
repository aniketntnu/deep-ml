import numpy as np


def iou(box_a, box_b):
    iw = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    ih = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter))


def mean_average_precision(preds, gts, iou_thresh=0.5):
    classes = sorted({c for c, _ in gts})
    if not classes:
        return 0.0
    aps = []
    for c in classes:
        gt_boxes = [b for cls, b in gts if cls == c]
        cls_preds = sorted((p for p in preds if p[0] == c), key=lambda p: -p[1])
        if not cls_preds:
            aps.append(0.0)
            continue
        used = [False] * len(gt_boxes)
        tp = np.zeros(len(cls_preds))
        for i, (_, _, box) in enumerate(cls_preds):
            # best-IoU unmatched gt box of this class
            best, best_j = 0.0, -1
            for j, g in enumerate(gt_boxes):
                if not used[j]:
                    v = iou(box, g)
                    if v > best:
                        best, best_j = v, j
            if best_j >= 0 and best >= iou_thresh:
                used[best_j] = True
                tp[i] = 1
        ctp = np.cumsum(tp)
        cfp = np.cumsum(1 - tp)
        recall = ctp / len(gt_boxes)
        precision = ctp / (ctp + cfp)
        # monotone envelope: p[i] = max(p[i:])
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        r_prev = np.concatenate(([0.0], recall[:-1]))
        aps.append(float(np.sum((recall - r_prev) * precision)))
    return round(float(np.mean(aps)), 4)
