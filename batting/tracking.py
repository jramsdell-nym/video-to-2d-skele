"""Association adapted from experiments/2d-tracking/src/common.py."""
import math


def joint_box(points):
    visible = [p for p in points.values() if p['usable']]
    if len(visible) < 5:
        return None
    return (min(p['normalized'][0] for p in visible),
            min(p['normalized'][1] for p in visible),
            max(p['normalized'][0] for p in visible),
            max(p['normalized'][1] for p in visible))


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap(a, b):
    intersection = area((max(a[0], b[0]), max(a[1], b[1]),
                         min(a[2], b[2]), min(a[3], b[3])))
    return intersection / max(1e-9, area(a) + area(b) - intersection)


class HitterSelector:
    """Largest visible joint box, then gated association; never fill gaps."""

    def __init__(self):
        self.last = None

    def choose(self, poses):
        candidates = []
        for index, points in enumerate(poses):
            box = joint_box(points)
            if box is None or area(box) < .012:
                continue
            score = area(box)
            if self.last is not None:
                previous = self.last
                distance = math.hypot((box[0] + box[2] - previous[0] - previous[2]) / 2,
                                      (box[1] + box[3] - previous[1] - previous[3]) / 2)
                ratio = area(box) / max(1e-6, area(previous))
                if distance > .22 or not .30 < ratio < 3.3:
                    continue
                score = 2 * overlap(previous, box) - distance + .1 * min(ratio, 1 / ratio)
            candidates.append((score, index, box))
        candidates.sort(key=lambda item: -item[0])
        if not candidates:
            return None, 'missing'
        if self.last is not None and len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .08:
            return None, 'ambiguous'
        _, selected, self.last = candidates[0]
        return selected, 'observed'
