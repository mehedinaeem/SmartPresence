from __future__ import annotations

import math


class CentroidTracker:
    def __init__(self, max_distance: float = 100.0) -> None:
        self.max_distance, self.next_id, self.centroids = max_distance, 1, {}

    def update(self, boxes: list[tuple[int, int, int, int]]) -> dict[int, tuple[int, int, int, int]]:
        remaining = set(self.centroids); result = {}
        for box in boxes:
            x, y, w, h = box; center = (x + w // 2, y + h // 2)
            candidates = [(math.dist(center, self.centroids[key]), key) for key in remaining]
            if candidates and min(candidates)[0] <= self.max_distance:
                _, track_id = min(candidates); remaining.remove(track_id)
            else:
                track_id = self.next_id; self.next_id += 1
            self.centroids[track_id] = center; result[track_id] = box
        for track_id in remaining: self.centroids.pop(track_id, None)
        return result

