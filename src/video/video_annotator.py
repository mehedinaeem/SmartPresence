import cv2


def annotate(frame, box, label: str, color=(0, 180, 0)):
    x, y, w, h = box; cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2); cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return frame

