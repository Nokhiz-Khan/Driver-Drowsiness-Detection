import cv2
import time
import winsound
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

closed_since = None
ALERT_THRESHOLD_SECONDS = 6

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    is_closed = False
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = str(model.names[cls_id]).lower()
            if 'close' in class_name:
                is_closed = True
                break

    now = time.time()

    if is_closed:
        if closed_since is None:
            closed_since = now
        elapsed = now - closed_since
        if elapsed >= ALERT_THRESHOLD_SECONDS:
            cv2.putText(annotated_frame, "ALERT: DROWSINESS DETECTED!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)
            winsound.Beep(1000, 500)
        else:
            cv2.putText(annotated_frame, f"Eyes closed: {elapsed:.1f}s", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        closed_since = None

    cv2.imshow("Driver Drowsiness Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()