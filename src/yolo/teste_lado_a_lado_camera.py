import cv2
from ultralytics import YOLO

yolo26 = YOLO(r"C:\Users\rodri\dev\helipad-pose\runs\pose\h12-3\weights\best.pt")
yolo11 = YOLO(r"C:\Users\rodri\dev\helipad-pose\runs\pose\runs\pose\h12\weights\best.pt")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam.")

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        pred26 = yolo26.predict(source=frame, verbose=False)[0].plot()
        pred11 = yolo11.predict(source=frame, verbose=False)[0].plot()

        height, width = frame.shape[:2]
        pred26 = cv2.resize(pred26, (width, height))
        pred11 = cv2.resize(pred11, (width, height))

        # Se quiser n usar GPU:
        # pred26 = yolo26.predict(source=frame, device="cpu", verbose=False)[0].plot()
        # pred11 = yolo11.predict(source=frame, device="cpu", verbose=False)[0].plot()

        cv2.putText(pred26, "YOLO26", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pred11, "YOLO11", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Comparacao lado a lado", cv2.hconcat([pred26, pred11]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()