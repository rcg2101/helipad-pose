
import cv2
from ultralytics import YOLO

# Load the pretrained YOLO11n Spose estimation model
# The model detects people and estimates 17 body keypoints
model = YOLO("yolo11n-pose.pt")

# Open the default webcam (camera index 0)
# CAP_DSHOW selects the DirectShow backend on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Request the webcam's native resolution: 1280 x 720 (16:9)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Print the actual resolution accepted by the webcam
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {width}x{height}")

# Create a resizable window and set it to fullscreen
cv2.namedWindow("YOLO11n-Pose", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "YOLO11n-Pose",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# Continuously capture and process webcam frames
while cap.isOpened():

    # Read the next frame from the webcam
    ret, frame = cap.read()

    # Stop if the webcam fails to provide a frame
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Run pose estimation on the current frame
    # imgsz=640 sets the inference size; it does not resize the output frame
    # conf=0.25 sets the minimum detection confidence to 25%
    # verbose=False suppresses per-frame inference messages in the terminal
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    # Draw the detected people, keypoints and skeleton onto the frame
    # The annotated image retains the original frame dimensions
    annotated_frame = results[0].plot()

    # Display the annotated frame in the fullscreen window
    cv2.imshow("YOLO11n-Pose", annotated_frame)

    # Exit when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam so other applications can use it
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
