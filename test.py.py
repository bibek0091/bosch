import cv2
from ultralytics import YOLO

def main():
    # 1. Load your custom trained YOLO model
    print("Loading model...")
    model = YOLO("best.pt")

    # 2. Open the webcam (0 is the default camera. Use 1 or 2 if you have external cameras)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Starting webcam stream... Press 'q' to quit.")

    while True:
        # Read a single frame from the camera
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam.")
            break

        # 3. Run YOLO inference on the frame
        # stream=True is recommended for continuous video feeds for better performance
        # conf=0.5 sets a 50% confidence threshold for drawing bounding boxes
        results = model.predict(source=frame, conf=0.5, stream=True, verbose=False)

        # 4. Draw bounding boxes and labels onto the frame
        for result in results:
            # .plot() automatically draws the bounding boxes, confidence scores, and class names
            annotated_frame = result.plot()

        # 5. Display the annotated video feed
        cv2.imshow("YOLO Webcam Inference", annotated_frame)

        # 6. Listen for the 'q' key to stop the webcam loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up: release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()