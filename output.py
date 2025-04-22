import cv2

video_path = "output_yolo.mp4"  # Change to your specific output file for YOLOv5
cap = cv2.VideoCapture(video_path)

frame_number = 0  # Change this to capture a specific frame on demand

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

if ret:
    cv2.imwrite("sample_frame.jpg", frame)  # Save the frame as an image to disk
    print("Frame saved successfully.")
else:
    print("Error reading frame.")

cap.release()
cv2.destroyAllWindows()
