import cv2

video_path = "output_yolo.mp4"  # Change to your specific output file
cap = cv2.VideoCapture(video_path)

frame_number = 0  # Change this to capture a specific frame

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

if ret:
    cv2.imwrite("sample_frame.jpg", frame)  # Save the frame as an image
    print("Frame saved successfully.")
else:
    print("Error reading frame.")

cap.release()
cv2.destroyAllWindows()
