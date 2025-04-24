


import torch # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import torchvision.transforms as T # type: ignore
from torchvision.models.detection import maskrcnn_resnet50_fpn # type: ignore
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights # type: ignore

# Load Mask R-CNN model with updated weights parameter
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()

# Transform input image
transform = T.Compose([T.ToTensor()])

def detect_objects(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract bounding boxes, labels, and scores
    boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    masks = predictions[0]['masks'].squeeze().cpu().numpy()

    return boxes, labels, scores, masks

# Open video file
cap = cv2.VideoCapture("input_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_mask_rcnn.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), 
                     (int(cap.get(3)), int(cap.get(4))))

# Get class names from COCO dataset
COCO_INSTANCE_CATEGORY_NAMES = weights.meta["categories"]

# Choose a threshold for detections
threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    boxes, labels, scores, masks = detect_objects(frame)
    
    # Visualize results
    result_frame = frame.copy()
    
    # Draw detections that are above the threshold
    for i in range(len(scores)):
        if scores[i] > threshold:
            # Get box coordinates
            x1, y1, x2, y2 = boxes[i]
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label and score
            label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            text = f"{label}: {scores[i]:.2f}"
            cv2.putText(result_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Apply mask if score is high enough
            if scores[i] > threshold and i < len(masks):
                mask = masks[i] > 0.5
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask] = [0, 0, 255]  # Red mask
                
                # Blend mask with original frame
                alpha = 0.5
                result_frame = cv2.addWeighted(result_frame, 1, colored_mask, alpha, 0)
    
    out.write(result_frame)
    cv2.imshow("Mask R-CNN Segmentation", result_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
