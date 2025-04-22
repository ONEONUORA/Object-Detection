

import torch
import cv2
import numpy as np
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

# Load DeepLabV3+ model
model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).eval()

# Open video  file for reading
# Define preprocessing
print("Opening video file...")
cap = cv2.VideoCapture("input_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video loaded: {frame_width}x{frame_height} @ {fps} FPS")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_deeplabs.mp4", fourcc, fps, (frame_width, frame_height))

# For processing, I'll use a consistent input size for model to avoid resizing issues
# Define input size for the model
# Note: DeepLabV3+ typically expects a square input size, but we can resize to the model's expected input
# size and then resize back to the original frame size for output
# This is a common practice in segmentation tasks
# Note: The model expects input size of 480x640

input_height, input_width = 480, 640

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}...")
    
    # Keep a copy of the original frame
    original_frame = frame.copy()
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize for model input (explicit resize instead of transform)
    input_image = cv2.resize(rgb_frame, (input_width, input_height))
    
    # Convert to tensor
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    
    # Get segmentation mask
    mask = output.argmax(0).byte().cpu().numpy()
    
    # Debug print to check mask shape and values
    print(f"Mask shape: {mask.shape}, Values range: {mask.min()}-{mask.max()}")
    
    # Resize mask to match original frame dimensions
    mask_resized = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    
    # Create a 3-channel colored mask
    colored_mask = np.zeros_like(original_frame)
    
    # Apply color map to mask (ensure it's properly normalized to 0-255)
    normalized_mask = np.uint8(mask_resized * (255 / np.maximum(mask_resized.max(), 1)))
    temp_colored = cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET)
    
    # Ensure the colored mask has the same shape as the original frame
    assert temp_colored.shape == original_frame.shape, f"Shape mismatch: {temp_colored.shape} vs {original_frame.shape}"
    
    # Blend with original frame
    try:
        blended = cv2.addWeighted(original_frame, 0.6, temp_colored, 0.4, 0)
        print(f"Blend successful. Shapes: frame={original_frame.shape}, colored_mask={temp_colored.shape}")
    except cv2.error as e:
        print(f"Error blending: {e}")
        print(f"Frame shape: {original_frame.shape}, Colored mask shape: {temp_colored.shape}")
        # Fall back to just using the original frame
        blended = original_frame
    
    # Write and display frame
    out.write(blended)
    cv2.imshow("DeepLabV3+ Segmentation", blended)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Video saved as output_deeplab.mp4")