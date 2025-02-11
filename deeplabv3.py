# import torch
# import cv2
# import numpy as np
# from torchvision import models, transforms

# # Load DeepLabV3+ model
# model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# # Define preprocessing
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((480, 640)),
#     transforms.ToTensor()
# ])

# # Open video file
# cap = cv2.VideoCapture("input_video.mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("output_deeplab.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), 
#                       (int(cap.get(3)), int(cap.get(4))))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convert frame to tensor
#     input_tensor = transform(frame).unsqueeze(0)

#     # Run inference
#     with torch.no_grad():
#         output = model(input_tensor)["out"][0]
    
#     # Get segmentation mask
#     mask = output.argmax(0).byte().cpu().numpy()
    
#     # Overlay segmentation mask
#     colored_mask = cv2.applyColorMap(mask * 10, cv2.COLORMAP_JET)
#     blended = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)

#     out.write(blended)
#     cv2.imshow("DeepLabV3+ Segmentation", blended)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

import cv2

print("Opening video file...")
cap = cv2.VideoCapture("input_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video loaded: {frame_width}x{frame_height} @ {fps} FPS")

out = cv2.VideoWriter('output_deeplab.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # Apply your segmentation/model here

    out.write(frame)

cap.release()
out.release()
print("Processing complete. Video saved as output_deeplab.mp4")

