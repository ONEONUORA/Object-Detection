import torch
# import cv2
# import numpy as np
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

# # Load configuration and model
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detection
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x")  # Load weights
# predictor = DefaultPredictor(cfg)

# # Open video file
# cap = cv2.VideoCapture("input_video.mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("output_mask_rcnn.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), 
#                       (int(cap.get(3)), int(cap.get(4))))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convert to Detectron2 format and run inference
#     outputs = predictor(frame)
    
#     # Visualize results
#     v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     result_frame = v.get_image()[:, :, ::-1]

#     out.write(result_frame)
#     cv2.imshow("Mask R-CNN Segmentation", result_frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()


import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transform input image
transform = T.Compose([T.ToTensor()])

def detect_objects(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract bounding boxes, labels, and masks
    boxes = predictions[0]['boxes'].numpy().astype(int)
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()
    masks = predictions[0]['masks'].squeeze().numpy()

    return boxes, labels, scores, masks

