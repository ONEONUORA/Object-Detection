# Object Detection and Semantic Segmentation with YOLOv5, Mask R-CNN, and DeepLabV3+

This project demonstrates object detection and semantic segmentation using state-of-the-art deep learning models, including YOLOv5, Mask R-CNN, and DeepLabV3+. The current implementation focuses on semantic segmentation using the DeepLabV3+ model.

---

## Features

- **DeepLabV3+ Semantic Segmentation**:
  - Processes video frames to generate segmentation masks.
  - Overlays segmentation masks on the original video frames.
  - Saves the processed video with segmentation overlays.

- **Customizable Input/Output**:
  - Supports video input and output with OpenCV.
  - Handles resizing and preprocessing for model compatibility.

---

## Requirements

Ensure you have the following installed:

- Python 3.8 or later
- Required Python libraries:
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `numpy`

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ONEONUORA/Object-Detection.git
   cd Object-Detection-YOLOv5-Mask-R-CNN-DeepLabv3

2. **Set Up a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
3. **Install Dependencies:**
   ```bash
   pip install torch torchvision opencv-python numpy
4. **Usage**

     *1. Prepare the Input Video:*

   - Place your input video file in the project directory.
   - Ensure the file is named input_video.mp4 or update the script to 
      match your file name.

     *2. Run the Script:*
       ```bash
         python deeplabv3.py

    *3. Output:*

    - The processed video with segmentation overlays will be saved as 
      output_deeplab.mp4 in the project directory.

5. **File Structure**

Object-Detection-YOLOv5-Mask-R-CNN-DeepLabv3/


├── deeplabv3.py          # Main script for DeepLabV3+ segmentation

├── input_video.mp4       # Input video file (add your own)

├── output_deeplab.mp4    # Processed video with segmentation overlays

├── README.md             # Project documentation

└── venv/                 # Virtual environment (optional)


**How It Works**

*Model Loading:*

The script uses the DeepLabV3+ model with a ResNet-101 backbone, pre-trained on the COCO dataset.

Video Processing:

Reads frames from the input video

Preprocesses frames to match the model's input size (480x640).

Runs inference to generate segmentation masks.

Resizes masks to match the original frame size and overlays them on the frames.

*Output:*

Writes the processed frames to an output video file.

**Troubleshooting**

- Error: Could not open video file:

    Ensure the input video file exists and is named correctly.

    Check the file path in the script.

    Mismatch in frame and mask sizes:

    Ensure the mask is resized to match the original frame dimensions before 
    blending.

- Dependencies not found:

  Ensure all required libraries are installed using pip install.

**Future Enhancements**

Add support for YOLOv5 and Mask R-CNN models.

Implement real-time video processing via webcam.

Add a graphical user interface (GUI) for easier interaction.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.