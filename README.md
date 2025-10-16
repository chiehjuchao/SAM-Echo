# MedSAM Echo - Medical Image Segmentation with SAM

This repository provides fine-tuned SAM and MedSAM model checkpoints specifically trained on echocardiography images, along with inference code for left ventricular segmentation.

## Overview

This implementation provides clean, easy-to-use inference capabilities with bounding box prompts using foundation models (SAM) and domain-specific models (MedSAM) fine-tuned on cardiac ultrasound images.

## Features

- Clean inference pipeline for medical image segmentation
- Support for custom bounding box prompts
- Visualization utilities for masks and predictions
- Flexible model loading (multiple checkpoint formats)
- Batch processing capabilities

## Requirements

Install the required dependencies:

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install numpy
pip install segment-anything
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

## Model Checkpoints

We provide both fine-tuned SAM and MedSAM checkpoints used in our paper. For details on model training and evaluation, please refer to our publication: [Foundation versus domain-specific models for left ventricular segmentation on cardiac ultrasound](https://www.nature.com/articles/s41746-025-01730-y).

**Download checkpoints:** [Google Drive](https://drive.google.com/drive/folders/1c2uYJnKpvTALuJ2i6nPBbWBU2aDwMhi1?usp=sharing)

The code supports two loading methods:

1. **Simple checkpoint**: Direct model weights file
2. **Checkpoint with metadata**: Model with training state (epoch, optimizer, etc.)

Place your checkpoint file in the project directory and update the path in the notebook.

## Usage

### Basic Inference

Open the `MedSAM_inference.ipynb` notebook and follow these steps:

1. **Load the model**:
```python
from segment_anything import sam_model_registry

MedSAM_CKPT_PATH = "path/to/medsam_model_best.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()
```

2. **Load and preprocess your image**:
```python
import cv2
import numpy as np

img_cv = cv2.imread('path/to/image.png', cv2.IMREAD_COLOR)
img_3c = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
H, W, _ = img_3c.shape

# Resize to 1024x1024
img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)
```

3. **Define bounding box and run inference**:
```python
# Bounding box in original image coordinates [x0, y0, x1, y1]
box_np = np.array([[50, 50, 200, 200]])

# Scale to 1024x1024
box_1024 = box_np / np.array([W, H, W, H]) * 1024

# Generate embeddings and run inference
with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)
    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, 1024, 1024)
```

4. **Visualize results**:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(img_1024)
show_mask(medsam_seg, ax)
show_box(box_1024[0], ax)
plt.show()
```

### Utility Functions

#### `medsam_inference(medsam_model, img_embed, box_1024, H, W)`
Performs segmentation inference given image embeddings and bounding box.

**Parameters:**
- `medsam_model`: MedSAM model instance
- `img_embed`: Image embeddings from encoder (B, 256, 64, 64)
- `box_1024`: Bounding box in 1024x1024 scale (B, 4)
- `H`: Target height for output mask
- `W`: Target width for output mask

**Returns:**
- Binary segmentation mask (H, W)

#### `get_bboxes_from_mask(mask)`
Extracts the largest bounding box from a binary mask.

**Parameters:**
- `mask`: Binary mask (numpy array or torch tensor)

**Returns:**
- Bounding box coordinates [x0, y0, x1, y1]

## File Structure

```
.
├── MedSAM_inference.ipynb          # Main inference notebook
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## Notes

- Images are resized to 1024x1024 for inference (MedSAM's native resolution)
- Bounding boxes must be scaled appropriately when resizing images
- The model outputs binary masks thresholded at 0.5
- GPU is recommended for faster inference

## Citation

If you use MedSAM in your research, please cite:

```bibtex
@article{medsam2023,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  year={2024}
}
```

## License

This project follows the license terms of the original SAM and MedSAM projects.

## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [MedSAM](https://github.com/bowang-lab/MedSAM) by Wang Lab

## Contact

For questions or issues, please open an issue in the repository.
