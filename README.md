**Face Mask Detector (Computer Vision)**

An OpenCV project that detects whether a person is wearing a face mask in an image. The repository contains preprocessing code, model training, evaluation, and inference scripts implemented and demonstrated inside the Jupyter notebook `/mnt/data/v-ml-mask-detector.ipynb`.

## Features
- Data preprocessing and augmentation
- CNN-based model training
- Model evaluation
- Inference/demo

## Directory structure (recommended)
```
project-root/
├─ dataset/
├─ v-ml-mask-detector.ipynb
└─ README.md
```

## Requirements
- Python 3.8+
- numpy, pandas, matplotlib, opencv-python, tensorflow/torch, scikit-learn, jupyter, tqdm

## Setup
```bash
python -m venv venv
pip install -r requirements.txt
jupyter notebook /mnt/data/v-ml-mask-detector.ipynb
```

## How to run
```bash
python src/train.py ...
python src/infer.py ...
```

## Model details
- Architecture: MobileNetV2 or custom CNN
- Loss: BCE
- Optimizer: Adam
- Metrics: Accuracy, Precision, Recall

## Evaluation
Training/validation curves and confusion matrix are inside the notebook.

## Inference
Utility function `predict_image()` included.

## Troubleshooting
- GPU usage
- Low accuracy fixes

## License
MIT or Apache-2.0 recommended.
