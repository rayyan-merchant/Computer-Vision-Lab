# Medical Imaging Portfolio — Lab 02

This repository contains three medical image-processing tasks implemented using Python, OpenCV, and NumPy.

## Tasks

### Task 1 — Chest X-Ray Enhancement
Enhances an underexposed chest X-ray using histogram equalization, false-color mapping, thresholding, logarithmic transformation, and gamma correction.

### Task 2 — Cardiac CT-MRI Fusion
Enhances and combines corresponding CT and MRI slices using weighted multimodal image fusion.

### Task 3 — Echocardiogram Video Analysis
Processes an ultrasound video frame-by-frame and produces raw-vs-enhanced comparisons.

## Repository Structure

StudentName_Medical_Imaging_Portfolio/
│
├── Task_1_Chest_XRay/
│   ├── data/
│   ├── output/
│   ├── xray_enhancement.ipynb
│   └── README.md
│
├── Task_2_Cardiac_Fusion/
│   ├── data/
│   ├── output/
│   ├── modal_fusion.ipynb
│   └── README.md
│
├── Task_3_Echo_Analysis/
│   ├── data/
│   ├── output/
│   ├── realtime_echo.ipynb
│   └── README.md
│
├── requirements.txt
└── README.md

## Setup

Install dependencies:

pip install -r requirements.txt

Then open the required notebook and run the cells sequentially.

## Notes
- All scripts use relative file paths.
- Only sample images/video used by the notebooks are included in the repository.
- Large datasets are not uploaded to the repository.