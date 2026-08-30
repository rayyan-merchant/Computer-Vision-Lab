# Task 1 — Diagnostic Enhancement of Chest X-Rays

## Objective
Enhance an underexposed grayscale chest X-ray using basic image-processing techniques to improve the visibility of intensity and structural variations.

## Processing Pipeline
1. Load the X-ray as a grayscale image.
2. Apply Histogram Equalization for contrast enhancement.
3. Apply `COLORMAP_JET` for false-color visualization.
4. Perform Gray World color balancing.
5. Apply intensity thresholding to isolate high-intensity regions.
6. Apply logarithmic transformation to enhance darker regions.
7. Apply gamma correction with γ < 1 to improve midtones.

## Key Equations

Log transformation:

s = c log(1 + r)

Gamma transformation:

s = c r^γ

## Output
Processed images and comparison figures are stored in the `output/` directory.

## Run
Open and execute:

`xray_enhancement.ipynb`