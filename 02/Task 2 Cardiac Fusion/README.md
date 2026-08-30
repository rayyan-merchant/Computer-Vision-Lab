# Task 2 — Multi-Modal Cardiac Image Fusion

## Objective
Combine corresponding cardiac CT and MRI slices into a single fused representation.

CT contributes stronger structural boundaries, while MRI contributes soft-tissue intensity variations.

## Processing Pipeline
1. Load corresponding CT and MRI slices.
2. Apply Histogram Equalization independently to both images.
3. Convert the enhanced images into false-color heatmaps.
4. Perform weighted CT-MRI fusion using `cv2.addWeighted()`.
5. Apply logarithmic and gamma transformations to the fused image.
6. Compare CT, MRI, and fused outputs.

## Weighted Fusion

F = α(CT) + β(MRI)

A higher weight is assigned to CT to preserve structural information while MRI provides additional tissue variation.

Example:

α = 0.7  
β = 0.3

## Output
Enhanced CT/MRI images, fused images, and comparison figures are stored in `output/`.

## Run
Open and execute:

`modal_fusion.ipynb`