# Task 3 — Real-Time Echocardiogram Video Analysis

## Objective
Process an echocardiogram video frame-by-frame and compare the original ultrasound stream with an enhanced version.

## Processing Pipeline
For every video frame:

1. Read the frame using `cv2.VideoCapture()`.
2. Convert it to grayscale.
3. Apply Histogram Equalization.
4. Apply `COLORMAP_JET`.
5. Apply Gray World color balancing.
6. Apply logarithmic transformation.
7. Apply gamma transformation.
8. Display or save the raw and enhanced frames side-by-side.

## Video Processing
The input `.mp4` is processed sequentially using a `while` loop until all frames have been read.

Representative comparison frames and the processed video are saved in `output/`.

## Output
Examples:

`raw_vs_enhanced_frame50.png`

`raw_vs_enhanced.mp4`

## Run
Open and execute:

`realtime_echo.ipynb`