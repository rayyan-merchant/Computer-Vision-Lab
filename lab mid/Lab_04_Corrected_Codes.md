# Lab 04 — Corrected and Tested Python/OpenCV Code

**Course:** AI-4002 Computer Vision Lab (FAST-NUCES Karachi)  
**Basis:** *Lab 04 Manual.pdf* (23 PDF pages)  
**Purpose:** Exam-ready **working implementations**, not blind copies of the manual. The first 15 sections correspond to its 15 printed code examples, in the same order. The appendix includes additional implementations for topics the manual only describes.

> **IMPORTANT:** These snippets intentionally correct mistakes, ambiguities, and incomplete examples in the source. The original PDF is the source for topic order, **not** an authority for algorithm correctness. Corrections are explained beneath the headings. Each code fence is standalone, can be pasted into a Jupyter cell or Python file, and uses `image.jpg` in the current directory. Replace the path as needed.

## Setup

```bash
pip install numpy opencv-python matplotlib scikit-image
```

All snippets use Matplotlib instead of `cv2.imshow()`, making them suitable for notebooks. If `cv2.imread` fails, check the file path. **Signed gradient/derivative responses use float types** so negative values are preserved. Code was smoke-tested with a synthetic sample image; arbitrary image paths and platform installations have not been tested.

## Contents

1. [HOG — Histogram of Oriented Gradients](#1-hog--histogram-of-oriented-gradients)
2. [LBP — Local Binary Pattern](#2-lbp--local-binary-pattern)
3. [HED — Histogram of Edge Directions](#3-hed--histogram-of-edge-directions)
4. [HIG — Histogram of Intensity Gradients](#4-hig--histogram-of-intensity-gradients)
5. [Texture energy and local contrast histograms](#5-texture-energy-and-local-contrast-histograms)
6. [Box / mean blur](#6-box--mean-blur)
7. [Gaussian blur](#7-gaussian-blur)
8. [Sobel and Scharr gradient filters](#8-sobel-and-scharr-gradient-filters)
9. [Embossing filter](#9-embossing-filter)
10. [Standard 2D mathematical convolution (same-sized)](#10-standard-2d-mathematical-convolution-same-sized)
11. [Valid convolution — NO padding](#11-valid-convolution--no-padding)
12. [Same convolution — ZERO padding](#12-same-convolution--zero-padding)
13. [Valid convolution with stride 2](#13-valid-convolution-with-stride-2)
14. [Gradient-based edge detection (Sobel + threshold)](#14-gradient-based-edge-detection-sobel--threshold)
15. [Canny edge detector](#15-canny-edge-detector)

---

## 1. HOG — Histogram of Oriented Gradients

**Source location:** PDF p. 7 / manual §3.1.

**What was fixed / what to remember:** Use the numeric `features` for classification; `hog_vis` is only a visualization. Nine unsigned-orientation bins, 8×8 cells, overlapping 2×2-cell blocks.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

features, hog_vis = hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)
hog_vis = cv2.normalize(hog_vis, None, 0, 255,
                        cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(hog_vis, cmap='gray')
plt.title('HOG visualization')
plt.axis('off')
plt.tight_layout()
plt.show()
print('HOG feature vector shape:', features.shape)
print('First 20 features:', features[:20])
```

## 2. LBP — Local Binary Pattern

**Source location:** PDF pp. 8–9 / manual §3.2.

**What was fixed / what to remember:** `uniform` LBP with P=8 has P+2=10 histogram bins. The histogram, not the LBP display image, is the compact texture descriptor.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

P, R = 8, 1
lbp_image = local_binary_pattern(image, P, R, method='uniform')

hist, _ = np.histogram(lbp_image.ravel(),
                       bins=np.arange(P + 3))
hist = hist.astype(np.float64)
hist /= max(hist.sum(), 1.0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(lbp_image, cmap='gray')
plt.title('LBP codes')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.bar(np.arange(P + 2), hist)
plt.title('Normalized uniform LBP histogram')
plt.xlabel('LBP category')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
print('LBP feature vector:', hist)
```

## 3. HED — Histogram of Edge Directions

**Source location:** PDF pp. 10–11 / manual §3.4.

**What was fixed / what to remember:** Compute the histogram **only at Canny edge pixels**. `arctan2` returns signed angles, so use modulo 360. Values below are gradient-normal directions; for the geometric direction *along* edges, add 90°.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

smooth = cv2.GaussianBlur(image, (5, 5), 1)
edges = cv2.Canny(smooth, 50, 150, L2gradient=True)
gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)

gradient_angles = np.degrees(np.arctan2(gy, gx)) % 360
angles_at_edges = gradient_angles[edges > 0]
hist, bins = np.histogram(angles_at_edges, bins=8,
                          range=(0, 360))
hist = hist.astype(np.float64)
hist /= max(hist.sum(), 1.0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Canny edges')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge')
plt.xticks(np.arange(0, 361, 45))
plt.title('HED: gradient direction at edges')
plt.xlabel('Degrees')
plt.tight_layout()
plt.show()
print('HED vector:', hist)
```

## 4. HIG — Histogram of Intensity Gradients

**Source location:** PDF pp. 12–13 / manual §3.5.

**What was fixed / what to remember:** This is a **global** orientation histogram, unlike block-normalized HOG. Use unsigned directions 0–180°, ignore zero gradients, and weight bins by gradient strength. These are improvements to the manual’s raw-angle-count version.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.hypot(gx, gy)
orientation = np.degrees(np.arctan2(gy, gx)) % 180
mask = magnitude > 1e-12

hist, bins = np.histogram(orientation[mask], bins=9,
                          range=(0, 180), weights=magnitude[mask])
hist = hist.astype(np.float64)
hist /= max(hist.sum(), 1e-12)

plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge')
plt.xticks(np.arange(0, 181, 20))
plt.title('Magnitude-weighted intensity gradient histogram')
plt.xlabel('Gradient orientation (degrees)')
plt.ylabel('Normalized weight')
plt.show()
print('HIG feature vector:', hist)
```

## 5. Texture energy and local contrast histograms

**Source location:** PDF pp. 14–15 / manual §3.6.

**What was fixed / what to remember:** Energy follows the manual’s **sum of squared local intensities**. Contrast is the true local population standard deviation, not a sum of image pixels. Convert to floating point **before squaring** to prevent `uint8` overflow. This is not the GLCM definition of texture energy/contrast.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
image = image.astype(np.float64)
window = (3, 3)

energy = cv2.boxFilter(image ** 2, -1, window, normalize=False)
local_mean = cv2.blur(image, window)
local_mean_squared = cv2.blur(image ** 2, window)
variance = np.maximum(local_mean_squared - local_mean ** 2, 0)
contrast = np.sqrt(variance)

energy_hist, energy_bins = np.histogram(energy.ravel(), bins=32)
contrast_hist, contrast_bins = np.histogram(contrast.ravel(), bins=32)
energy_hist = energy_hist / max(energy_hist.sum(), 1)
contrast_hist = contrast_hist / max(contrast_hist.sum(), 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(energy_bins[:-1], energy_hist,
        width=np.diff(energy_bins), align='edge')
plt.title('Local squared-intensity energy')
plt.subplot(1, 2, 2)
plt.bar(contrast_bins[:-1], contrast_hist,
        width=np.diff(contrast_bins), align='edge')
plt.title('Local standard-deviation contrast')
plt.tight_layout()
plt.show()
print('Energy histogram shape:', energy_hist.shape)
print('Contrast histogram shape:', contrast_hist.shape)
```

## 6. Box / mean blur

**Source location:** PDF p. 16 / manual §4.2.

**What was fixed / what to remember:** A 3×3 normalized all-ones kernel performs 2D averaging. `cv2.blur` is the shorter equivalent for typical blur operations.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

kernel = np.ones((3, 3), dtype=np.float32) / 9
box_filter = cv2.filter2D(image, -1, kernel)
box_blur = cv2.blur(image, (3, 3))

plt.figure(figsize=(10, 3))
for i, (title, img) in enumerate([
    ('Original', image), ('filter2D average', box_filter),
    ('cv2.blur', box_blur)
], 1):
    plt.subplot(1, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## 7. Gaussian blur

**Source location:** PDF p. 17 / manual §4.2.

**What was fixed / what to remember:** `getGaussianKernel(5,1)` is only a 5×1 **one-dimensional** vector. To form a 2D Gaussian kernel, multiply it by its transpose—or simply use `cv2.GaussianBlur`.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

gaussian_blur = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

gaussian_1d = cv2.getGaussianKernel(5, 1.0)
gaussian_2d = gaussian_1d @ gaussian_1d.T
gaussian_filter2d = cv2.filter2D(image, -1, gaussian_2d)

plt.figure(figsize=(10, 3))
for i, (title, img) in enumerate([
    ('Original', image), ('GaussianBlur', gaussian_blur),
    ('2D kernel / filter2D', gaussian_filter2d)
], 1):
    plt.subplot(1, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## 8. Sobel and Scharr gradient filters

**Source location:** PDF p. 17 / manual §4.2.

**What was fixed / what to remember:** Use grayscale and a signed floating-point depth to preserve negative derivatives; calculate magnitudes using `np.hypot`. `Gx` measures changes along x (often highlighting vertical boundaries).

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = np.hypot(sobel_x, sobel_y)

scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
scharr_magnitude = np.hypot(scharr_x, scharr_y)

plt.figure(figsize=(10, 4))
for i, (title, img) in enumerate([
    ('Original', image), ('Sobel magnitude', sobel_magnitude),
    ('Scharr magnitude', scharr_magnitude)
], 1):
    plt.subplot(1, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## 9. Embossing filter

**Source location:** PDF p. 17 / manual §4.2.

**What was fixed / what to remember:** Use a **zero-sum** directional emboss kernel, compute a signed response, then add a neutral-gray offset of 128. Otherwise negative responses are clipped and flat regions can appear incorrectly.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

emboss_kernel = np.array([
    [-2, -1, 0],
    [-1,  0, 1],
    [ 0,  1, 2]
], dtype=np.float32)
response = cv2.filter2D(image.astype(np.float32),
                        cv2.CV_32F, emboss_kernel)
embossed = np.clip(response + 128, 0, 255).astype(np.uint8)

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(embossed, cmap='gray', vmin=0, vmax=255)
plt.title('Embossed')
plt.axis('off')
plt.tight_layout()
plt.show()
```

## 10. Standard 2D mathematical convolution (same-sized)

**Source location:** PDF pp. 17–18 / manual §4.2.1.1.

**What was fixed / what to remember:** `cv2.filter2D` performs **correlation**. To implement mathematical convolution with an asymmetric kernel, rotate the kernel by 180° (`cv2.flip(kernel,-1)`). This example uses a same-sized result and zero padding.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
image = image.astype(np.float32)

kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float32)

conv_kernel = cv2.flip(kernel, -1)
result = cv2.filter2D(image, cv2.CV_32F, conv_kernel,
                     borderType=cv2.BORDER_CONSTANT)

plt.imshow(result, cmap='gray')
plt.title('2D convolution (signed output)')
plt.axis('off')
plt.show()
print('Input shape:', image.shape, 'Output shape:', result.shape)
```

## 11. Valid convolution — NO padding

**Source location:** PDF p. 18 / manual §4.2.1.2.

**What was fixed / what to remember:** `BORDER_CONSTANT` alone does **not** make valid output in OpenCV. Compute a same-sized response and crop away the kernel radius. This compact method assumes an odd-size 2D kernel and an image at least as large as the kernel.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
image = image.astype(np.float32)

kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float32)
kh, kw = kernel.shape
assert kh % 2 == 1 and kw % 2 == 1
assert image.shape[0] >= kh and image.shape[1] >= kw

response = cv2.filter2D(image, cv2.CV_32F, cv2.flip(kernel, -1),
                        borderType=cv2.BORDER_CONSTANT)
py, px = kh // 2, kw // 2
valid = response[py:image.shape[0]-py, px:image.shape[1]-px]

print('Input:', image.shape, 'Kernel:', kernel.shape)
print('VALID output:', valid.shape)  # (H-kh+1, W-kw+1)
plt.imshow(valid, cmap='gray')
plt.title('Valid convolution')
plt.axis('off')
plt.show()
```

## 12. Same convolution — ZERO padding

**Source location:** PDF pp. 18–19 / manual §4.2.1.3.

**What was fixed / what to remember:** `BORDER_REFLECT` is **reflection padding**, not zero padding. With stride 1 and an odd-sized centered kernel, `BORDER_CONSTANT` produces a zero-padded result with the **same** spatial dimensions.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
image = image.astype(np.float32)

kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float32)
result = cv2.filter2D(image, cv2.CV_32F,
                     cv2.flip(kernel, -1),
                     borderType=cv2.BORDER_CONSTANT)

print('Input:', image.shape, 'SAME output:', result.shape)
plt.imshow(result, cmap='gray')
plt.title('Same-sized convolution (zero padding)')
plt.axis('off')
plt.show()
```

## 13. Valid convolution with stride 2

**Source location:** PDF p. 19 / manual §4.2.1.4.

**What was fixed / what to remember:** `cv2.filter2D` does not support a `strides=` argument. Obtain the correctly cropped valid response and sample every second location. For kernel K, image N, stride S, the output dimension is `floor((N-K)/S)+1`.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
image = image.astype(np.float32)

kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float32)
kh, kw = kernel.shape
assert kh % 2 == 1 and kw % 2 == 1
assert image.shape[0] >= kh and image.shape[1] >= kw

response = cv2.filter2D(image, cv2.CV_32F, cv2.flip(kernel, -1),
                        borderType=cv2.BORDER_CONSTANT)
py, px = kh // 2, kw // 2
valid = response[py:image.shape[0]-py, px:image.shape[1]-px]

stride = 2
strided = valid[::stride, ::stride]

expected_h = (image.shape[0] - kh) // stride + 1
expected_w = (image.shape[1] - kw) // stride + 1
assert strided.shape == (expected_h, expected_w)
print('Input:', image.shape, 'Valid stride-2:', strided.shape)
plt.imshow(strided, cmap='gray')
plt.title('Valid convolution / stride 2')
plt.axis('off')
plt.show()
```

## 14. Gradient-based edge detection (Sobel + threshold)

**Source location:** PDF p. 21 / manual §5.1.

**What was fixed / what to remember:** Combine both signed Sobel derivatives into a magnitude image. Threshold its magnitude to create a binary edge map; unlike Canny this approach does **not** include non-maximum suppression or hysteresis.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.hypot(gx, gy)
direction_degrees = np.degrees(np.arctan2(gy, gx)) % 360

threshold = 100
edges = (magnitude >= threshold).astype(np.uint8) * 255

plt.figure(figsize=(10, 3))
for i, (title, img) in enumerate([
    ('Original', image), ('Gradient magnitude', magnitude),
    ('Thresholded edges', edges)
], 1):
    plt.subplot(1, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()
print('Direction image shape:', direction_degrees.shape)
```

## 15. Canny edge detector

**Source location:** PDF p. 23 / manual §5.2.1.5.

**What was fixed / what to remember:** Explicit smoothing makes the preprocessing clear. `L2gradient=True` uses Euclidean gradient magnitude; thresholds must be tuned for the input.

```python
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

smooth = cv2.GaussianBlur(image, (5, 5), 1)
edges = cv2.Canny(smooth, threshold1=100,
                  threshold2=200, L2gradient=True)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny edges')
plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## Additional exam-useful code (not printed in the manual)

The manual describes the following topics or asks an exercise but does not provide corresponding code. These implementations are **supplemental**.

### A1. Color histogram (BGR channels)

Manual §3.3 describes color histograms but supplies no Python snippet.

```python
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError('image.jpg')

for channel, color in enumerate(('b', 'g', 'r')):
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
    plt.plot(hist.ravel(), color=color)

plt.xlim(0, 255)
plt.xlabel('Pixel intensity')
plt.ylabel('Pixel count')
plt.title('BGR color histograms')
plt.show()
```

### A2. Prewitt gradients

Manual §5.1 names Prewitt but provides no separate implementation.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
prewitt_y = prewitt_x.T
gx = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
gy = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
magnitude = np.hypot(gx, gy)

plt.imshow(magnitude, cmap='gray')
plt.title('Prewitt gradient magnitude')
plt.axis('off')
plt.show()
```

### A3. Laplacian of Gaussian (LoG) response + zero crossings

Manual §5.1 mentions LoG; this is supplementary code, not a transcription.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

smooth = cv2.GaussianBlur(image, (5, 5), 1)
log_response = cv2.Laplacian(smooth, cv2.CV_64F, ksize=3)

# Mark pixel neighborhoods whose Laplacian has both signs and
# sufficient response range (a simple zero-crossing approximation).
max_neighborhood = cv2.dilate(log_response, np.ones((3, 3), np.uint8))
min_neighborhood = cv2.erode(log_response, np.ones((3, 3), np.uint8))
threshold = 15.0
zero_crossings = ((max_neighborhood > 0) &
                  (min_neighborhood < 0) &
                  ((max_neighborhood - min_neighborhood) > threshold))
edges = zero_crossings.astype(np.uint8) * 255

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.imshow(log_response, cmap='gray')
plt.title('LoG response')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Approx. zero-crossings')
plt.axis('off')
plt.tight_layout()
plt.show()
```

### A4. Material texture descriptor using LBP + HOG

The material classification exercise appears on PDF p. 15; it does not include a supplied solution. This example extracts features but does not train a classifier.

```python
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

# Resize before extracting features so vectors have fixed sizes.
image = cv2.resize(image, (128, 128))

lbp = local_binary_pattern(image, 8, 1, method='uniform')
lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11))
lbp_hist = lbp_hist.astype(np.float64)
lbp_hist /= max(lbp_hist.sum(), 1)

hog_vector = hog(image, orientations=9,
                 pixels_per_cell=(8, 8), cells_per_block=(2, 2))
hog_vector = hog_vector.astype(np.float64)
hog_vector /= max(np.linalg.norm(hog_vector), 1e-12)

combined_features = np.concatenate([lbp_hist, hog_vector])
print('LBP shape:', lbp_hist.shape)
print('HOG shape:', hog_vector.shape)
print('Combined descriptor shape:', combined_features.shape)
# Apply consistent scaling/normalization on training data before ML.
```

---

## Critical corrections in one glance

| Manual issue | Correct approach |
|---|---|
| Gaussian smoothing uses only a 5×1 Gaussian column from `getGaussianKernel` | Use `cv2.GaussianBlur` or the outer product `g @ g.T` for a full 2D kernel |
| HED histogram uses all image pixels and excludes negative angles | Mask orientations by actual edge pixels and wrap with `% 360` |
| HIG histogram ignores negative orientations | Convert to unsigned `% 180`; optionally weight by gradient magnitude |
| Texture contrast implemented as a local sum | Use local standard deviation from `E[I²] - E[I]²` |
| Squaring an 8-bit image can overflow | Convert to float **before** squaring |
| `BORDER_CONSTANT` called valid convolution | Same-sized filtered image must be cropped for true valid output |
| `BORDER_REFLECT` called zero padding | Use `BORDER_CONSTANT` for zero padding |
| `cv2.filter2D(..., strides=(2,2))` | Invalid argument; crop to valid then take `[::2, ::2]` |
| `filter2D` called true mathematical convolution | It does correlation; flip an asymmetric kernel for mathematical convolution |
| 8-bit derivative outputs may clip negative values | Use `cv2.CV_32F` or `cv2.CV_64F` |

## Exam formulas

- Gradient magnitude: `np.hypot(gx, gy)` = √(Gx² + Gy²)
- Gradient orientation: `np.degrees(np.arctan2(gy, gx))` (radians → degrees)
- Unsigned orientation: `angle % 180` | signed full-circle orientation: `angle % 360`
- Local standard deviation: `sqrt(max(E[I²] − E[I]², 0))`
- General convolution output (per dimension): `floor((N + 2P − K)/S) + 1`
- Valid/no padding (per dimension): `floor((N − K)/S) + 1`
- "Same" with stride 1 and odd kernel: choose padding to preserve input height and width

**Study tip:** Know which operations produce **an image** (filters, Canny) and which produce **a feature vector** (HOG, LBP histogram, HED/HIG histograms). The former can feed into the latter.
