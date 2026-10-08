# Lab 04 — Exam-Style Solved Coding Questions

**Course:** AI-4002 Computer Vision Lab (FAST-NUCES Karachi)  
**Basis:** `Lab 04 Manual.pdf` — feature extraction, filtering and convolution, edge detection  
**Status:** **13 original practice questions with corrected solutions**; these are **not** past-paper questions.  

Each question has a problem statement, a **standalone** Python solution, an explanation of the key steps, and an exam variation. Most questions assume an image called `image.jpg` in your working directory; the **Practice input generator** below can create one. Question 13 additionally needs a labeled material dataset; optional synthetic training images are supplied so the entire exercise can run.

**Install once:**

```bash
pip install numpy opencv-python matplotlib scikit-image scikit-learn
```

> **Correctness notes:** `cv2.filter2D` performs **correlation**, not mathematical convolution (the latter flips the kernel). It preserves spatial dimensions and does not take a `strides=` argument. Negative gradient responses require a signed/float output depth. HED is computed at detected edge pixels. Local contrast means **standard deviation**, not a box-filtered intensity image.

## Practice input generator (optional)

Run this once if you have no `image.jpg`. The generated image contains shapes, gradients, and texture so the algorithms have meaningful signals.

```python
import cv2
import numpy as np
from pathlib import Path

if not Path('image.jpg').exists():
    rng = np.random.default_rng(10)
    h, w = 160, 192
    yy, xx = np.indices((h, w))
    base = 45 + 0.65 * xx + 24 * np.sin(yy / 5)
    image = np.clip(base + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    cv2.rectangle(image, (22, 18), (102, 93), 220, -1)
    cv2.circle(image, (135, 100), 29, 15, -1)
    cv2.line(image, (10, 145), (180, 24), 250, 3)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color, (22, 18), (102, 93), (40, 200, 245), -1)
    cv2.circle(color, (135, 100), 29, (190, 20, 30), -1)
    cv2.line(color, (10, 145), (180, 24), (50, 255, 50), 3)
    cv2.imwrite('image.jpg', color)
    print('Created image.jpg with distinct colors')
else:
    print('Using existing image.jpg')
```

---

## Q1. Remove noise using averaging and Gaussian filtering

**Difficulty:** Easy  
**Manual reference:** §4.1–4.2, PDF pp. 16–17

**Question:** Read a grayscale image, add artificial Gaussian noise, apply a 5×5 mean filter and a 5×5 Gaussian filter, and display four images: original, noisy, mean-filtered, Gaussian-filtered.

### Solution

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

rng = np.random.default_rng(42)
noise = rng.normal(0, 20, image.shape)
noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

mean_filtered = cv2.blur(noisy, (5, 5))
gaussian_filtered = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.2)

fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for ax, (title, img) in zip(axes, [
    ('Original', image), ('Noisy', noisy),
    ('Mean blur', mean_filtered), ('Gaussian blur', gaussian_filtered)
]):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()

print('Mean absolute change after Gaussian blur:',
      np.mean(np.abs(noisy.astype(float) - gaussian_filtered.astype(float))))
```

**Why it works:** `cv2.blur` assigns equal weights in the 5×5 window. `cv2.GaussianBlur` weights nearer pixels more heavily. Both reduce random noise but also remove some fine details.

**Exam variation:** Replace 5×5 by 9×9. Which image loses more detail? Experiment with `sigmaX=0.5` and `sigmaX=3`.

---

## Q2. Implement an emboss filter and a sharpening filter

**Difficulty:** Easy  
**Manual reference:** §4.2, PDF p. 17

**Question:** Create an embossed image with a custom kernel and a sharpened image with a high-pass kernel. Display the input and both outputs.

### Solution

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
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

emboss_response = cv2.filter2D(image, cv2.CV_32F, emboss_kernel)
embossed = np.clip(emboss_response + 128, 0, 255).astype(np.uint8)
sharpened = cv2.filter2D(image, -1, sharpen_kernel)

fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, (title, img) in zip(axes, [
    ('Original', image), ('Emboss', embossed), ('Sharpened', sharpened)
]):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**Why it works:** The emboss kernel has a zero sum and creates positive/negative directional responses. Adding 128 maps flat regions to middle gray. The sharpening kernel strengthens local pixel differences.

**Exam variation:** Change the sign of every emboss coefficient. How does the apparent lighting direction change?

---

## Q3. Implement valid, same, and strided convolution manually

**Difficulty:** Hard  
**Manual reference:** §4.2.1, PDF pp. 17–19

**Question:** Write a NumPy-based function for **mathematical convolution** (not correlation). For a 7×7 patch and a 3×3 kernel, compute valid/no-padding stride 1, same/zero-padding stride 1, and valid stride 2. Compare the same output with a correctly flipped OpenCV kernel.

### Solution

```python
import cv2
import numpy as np

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
patch = image[:7, :7].astype(np.float64)

kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float64)

def convolve2d(image2d, kernel2d, padding=0, stride=1):
    if padding < 0 or stride < 1:
        raise ValueError('padding must be >= 0 and stride must be >= 1')
    padded = np.pad(image2d, padding, mode='constant')
    flipped = np.flip(kernel2d, axis=(0, 1))  # True convolution
    kh, kw = kernel2d.shape
    oh = (padded.shape[0] - kh) // stride + 1
    ow = (padded.shape[1] - kw) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError('Kernel does not fit in the padded image')
    output = np.zeros((oh, ow), dtype=np.float64)
    for y in range(oh):
        for x in range(ow):
            window = padded[y*stride:y*stride+kh, x*stride:x*stride+kw]
            output[y, x] = np.sum(window * flipped)
    return output

valid = convolve2d(patch, kernel, padding=0, stride=1)
same = convolve2d(patch, kernel, padding=1, stride=1)
stride2 = convolve2d(patch, kernel, padding=0, stride=2)

opencv_same = cv2.filter2D(
    patch, cv2.CV_64F, np.flip(kernel, axis=(0, 1)),
    borderType=cv2.BORDER_CONSTANT
)
print('Input shape:', patch.shape)
print('Valid shape:', valid.shape)   # (5, 5)
print('Same shape:', same.shape)     # (7, 7)
print('Stride-2 shape:', stride2.shape)  # (3, 3)
assert valid.shape == (5, 5)
assert same.shape == (7, 7)
assert stride2.shape == (3, 3)
assert np.allclose(same, opencv_same)
print('Manual same convolution matches OpenCV with flipped kernel.')
```

**Why it works:** Output size per dimension is `floor((N+2P-K)/S)+1`. Convolution rotates the kernel 180°. `filter2D()` performs correlation and cannot directly specify a stride.

**Exam variation:** Calculate the output size for a 9×9 input, 5×5 kernel, P=1 and S=2 before running code.

---

## Q4. Detect strong boundaries with Sobel gradients

**Difficulty:** Medium  
**Manual reference:** §4.2, §5.1, PDF pp. 17, 20–21

**Question:** Calculate Sobel `Gx`, `Gy`, gradient magnitude and orientation. Threshold the magnitude to produce a binary edge map and display all intermediate results.

### Solution

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

smooth = cv2.GaussianBlur(image, (5, 5), 1)
gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.hypot(gx, gy)
orientation = np.degrees(np.arctan2(gy, gx)) % 360

magnitude_8bit = cv2.normalize(magnitude, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
_, binary_edges = cv2.threshold(magnitude_8bit, 90, 255,
                                cv2.THRESH_BINARY)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
items = [
    ('Original', image), ('Sobel Gx', gx), ('Sobel Gy', gy),
    ('Magnitude', magnitude_8bit), ('Direction (degrees)', orientation),
    ('Thresholded edges', binary_edges)
]
for ax, (title, img) in zip(axes.flat, items):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Gradient magnitude max:', magnitude.max())
```

**Why it works:** `Gx` and `Gy` are directional derivatives, not image colors. `np.hypot` gives magnitude. `arctan2` gives angle. `CV_64F` preserves negative gradients. Thresholding magnitude yields thick/noisy edges compared with Canny.

**Exam variation:** Find and print the number of edge pixels at thresholds 50, 100, 150.

---

## Q5. Compare Canny edges at three threshold pairs

**Difficulty:** Easy  
**Manual reference:** §5.2, PDF pp. 21–23

**Question:** Run Canny after Gaussian smoothing with `(30,90)`, `(70,140)`, and `(140,220)` thresholds. Display results and print edge-pixel counts.

### Solution

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
smooth = cv2.GaussianBlur(image, (5, 5), 1)

settings = [(30, 90), (70, 140), (140, 220)]
fig, axes = plt.subplots(1, 4, figsize=(14, 3))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for ax, (low, high) in zip(axes[1:], settings):
    edges = cv2.Canny(smooth, low, high, L2gradient=True)
    count = int(np.count_nonzero(edges))
    ax.imshow(edges, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Canny {low}/{high}\n{count} edge pixels')
    ax.axis('off')
    print(f'Thresholds {low}/{high}: {count} edge pixels')
plt.tight_layout()
plt.show()
```

**Why it works:** Canny performs smoothing (provide it explicitly), gradient estimation, non-maximum suppression, double thresholding and hysteresis. Low thresholds usually admit more edge candidates; higher thresholds are more selective.

**Exam variation:** Keep thresholds fixed but change Gaussian kernel size from 3×3 to 9×9.

---

## Q6. Compute grayscale and BGR color histograms

**Difficulty:** Easy  
**Manual reference:** §3.3, PDF p. 9

**Question:** Read a color image, build a 256-bin histogram for each BGR channel plus the grayscale version, and identify the dominant intensity bin for each channel.

### Solution

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 4))
for idx, color in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([image], [idx], None, [256], [0, 256]).ravel()
    plt.plot(np.arange(256), hist, color=color,
             label=['Blue', 'Green', 'Red'][idx])
    print(f'{color.upper()} dominant intensity:', np.argmax(hist))

gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
plt.plot(np.arange(256), gray_hist, color='black', alpha=0.6,
         label='Gray')
plt.xlabel('Intensity (0-255)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Image intensity histograms')
plt.tight_layout()
plt.show()
print('Total pixels represented:', int(gray_hist.sum()))
assert int(gray_hist.sum()) == image.shape[0] * image.shape[1]
```

**Why it works:** OpenCV images use **BGR**, not RGB. A histogram counts how often each intensity appears but does not preserve the positions of those colors.

**Exam variation:** Use a binary mask to calculate the histogram of only the center region with `cv2.calcHist`.

---

## Q7. Extract and visualize HOG features

**Difficulty:** Medium  
**Manual reference:** §3.1, PDF pp. 6–7

**Question:** Resize the input to 64×128, compute HOG with 9 orientations, 8×8 pixels/cell and 2×2 cells/block. Display the visualization and print the descriptor length.

### Solution

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
resized = cv2.resize(image, (64, 128))  # (width, height)

features, hog_vis = hog(
    resized, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), block_norm='L2-Hys',
    visualize=True
)
print('HOG vector length:', len(features))
assert len(features) == 3780

fig, axes = plt.subplots(1, 2, figsize=(7, 5))
axes[0].imshow(resized, cmap='gray')
axes[0].set_title('64×128 input')
axes[1].imshow(hog_vis, cmap='gray')
axes[1].set_title('HOG visualization')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**Why it works:** 128/8=16 cells vertically, 64/8=8 horizontally. With overlapping 2×2 blocks: `(16-1)*(8-1)*2*2*9 = 3780` features. The visualization is **not** the feature vector.

**Exam variation:** Set `pixels_per_cell=(16,16)` and calculate the expected feature-vector length before running it.

---

## Q8. Create normalized LBP histograms at two radii

**Difficulty:** Medium  
**Manual reference:** §3.2, PDF pp. 7–9

**Question:** Compute uniform Local Binary Patterns at `(P=8,R=1)` and `(P=16,R=2)`. Draw a histogram for each and verify its total probability is 1.

### Solution

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for ax, (P, R) in zip(axes, [(8, 1), (16, 2)]):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(P+3))
    hist = hist.astype(float)
    hist /= max(hist.sum(), 1)
    print(f'P={P}, R={R}: length={len(hist)}, sum={hist.sum():.5f}')
    assert len(hist) == P + 2
    assert np.isclose(hist.sum(), 1)
    ax.bar(np.arange(P+2), hist)
    ax.set_title(f'Uniform LBP P={P}, R={R}')
    ax.set_xlabel('LBP bin')
    ax.set_ylabel('Relative frequency')
plt.tight_layout()
plt.show()
```

**Why it works:** Each pixel gets a pattern code from neighbor comparisons. The histogram of uniform LBP (`method="uniform"`) contains `P+2` categories: 10 when P=8 and 18 when P=16.

**Exam variation:** Replace the image with its Gaussian-blurred version. Which LBP patterns change the most?

---

## Q9. Build the Histogram of Edge Directions (HED)

**Difficulty:** Medium  
**Manual reference:** §3.4, PDF pp. 9–11

**Question:** Detect Canny edges, compute Sobel gradient orientations, and make an 8-bin histogram from **only the pixels that are edges**.

### Solution

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
angles = np.degrees(np.arctan2(gy, gx)) % 360

edge_angles = angles[edges > 0]
hist, bins = np.histogram(edge_angles, bins=8, range=(0, 360))
hist = hist.astype(float)
hist /= max(hist.sum(), 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].imshow(edges, cmap='gray')
axes[0].set_title('Canny edge map')
axes[0].axis('off')
axes[1].bar(bins[:-1], hist, width=np.diff(bins), align='edge')
axes[1].set_xlim(0, 360)
axes[1].set_xlabel('Gradient orientation (degrees)')
axes[1].set_title('HED: 8 bins')
plt.tight_layout()
plt.show()
print('HED descriptor:', hist)
```

**Why it works:** The edge mask `edges > 0` is essential: a general full-image orientation histogram is not edge-restricted HED. Gradient directions are perpendicular to the geometric direction along edges.

**Exam variation:** Use 18 bins instead of 8 and discuss the trade-off in angular resolution.

---

## Q10. Build a magnitude-weighted HIG descriptor

**Difficulty:** Medium  
**Manual reference:** §3.5, PDF pp. 11–13

**Question:** Compute a 9-bin global Histogram of Intensity Gradients over unsigned 0°–180° orientations. Weight each histogram entry by its gradient magnitude.

### Solution

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
angles = np.degrees(np.arctan2(gy, gx)) % 180
valid = magnitude > 1e-12

hist, bins = np.histogram(
    angles[valid], bins=9, range=(0, 180),
    weights=magnitude[valid]
)
hist = hist.astype(float)
hist /= max(hist.sum(), 1e-12)

plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge')
plt.xlim(0, 180)
plt.xlabel('Gradient angle')
plt.ylabel('Normalized gradient strength')
plt.title('Global HIG descriptor')
plt.show()
print('HIG length:', len(hist))
print('HIG vector:', hist)
```

**Why it works:** HIG here is one global weighted orientation histogram, not HOG. It does **not** use HOG cells or block normalization. Ignoring near-zero magnitudes avoids arbitrary angles where no gradient exists.

**Exam variation:** Make an unweighted histogram and compare it with the weighted one.

---

## Q11. Compute texture-energy and local-contrast histograms

**Difficulty:** Hard  
**Manual reference:** §3.6, PDF pp. 13–15

**Question:** For each 5×5 neighborhood, calculate squared-intensity **energy** as a sum and **contrast** as standard deviation. Display both images and their normalized histograms.

### Solution

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
image_f = image.astype(np.float64)
window = (5, 5)

energy = cv2.boxFilter(image_f**2, -1, window, normalize=False)
local_mean = cv2.blur(image_f, window)
local_mean_sq = cv2.blur(image_f**2, window)
variance = np.maximum(local_mean_sq - local_mean**2, 0)
contrast = np.sqrt(variance)

energy_hist, energy_bins = np.histogram(energy.ravel(), bins=32)
contrast_hist, contrast_bins = np.histogram(contrast.ravel(), bins=32)
energy_hist = energy_hist / max(energy_hist.sum(), 1)
contrast_hist = contrast_hist / max(contrast_hist.sum(), 1)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes[0, 0].imshow(energy, cmap='gray')
axes[0, 0].set_title('Local energy image')
axes[0, 1].imshow(contrast, cmap='gray')
axes[0, 1].set_title('Local contrast (std) image')
axes[1, 0].bar(energy_bins[:-1], energy_hist,
               width=np.diff(energy_bins), align='edge')
axes[1, 0].set_title('Energy histogram')
axes[1, 1].bar(contrast_bins[:-1], contrast_hist,
               width=np.diff(contrast_bins), align='edge')
axes[1, 1].set_title('Contrast histogram')
for ax in axes[0]:
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Mean local energy:', energy.mean())
print('Mean local contrast:', contrast.mean())
```

**Why it works:** Convert to float **before squaring**. Energy is `sum(I²)` within the window. Contrast is `sqrt(E[I²] − (E[I])²)`. An all-ones filter applied directly to the original image does **not** calculate standard deviation.

**Exam variation:** Create a constant-valued grayscale image. Its interior contrast should be zero even though energy is positive.

---

## Q12. Compare Sobel, Scharr, Prewitt and Laplacian-of-Gaussian

**Difficulty:** Hard  
**Manual reference:** §5.1, PDF pp. 20–21

**Question:** Implement four edge-response methods on the same grayscale image. Display comparable normalized magnitude maps.

### Solution

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('image.jpg')
smooth = cv2.GaussianBlur(image, (5, 5), 1)

sx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.hypot(sx, sy)

cx = cv2.Scharr(smooth, cv2.CV_64F, 1, 0)
cy = cv2.Scharr(smooth, cv2.CV_64F, 0, 1)
scharr = np.hypot(cx, cy)

px_kernel = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float64)
px = cv2.filter2D(smooth, cv2.CV_64F, px_kernel)
py = cv2.filter2D(smooth, cv2.CV_64F, px_kernel.T)
prewitt = np.hypot(px, py)

log_response = cv2.Laplacian(smooth, cv2.CV_64F, ksize=3)
log_magnitude = np.abs(log_response)

def to_8bit(array):
    return cv2.normalize(array, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, (title, response) in zip(axes.flat, [
    ('Sobel magnitude', sobel), ('Scharr magnitude', scharr),
    ('Prewitt magnitude', prewitt), ('LoG absolute response', log_magnitude)
]):
    ax.imshow(to_8bit(response), cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**Why it works:** Sobel, Scharr and Prewitt are first-order gradient estimators. LoG combines Gaussian smoothing with a second-order Laplacian. **Absolute LoG response is not a complete zero-crossing edge map**; that needs an extra zero-crossing step.

**Exam variation:** Threshold each normalized response at 100 and compare the binary output.

---

## Optional practice materials for Q13 — synthetic demonstration only

This generator is **not part of the manual**. It creates three obviously different artificial texture classes so you can practice the pipeline without downloading a dataset. It does **not** validate a material classifier on real photographs. Run it only if you do not already have a `materials/` directory.

```python
from pathlib import Path
import cv2
import numpy as np

root = Path('materials')
if not root.exists():
    rng = np.random.default_rng(2)
    yy, xx = np.indices((96, 96))
    for category in ('fabric', 'metal', 'wood'):
        (root / category).mkdir(parents=True, exist_ok=True)
        for i in range(24):
            noise = rng.normal(0, 1, (96, 96))
            if category == 'metal':
                texture = 145 + rng.uniform(1, 4) * noise
            elif category == 'wood':
                texture = (120 + 44 * np.sin(yy / 4.0 + np.sin(xx / 16))
                           + 7 * noise)
            else:  # fabric: cross-hatched pattern
                texture = (115 + 42 * np.sin(xx / 3.0)
                           + 42 * np.sin(yy / 3.0) + 7 * noise)
            texture = np.clip(texture, 0, 255).astype(np.uint8)
            cv2.imwrite(str(root / category / f'{i:02d}.png'), texture)
    print('Created 72 synthetic sample material images')
else:
    print('Existing materials/ kept unchanged')
```

---

## Q13. Train a simple material classifier using texture features

**Difficulty:** Challenge  
**Manual reference:** Manual material-classification task, PDF p. 15

**Question:** Assume labeled images are stored under `materials/wood/`, `materials/metal/`, and `materials/fabric/`. Extract **uniform LBP** plus a local-contrast histogram and use 3-nearest neighbors to classify material types. Display held-out accuracy. If you do not have material images, run the optional synthetic dataset generator below first.

### Solution

```python
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

root = Path('materials')
classes = ['fabric', 'metal', 'wood']

def texture_features(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(str(image_path))
    gray = cv2.resize(gray, (96, 96))

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= max(lbp_hist.sum(), 1)

    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (5, 5))
    mean_sq = cv2.blur(gray_f**2, (5, 5))
    contrast = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    contrast_hist, _ = np.histogram(contrast, bins=16, range=(0, 128))
    contrast_hist = contrast_hist.astype(float)
    contrast_hist /= max(contrast_hist.sum(), 1)

    return np.hstack([lbp_hist, contrast_hist])

X, y = [], []
for label in classes:
    images = sorted([p for p in (root / label).glob('*')
                     if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if len(images) < 4:
        raise ValueError(f'Need >= 4 images in {root/label}; found {len(images)}')
    for path in images:
        X.append(texture_features(path))
        y.append(label)

X = np.asarray(X)
y = np.asarray(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('Feature-vector shape:', X.shape)   # (samples, 26)
assert X.shape[1] == 26
print('Held-out accuracy:', accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
```

**Why it works:** Each image becomes a 26-dimensional feature vector: 10 uniform-LBP frequencies plus 16 local-contrast frequencies. A train/test split estimates performance on held-out samples. Synthetic textures below only demonstrate the pipeline—they do not establish real-world accuracy.

**Exam variation:** Replace `KNeighborsClassifier` with `RandomForestClassifier(random_state=42)`; compare scores on **real**, properly separated test data.

---

## Last-minute formula and API reference

| Concept | Formula / reminder |
|---|---|
| Gradient magnitude | `np.hypot(gx, gy)` = sqrt(Gx²+Gy²) |
| Gradient orientation | `np.degrees(np.arctan2(gy, gx)) % 360` |
| Output dimension | `floor((N+2P-K)/S)+1` (independently for H and W) |
| Local contrast (std) | `sqrt(E[I²] - E[I]²)` |
| Local squared-intensity energy | `sum(I²)` in a neighborhood |
| Uniform LBP length | `P+2` (for scikit-image `method='uniform'`) |
| 64×128 HOG length | 3780 for 9 bins, 8×8 cells, 2×2 blocks, 1-cell block stride |
| Signed derivatives | `cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)` |
| Correct color order | OpenCV is **BGR**, `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)` for RGB display |
| Canny | `cv2.Canny(smoothed, low, high, L2gradient=True)` |

**Exam challenge:** First solve Q3, Q4, Q8, Q9 and Q11 with the solution hidden. Then complete Q13 using either a labeled dataset or the synthetic demo textures, and explain the limitations.
