# COMPUTER VISION LAB MID EXAM CHEATSHEET

**FAST-NUCES | AI-4002 | Labs 01-06 | Python + OpenCV**

Designed for an open-book lab exam: runnable patterns first, theory only where it helps choose the right code.

# 0. How to use this sheet

- Find the question keyword (threshold / homography / Hough / SIFT / watershed / etc.), jump to that lab, copy the smallest matching template, then change paths/parameters.

- OpenCV images are NumPy arrays indexed image[y, x]. Geometric points are usually written (x, y). OpenCV color is BGR; Matplotlib expects RGB.

- For binary masks use uint8 values 0 and 255. For calculations that may go negative or exceed 255, use float32/float64 and clip/convert at the end.

- Every code block below is exam-safe Python/OpenCV syntax. Where a source example was unsafe/invalid, the corrected version is shown.

## Coverage checklist

| Lab | Included material / tasks |
| --- | --- |
| 01 | Python/OOP + image I/O, display, RGB/gray, resize, crop, blur, drawing/text, threshold, rotate, masks/bitwise, blend/add, equalization, Pandas stats; Tasks 1-9 |
| 02 | Histogram/equalization, colormap/color balance, threshold+mask, log/gamma, CT-MRI fusion, stats/save, real-time video processing; Tasks 1-3 |
| 03 | Linear/affine/projective hierarchy + scale, rotate, shear, translate, rigid, similarity, general affine, perspective, homography/panorama, composition; Tasks 1-10 |
| 04 | HOG, LBP, color histogram, HED, HIG, texture energy/contrast, filtering/convolution, padding/stride, Sobel/Scharr/Prewitt/LoG/Canny |
| 05 | Global/adaptive/Otsu/HSV/Canny segmentation, region growing, watershed, marker experiments, K-means, comparison, active contour; Tasks 1-10 |
| 06 | Wavelets, Sobel/Canny/LoG, Hough lines/circles, SIFT, matching/homography + all 8 task scenarios (screen, asset, anomaly, video recognition, panorama, lanes, coins, security) |

## Critical source-code traps corrected

- cv2.filter2D() does NOT accept a strides= argument. Filter normally, crop to valid area if required, then subsample with result[::sy, ::sx].

- VideoWriter frame size must exactly match every frame written. If you horizontally stack RAW + ENHANCED, construct the writer with (width*2, height) and write only the stacked frame.

- cv2.getGaussianKernel() returns a 1-D column kernel. For filter2D, make a 2-D Gaussian kernel with g @ g.T, or simply use cv2.GaussianBlur().

- Histogram/Otsu/adaptive thresholding operate on single-channel grayscale images unless you intentionally threshold a selected channel.

- If Matplotlib colors look wrong, convert BGR -> RGB before plt.imshow().

## Universal imports + I/O

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")
if img is None:
    raise FileNotFoundError("image.jpg not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

plt.imshow(rgb); plt.axis("off"); plt.show()
cv2.imwrite("output.png", img)
```

## Universal display/comparison pattern

```python
images = [rgb, gray]
titles = ["Original", "Gray"]
plt.figure(figsize=(12, 5))
for i, (im, title) in enumerate(zip(images, titles), start=1):
    plt.subplot(1, len(images), i)
    plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
    plt.title(title); plt.axis("off")
plt.tight_layout(); plt.show()
```

# LAB 01 - Python + Core OpenCV

## Task 1 - OOP Grocery Manager

```python
class GroceryManager:
    def __init__(self):
        self.items = []

    def add_item(self, item, quantity, price):
        self.items.append({"item": item, "quantity": quantity, "price": price})

    def remove_item(self, item):
        for x in self.items:
            if x["item"] == item:
                self.items.remove(x)
                return True
        return False

    def view_list(self):
        for x in self.items:
            print(x)

    def calculate_total(self):
        return sum(x["quantity"] * x["price"] for x in self.items)

g = GroceryManager()
g.add_item("banana", 12, 1.2)
g.view_list()
print(g.calculate_total())
```

## Task 2 - Student dictionary + search/maximum

```python
students = {
    73: {"Name":"Rayyan", "Major":"Artificial Intelligence", "Grades":[88,86,92,91]},
    74: {"Name":"Malik",  "Major":"Artificial Intelligence", "Grades":[90,87,94,89]},
}

def highest_avg_grade(students):
    best_id = max(students, key=lambda sid: np.mean(students[sid]["Grades"]))
    avg = np.mean(students[best_id]["Grades"])
    print(f"{students[best_id]['Name']}: {avg:.2f}")

def search_major(students, major):
    for sid, data in students.items():
        if data["Major"] == major:
            print(sid, data["Name"])

highest_avg_grade(students)
search_major(students, "Artificial Intelligence")
```

## Task 3 - Safe image load + RGB display

```python
image = cv2.imread("Sukuna.jpg")
if image is None:
    print("Image not found")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6)); plt.imshow(image_rgb)
    plt.title("Image"); plt.axis("off"); plt.show()
```

## Manual basics - grayscale, resize, crop

```python
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize by factor
small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
# Resize to exact size (width, height)
resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)

# Crop = NumPy slicing: image[y1:y2, x1:x2]
h, w = image.shape[:2]
left_half = image[:, :w//2]
roi = image[100:300, 200:500]
```

## Task 4 - blank canvas + geometry

```python
canvas = np.zeros((800, 800, 3), dtype=np.uint8)
center = (400, 400)
for r in [250, 200, 150, 100, 50]:
    cv2.circle(canvas, center, r, (255,255,255), 2)
cv2.rectangle(canvas, (150,150), (650,650), (0,0,255), 4)
cv2.line(canvas, (0,0), (799,799), (0,255,0), 2)

# Filled shape => thickness=-1
cv2.circle(canvas, center, 30, (255,0,0), -1)
```

## Task 5 - Gaussian blur + center ROI

```python
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
blurred = cv2.GaussianBlur(rgb, (25,25), 0)   # kernel sizes must be odd
h, w = rgb.shape[:2]; cx, cy = w//2, h//2
half = 150
roi_original = rgb[cy-half:cy+half, cx-half:cx+half]
roi_blurred  = blurred[cy-half:cy+half, cx-half:cx+half]
```

## Task 6 - alpha blending + text overlay

```python
h, w = image.shape[:2]
start_y = int(h * 0.80)
overlay = image.copy()
cv2.rectangle(overlay, (0,start_y), (w,h), (255,0,0), -1)
blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
cv2.putText(blended, "Hello", (50,h-50), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (255,255,255), 3, cv2.LINE_AA)
```

## Manual - add vs weighted blend

```python
# Images MUST have same shape/type
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
added = cv2.add(img1, img2)              # saturated addition, may become very bright
blend = cv2.addWeighted(img1, .6, img2, .4, 0)  # alpha*img1 + beta*img2 + gamma
```

## Task 7 - global/adaptive threshold + rotation

```python
gray = cv2.imread("document.png", cv2.IMREAD_GRAYSCALE)
_, global_t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
adaptive = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

h, w = adaptive.shape; center=(w//2, h//2)
M = cv2.getRotationMatrix2D(center, 45, 0.8)
rotated = cv2.warpAffine(adaptive, M, (w,h))
```

## Task 8 - binary mask + bitwise compositing

```python
img1 = cv2.imread("Sukuna.jpg")
img2 = cv2.resize(cv2.imread("image.jpg"), (img1.shape[1], img1.shape[0]))
h, w = img1.shape[:2]
mask = np.zeros((h,w), dtype=np.uint8)
cv2.circle(mask, (w//2,h//2), min(w,h)//4, 255, -1)

fg = cv2.bitwise_and(img1, img1, mask=mask)
inv = cv2.bitwise_not(mask)
bg = cv2.bitwise_and(img2, img2, mask=inv)
result = cv2.bitwise_or(fg, bg)
```

## Task 9 - image statistics with Pandas

```python
import pandas as pd
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
df = pd.DataFrame({
    "Red":   rgb[:,:,0].ravel(),
    "Green": rgb[:,:,1].ravel(),
    "Blue":  rgb[:,:,2].ravel(),
})
print(df.describe())
```

## Manual - histogram equalization

```python
gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
equalized = cv2.equalizeHist(gray)
plt.hist(gray.ravel(), 256, [0,256]); plt.show()
plt.hist(equalized.ravel(), 256, [0,256]); plt.show()
```

# LAB 02 - Enhancement, Fusion, Video

## Task 1 - Chest X-ray enhancement pipeline

```python
from pathlib import Path
DATA_DIR = Path("data"); OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
img = cv2.imread(str(DATA_DIR/"xray.png"), cv2.IMREAD_GRAYSCALE)
if img is None: raise FileNotFoundError("xray.png")
print(img.shape, img.dtype, img.min(), img.max(), img.mean())

# Histogram + equalization
plt.hist(img.ravel(), bins=256, range=[0,256]); plt.show()
eq = cv2.equalizeHist(img)

# Pseudocolor
heat = cv2.applyColorMap(eq, cv2.COLORMAP_JET)

# High-intensity threshold + masked extraction
_, mask = cv2.threshold(eq, 200, 255, cv2.THRESH_BINARY)
filtered = cv2.bitwise_and(eq, eq, mask=mask)
```

## Gray-world color balance

```python
def gray_world_balance(bgr):
    f = bgr.astype(np.float32)
    b,g,r = cv2.split(f)
    mb,mg,mr = b.mean(), g.mean(), r.mean()
    target = (mb+mg+mr)/3.0; eps=1e-6
    b *= target/(mb+eps); g *= target/(mg+eps); r *= target/(mr+eps)
    return np.clip(cv2.merge([b,g,r]),0,255).astype(np.uint8)

balanced = gray_world_balance(heat)
```

## Log and gamma transforms

```python
def log_transform(image):
    f = image.astype(np.float32)
    mx = f.max()
    if mx == 0: return np.zeros_like(image)
    c = 255.0 / np.log1p(mx)
    return np.clip(c*np.log1p(f),0,255).astype(np.uint8)

def gamma_transform(image, gamma=0.6):
    norm = image.astype(np.float32)/255.0
    return np.clip(255*np.power(norm, gamma),0,255).astype(np.uint8)

log_img = log_transform(img)
gamma_img = gamma_transform(img, .6)
```

EXAM NOTE: Gamma < 1 usually brightens dark regions; gamma > 1 darkens/suppresses bright regions. Log expands dark intensities and compresses bright ones.

## Task 2 - CT/MRI fusion

```python
ct  = cv2.imread("ct.png",  cv2.IMREAD_GRAYSCALE)
mri = cv2.imread("mri.png", cv2.IMREAD_GRAYSCALE)
if ct.shape != mri.shape:
    mri = cv2.resize(mri, (ct.shape[1], ct.shape[0]))

ct_eq, mri_eq = cv2.equalizeHist(ct), cv2.equalizeHist(mri)
ct_color  = cv2.applyColorMap(ct_eq,  cv2.COLORMAP_JET)
mri_color = cv2.applyColorMap(mri_eq, cv2.COLORMAP_HOT)
fused = cv2.addWeighted(ct_color, .7, mri_color, .3, 0)
fused_log = log_transform(fused)
fused_gamma = gamma_transform(fused, .8)

for name, im in [("CT",ct_eq),("MRI",mri_eq),("Fused",fused)]:
    print(name, "min",im.min(),"max",im.max(),"mean",im.mean(),"std",im.std())
```

## Save outputs

```python
cv2.imwrite("output/ct_equalized.png", ct_eq)
cv2.imwrite("output/mri_equalized.png", mri_eq)
cv2.imwrite("output/weighted_fusion.png", fused)
cv2.imwrite("output/fusion_log.png", fused_log)
cv2.imwrite("output/fusion_gamma.png", fused_gamma)
```

## Task 3 - real-time/video frame processing

```python
def process_frame(frame, gamma=1.2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    heat = cv2.applyColorMap(eq, cv2.COLORMAP_JET)
    bal = gray_world_balance(heat)
    logf = log_transform(bal)
    return gamma_transform(logf, gamma)

cap = cv2.VideoCapture("echo.mp4")
if not cap.isOpened(): raise FileNotFoundError("Could not open video")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("raw_vs_enhanced.mp4", fourcc, fps, (w*2, h))

import time
while True:
    ok, frame = cap.read()
    if not ok: break
    t0 = time.perf_counter()
    enhanced = process_frame(frame, 1.2)
    proc_fps = 1.0/max(time.perf_counter()-t0, 1e-9)
    raw = frame.copy()
    cv2.putText(raw,"RAW",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(enhanced,"ENHANCED",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    combined = np.hstack((raw, enhanced))
    cv2.putText(combined,f"Processing FPS: {proc_fps:.1f}",(20,h-20),
                cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    writer.write(combined)            # size exactly (2w,h)
    # cv2.imshow("Echo", combined)
    # if cv2.waitKey(max(1,int(1000/fps))) & 0xFF == ord('q'): break

cap.release(); writer.release(); cv2.destroyAllWindows()
```

## Video metadata

```python
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frames/fps if fps>0 else 0
print(fps, width, height, frames, duration)
```

# LAB 03 - Geometric Transformations

Hierarchy: Linear (2x2) -> Affine (linear + translation; 2x3/3x3 homogeneous) -> Projective/Homography (3x3). Rigid = rotation+translation. Similarity = uniform scale+rotation+translation.

## Core matrices / formulas

```python
# Point p = [x,y,1]^T.  In homogeneous form p' ~ H @ p.
# Scale                 Rotation(theta)           Translation
S = [[sx,0,0],          R = [[c,-s,0],            T = [[1,0,tx],
     [0,sy,0],               [s, c,0],                 [0,1,ty],
     [0, 0,1]]               [0, 0,1]]                 [0,0, 1]]

# X-shear / Y-shear
Hx = [[1,kx,0],[0,1,0],[0,0,1]]
Hy = [[1,0,0],[ky,1,0],[0,0,1]]

# Affine: [x';y'] = [[a,b,tx],[c,d,ty]] [x;y;1]
# Projective: x'=(h11*x+h12*y+h13)/(h31*x+h32*y+h33)
#             y'=(h21*x+h22*y+h23)/(h31*x+h32*y+h33)
```

EXAM NOTE: Matrix composition order matters: M = A @ B means B acts first, then A. OpenCV warpAffine uses 2x3; warpPerspective uses 3x3.

## Task 1 - Linear scale (300% / 3x) and keep centered

```python
image = cv2.imread("image.jpg")
h,w = image.shape[:2]
# Expanded output: 3x dimensions
scaled = cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)

# OR keep original canvas and scale about image center:
s = 3.0; cx,cy=w/2,h/2
M = np.float32([[s,0,(1-s)*cx], [0,s,(1-s)*cy]])
center_scaled = cv2.warpAffine(image, M, (w,h))
```

## Task 2 - 45-degree rotation WITHOUT clipping

```python
h,w = image.shape[:2]; cx,cy=w/2,h/2
angle = -45        # sign depending on direction needed
M = cv2.getRotationMatrix2D((cx,cy), angle, 1.0)
cos = abs(M[0,0]); sin = abs(M[0,1])
new_w = int(h*sin + w*cos)
new_h = int(h*cos + w*sin)
M[0,2] += new_w/2 - cx
M[1,2] += new_h/2 - cy
rotated = cv2.warpAffine(image, M, (new_w,new_h))
```

## Task 3 - X shear

```python
h,w = image.shape[:2]
kx = -0.5    # choose sign to undo slant
M = np.float32([[1,kx,0],[0,1,0]])
new_w = int(w + abs(kx)*h)
# If kx<0, shift right so negative x is not clipped
if kx < 0: M[0,2] = abs(kx)*h
sheared = cv2.warpAffine(image, M, (new_w,h))
```

## Task 4 - translation / homogeneous movement

```python
tx,ty = 150,80   # shift right/down; negative values shift left/up
M = np.float32([[1,0,tx],[0,1,ty]])
translated = cv2.warpAffine(image, M, (w,h))

T3 = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
```

## Task 5 - rigid transformation = rotation + translation, scale=1

```python
h,w = image.shape[:2]; center=(w//2,h//2)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
M[0,2] += 100
M[1,2] += 50
rigid = cv2.warpAffine(image, M, (w,h))
```

## Task 6 - similarity = uniform scale + rotation + translation

```python
h,w = image.shape[:2]
M = cv2.getRotationMatrix2D((w//2,h//2), 25, 0.75)  # uniform scale
M[0,2] += 80; M[1,2] += 40
similarity = cv2.warpAffine(image, M, (w,h))
```

## Task 7 - general affine from 3 point pairs

```python
src = np.float32([[50,50],[300,50],[50,300]])
dst = np.float32([[80,30],[330,80],[40,330]])
M = cv2.getAffineTransform(src, dst)   # solves 6 unknowns
out = cv2.warpAffine(image, M, (w,h))

# Manual linear-system solution (same idea):
A=[]; b=[]
for (x,y),(u,v) in zip(src,dst):
    A += [[x,y,1,0,0,0],[0,0,0,x,y,1]]
    b += [u,v]
params = np.linalg.solve(np.array(A,float), np.array(b,float))
M_manual = params.reshape(2,3)
```

## Task 8 - perspective / bird-eye view from 4 corners

```python
src = np.float32([[120,80],[500,100],[550,420],[90,450]])  # TL,TR,BR,BL
W,H = 500,400
dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
P = cv2.getPerspectiveTransform(src, dst)
birds_eye = cv2.warpPerspective(image, P, (W,H))
```

## Notebook perspective example

```python
h,w = image.shape[:2]
src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
dst = np.float32([[100,50],[w-150,0],[w-50,h-80],[50,h-20]])
P = cv2.getPerspectiveTransform(src,dst)
perspective = cv2.warpPerspective(image,P,(w,h))
```

## Task 9 - homography + simple panorama

```python
img1 = cv2.imread("left.jpg")
img2 = cv2.imread("right.jpg")
# 4+ matching points: img2 -> img1 coordinates
pts2 = np.float32([[...,...],[...,...],[...,...],[...,...]])
pts1 = np.float32([[...,...],[...,...],[...,...],[...,...]])
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
canvas_w = img1.shape[1] + img2.shape[1]
canvas_h = max(img1.shape[0], img2.shape[0])
pano = cv2.warpPerspective(img2, H, (canvas_w, canvas_h))
pano[:img1.shape[0], :img1.shape[1]] = img1
# crop black area if needed using nonzero boundingRect
```

## Task 10 - scale -> rigid -> projective as ONE 3x3 matrix

```python
theta=np.deg2rad(20); c,s=np.cos(theta),np.sin(theta)
S=np.array([[.7,0,0],[0,.7,0],[0,0,1]],float)
R=np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
T=np.array([[1,0,120],[0,1,60],[0,0,1]],float)
P=np.array([[1,0,0],[0,1,0],[1e-4,-2e-4,1]],float)  # example projective warp
M = P @ T @ R @ S       # S first, then R, T, P
out = cv2.warpPerspective(image, M, (w,h))
```

## Quick transform API chooser

| Need | API / minimum points |
| --- | --- |
| Resize only | cv2.resize / scale factors |
| Rotate/translate/rigid/similarity | getRotationMatrix2D + warpAffine |
| General affine | getAffineTransform, exactly 3 point pairs -> warpAffine |
| Perspective from 4 correspondences | getPerspectiveTransform, 4 pairs -> warpPerspective |
| Robust homography / panorama | findHomography, >=4 pairs, use RANSAC -> warpPerspective |

# LAB 04 - Feature Extraction, Filtering, Edge Detection

## HOG - Histogram of Oriented Gradients

```python
from skimage.feature import hog
from skimage import exposure
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
features, hog_image = hog(image, pixels_per_cell=(8,8), cells_per_block=(2,2),
                          visualize=True, feature_vector=True)
hog_vis = exposure.rescale_intensity(hog_image, in_range=(0,10))
print(features.shape)
```

HOG workflow: grayscale -> gradients -> cells -> orientation histogram -> block normalization -> concatenate feature vector.

## LBP - Local Binary Pattern

```python
from skimage import feature
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
radius=1; n_points=8*radius
lbp = feature.local_binary_pattern(image, n_points, radius, method="uniform")
hist,_ = np.histogram(lbp.ravel(), bins=np.arange(0,n_points+3), range=(0,n_points+2))
hist = hist.astype(float); hist /= (hist.sum()+1e-6)
print(hist)
```

## Color histogram

```python
img = cv2.imread("image.jpg")
for i, color in enumerate(["b","g","r"]):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist, label=color)
plt.legend(); plt.show()

# Histogram with mask:
hist_roi = cv2.calcHist([img],[0],mask,[256],[0,256])
```

## HED - Histogram of Edge Directions

```python
gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
sm = cv2.GaussianBlur(gray,(5,5),0)
edges = cv2.Canny(sm,30,70)
gx = cv2.Sobel(sm,cv2.CV_64F,1,0,ksize=3)
gy = cv2.Sobel(sm,cv2.CV_64F,0,1,ksize=3)
ori = (np.degrees(np.arctan2(gy,gx)) + 180) % 180
hist,bins = np.histogram(ori[edges>0], bins=8, range=(0,180))
print(hist)
```

## HIG - Histogram of Intensity Gradients

```python
gx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
gy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
mag = np.hypot(gx,gy)
ori = (np.degrees(np.arctan2(gy,gx)) + 180) % 180
# orientation distribution, optionally weighted by gradient magnitude
hist,bins = np.histogram(ori.ravel(), bins=9, range=(0,180), weights=mag.ravel())
hist = hist/(hist.sum()+1e-9)
```

## Texture energy + local contrast histograms

```python
grayf = gray.astype(np.float32); n=3
K = np.ones((n,n),np.float32)              # sum kernel
energy = cv2.filter2D(grayf**2, -1, K)
# Correct local contrast = local standard deviation
Kmean = K/(n*n)
mean = cv2.filter2D(grayf,-1,Kmean)
mean2 = cv2.filter2D(grayf**2,-1,Kmean)
contrast = np.sqrt(np.maximum(mean2 - mean**2, 0))
energy_hist, ebin = np.histogram(energy, bins=256, range=(0,energy.max()))
contrast_hist, cbin = np.histogram(contrast,bins=256,range=(0,contrast.max()))
energy_hist = energy_hist/(energy_hist.sum()+1e-9)
contrast_hist = contrast_hist/(contrast_hist.sum()+1e-9)
```

EXAM NOTE: The manual-style cv2.filter2D(image, ones_kernel) computes a neighborhood SUM, not standard deviation. If the question explicitly asks texture contrast, local std = sqrt(E[x^2]-E[x]^2) is the correct statistic.

## Filtering / convolution essentials

```python
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
# Box blur via explicit kernel
K = np.ones((3,3),np.float32)/9
box = cv2.filter2D(img,-1,K)
# Same thing built-in
box2 = cv2.blur(img,(3,3))
# Gaussian
Gauss = cv2.GaussianBlur(img,(5,5),1.0)
# 2-D Gaussian kernel for filter2D
g = cv2.getGaussianKernel(5,1.0); G2 = g @ g.T
gauss2 = cv2.filter2D(img,-1,G2)
# Salt-and-pepper noise
median = cv2.medianBlur(img,5)
# Edge-preserving smoothing
bilateral = cv2.bilateralFilter(img,9,75,75)
```

## Sobel, Scharr, Prewitt, emboss

```python
sx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
mag = cv2.magnitude(sx.astype(np.float32), sy.astype(np.float32))

scharr_x = cv2.Scharr(img,cv2.CV_64F,1,0)
scharr_y = cv2.Scharr(img,cv2.CV_64F,0,1)

prewitt_x=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],np.float32)
prewitt_y=prewitt_x.T
px=cv2.filter2D(img,cv2.CV_32F,prewitt_x); py=cv2.filter2D(img,cv2.CV_32F,prewitt_y)

embossK=np.array([[-2,-1,0],[-1,1,1],[0,1,2]],np.float32)
emboss=cv2.filter2D(img,-1,embossK)
```

## Convolution border/padding + VALID + STRIDE (correct)

```python
K=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],np.float32)
# SAME output with zero or reflected borders
same_zero = cv2.filter2D(img,-1,K,borderType=cv2.BORDER_CONSTANT)
same_ref  = cv2.filter2D(img,-1,K,borderType=cv2.BORDER_REFLECT)

# VALID: filter then discard border where 3x3 kernel was incomplete
full = cv2.filter2D(img,-1,K,borderType=cv2.BORDER_CONSTANT)
pad = K.shape[0]//2
valid = full[pad:-pad, pad:-pad]

# STRIDE 2: OpenCV filter2D has NO strides argument
valid_stride2 = valid[::2, ::2]
```

## Gradient threshold / simple edge map

```python
gx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
gy=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
mag=np.sqrt(gx**2+gy**2)
edges=(mag>100).astype(np.uint8)*255
```

## Laplacian of Gaussian (LoG)

```python
blur = cv2.GaussianBlur(img,(5,5),1.4)
lap = cv2.Laplacian(blur,cv2.CV_64F,ksize=3)
log_edges = cv2.convertScaleAbs(lap)
```

## Canny

```python
gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(gray,(5,5),1.4)
edges = cv2.Canny(blur, threshold1=100, threshold2=200)
```

Canny stages: Gaussian smoothing -> gradient magnitude/direction -> non-maximum suppression -> double threshold -> hysteresis edge tracking. Lower thresholds = more/noisier edges; higher = fewer/stronger edges.

# LAB 05 - Image Segmentation

## Thresholding family - memorize these calls

```python
gray = cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)
# Global
_, b1 = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
_, inv = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV)
# Adaptive: blockSize odd >1, C subtracted from local statistic
mean_ad = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY,31,10)
gauss_ad = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,31,10)
# Otsu: threshold value must be 0; method selects T automatically
T, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("Otsu T:",T)
```

## Task 1 - global T=50/100/150 vs adaptive

```python
Ts=[50,100,150]
globals_=[cv2.threshold(gray,T,255,cv2.THRESH_BINARY_INV)[1] for T in Ts]
adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,31,10)
```

## Task 2 - sweep adaptive Mean/Gaussian parameters

```python
block_sizes=[11,31,61]; c_values=[5,10,15]
mean_results={}; gaussian_results={}
for bs in block_sizes:
    for C in c_values:
        mean_results[(bs,C)] = cv2.adaptiveThreshold(gray,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,bs,C)
        gaussian_results[(bs,C)] = cv2.adaptiveThreshold(gray,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bs,C)
```

blockSize small -> sensitive/noisy; very large -> less local. Increasing C lowers local threshold (because threshold = local_stat - C), usually removes more background but can lose foreground.

## Task 3 - Otsu + contrast change

```python
T1,mask1=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contrast=cv2.convertScaleAbs(gray,alpha=1.3,beta=0)
T2,mask2=cv2.threshold(contrast,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(T1,T2,T2-T1)
```

## Task 4 - HSV color segmentation

```python
img=cv2.imread("smarties.png")
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower=np.array([40,80,80]); upper=np.array([85,255,255])
mask=cv2.inRange(hsv,lower,upper)
extracted=cv2.bitwise_and(img,img,mask=mask)
```

EXAM NOTE: OpenCV Hue range is 0-179, not 0-360. Use cv2.inRange for multi-channel color thresholding.

## Task 5 - Canny threshold experiment

```python
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
e1=cv2.Canny(blur,50,100)
e2=cv2.Canny(blur,50,150)
e3=cv2.Canny(blur,50,250)
```

## Task 6 - region growing (4-connected)

```python
def region_growing(image, seed, threshold):
    mask=np.zeros_like(image,np.uint8)
    visited=np.zeros_like(image,bool)
    sx,sy=seed; seed_value=int(image[sy,sx])
    stack=[(sx,sy)]; h,w=image.shape
    while stack:
        x,y=stack.pop()
        if x<0 or x>=w or y<0 or y>=h or visited[y,x]: continue
        visited[y,x]=True
        if abs(int(image[y,x])-seed_value) <= threshold:
            mask[y,x]=255
            stack += [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    return mask

m10=region_growing(gray,(150,120),10)
m25=region_growing(gray,(150,120),25)
m50=region_growing(gray,(150,120),50)
```

Threshold too small -> under-segmentation; too large -> region leaks into dissimilar tissue/areas. Seed location strongly changes the result.

## Task 7 - marker-based watershed

```python
image=cv2.imread("coins.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
K=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(th,cv2.MORPH_OPEN,K,iterations=2)
sure_bg=cv2.dilate(opening,K,iterations=3)
dist=cv2.distanceTransform(opening,cv2.DIST_L2,5)
_,sure_fg=cv2.threshold(dist,0.5*dist.max(),255,0)
sure_fg=np.uint8(sure_fg)
unknown=cv2.subtract(sure_bg,sure_fg)
num,markers=cv2.connectedComponents(sure_fg)
markers=markers+1
markers[unknown==255]=0
ws=cv2.watershed(image.copy(),markers)
result=image.copy(); result[ws==-1]=[0,0,255]
labels=np.unique(ws); object_count=np.sum(labels>1)
print("objects:",object_count)
```

## Task 8 - watershed distance-threshold experiment

```python
def watershed_experiment(image,sure_bg,dist,factor):
    _,fg=cv2.threshold(dist,factor*dist.max(),255,0); fg=np.uint8(fg)
    unknown=cv2.subtract(sure_bg,fg)
    n,markers=cv2.connectedComponents(fg); fg_markers=n-1
    markers=markers+1; markers[unknown==255]=0
    ws=cv2.watershed(image.copy(),markers)
    regions=np.sum(np.unique(ws)>1)
    result=image.copy(); result[ws==-1]=[0,0,255]
    return fg,unknown,ws,result,fg_markers,regions

for f in [0.30,0.40,0.50,0.60]:
    *_, fg_n, regions = watershed_experiment(image,sure_bg,dist,f)
    print(f, fg_n, regions)
```

## Task 9 - K-means color segmentation

```python
def kmeans_segment(image_bgr,K):
    rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    Z=np.float32(rgb.reshape(-1,3))
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
    compactness,labels,centers=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    centers=np.uint8(centers)
    segmented=centers[labels.ravel()].reshape(rgb.shape)
    return segmented,centers,labels,compactness

seg2,*_=kmeans_segment(img,2)
seg4,*_=kmeans_segment(img,4)
seg6,*_=kmeans_segment(img,6)
```

Small K = stronger simplification; large K = more colors/detail. K-means clusters by feature distance (here RGB color only), not spatial continuity.

## Task 10 - compare global / Otsu / adaptive + foreground %

```python
global_T=150
_,global_mask=cv2.threshold(gray,global_T,255,cv2.THRESH_BINARY_INV)
otsu_T,otsu_mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
adaptive_mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,31,10)
def foreground_percentage(mask):
    return 100*np.count_nonzero(mask)/mask.size
for name,m in [("Global",global_mask),("Otsu",otsu_mask),("Adaptive",adaptive_mask)]:
    print(name, foreground_percentage(m))
```

## Active contour / snake (manual topic)

```python
# Requires scikit-image
from skimage.segmentation import active_contour
from skimage.filters import gaussian
s=np.linspace(0,2*np.pi,200)
r,c=100+60*np.sin(s), 120+60*np.cos(s)  # initial snake (row,col)
init=np.array([r,c]).T
snake=active_contour(gaussian(gray,3), init, alpha=0.015, beta=10, gamma=0.001)
```

# LAB 06 - Wavelets, Boundaries, Hough, SIFT

## Sobel / Canny / LoG (manual code)

```python
image=cv2.imread("image.jpeg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Sobel
gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
mag=np.sqrt(gx**2+gy**2)
mag8=np.uint8(np.clip(mag,0,255))
# Canny
blur=cv2.GaussianBlur(gray,(5,5),1.4)
canny=cv2.Canny(blur,50,150)
# LoG
lap=cv2.Laplacian(blur,cv2.CV_64F,ksize=3)
log=cv2.convertScaleAbs(lap)
```

## Wavelet essentials - DWT/CWT

```python
import pywt
# 1-D DWT decomposition/reconstruction
coeffs = pywt.wavedec(signal, wavelet="db4", level=4)  # [cA4,cD4,cD3,cD2,cD1]
reconstructed = pywt.waverec(coeffs, "db4")
# Single-level DWT
cA,cD = pywt.dwt(signal,"haar")
restored = pywt.idwt(cA,cD,"haar")
# CWT
scales=np.arange(1,64)
coef,freq = pywt.cwt(signal, scales, "morl")
```

## Lab 06 timing Task - wavelet anomaly detection in sensor data

```python
import pywt, numpy as np
# sensor = 1-D NumPy array
coeffs=pywt.wavedec(sensor,"db4",level=4)
sigma=np.median(np.abs(coeffs[-1]))/0.6745
thr=sigma*np.sqrt(2*np.log(len(sensor)))
denoised_coeffs=[coeffs[0]]+[pywt.threshold(c,thr,mode="soft") for c in coeffs[1:]]
denoised=pywt.waverec(denoised_coeffs,"db4")[:len(sensor)]
residual=sensor-denoised
z=(residual-residual.mean())/(residual.std()+1e-9)
anomaly_idx=np.where(np.abs(z)>3)[0]
print(anomaly_idx)
```

## Hough line transform

```python
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,150)
# Standard Hough: infinite lines (rho,theta)
lines=cv2.HoughLines(edges,1,np.pi/180,150)
if lines is not None:
    for rho,theta in lines[:,0]:
        a,b=np.cos(theta),np.sin(theta); x0,y0=a*rho,b*rho
        p1=(int(x0+1000*(-b)),int(y0+1000*a))
        p2=(int(x0-1000*(-b)),int(y0-1000*a))
        cv2.line(img,p1,p2,(0,0,255),2)

# Probabilistic Hough: finite segments (usually easier)
linesP=cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,
                       minLineLength=50,maxLineGap=10)
if linesP is not None:
    for x1,y1,x2,y2 in linesP[:,0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
```

## Hough circles

```python
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.medianBlur(gray,5)
circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,
                         param1=100,param2=30,minRadius=10,maxRadius=100)
count=0
if circles is not None:
    circles=np.uint16(np.around(circles[0]))
    count=len(circles)
    for x,y,r in circles:
        cv2.circle(img,(x,y),r,(0,255,0),2)
        cv2.circle(img,(x,y),2,(0,0,255),3)
print("count:",count)
```

## SIFT keypoints + descriptors

```python
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create()
kp,des=sift.detectAndCompute(gray,None)
vis=cv2.drawKeypoints(img,kp,None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(len(kp), des.shape if des is not None else None)
```

## SIFT matching + Lowe ratio + homography + object bounding box

```python
ref=cv2.imread("reference.jpg"); scene=cv2.imread("scene.jpg")
sift=cv2.SIFT_create()
kp1,d1=sift.detectAndCompute(cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY),None)
kp2,d2=sift.detectAndCompute(cv2.cvtColor(scene,cv2.COLOR_BGR2GRAY),None)
bf=cv2.BFMatcher(cv2.NORM_L2)
pairs=bf.knnMatch(d1,d2,k=2)
good=[m for m,n in pairs if m.distance < 0.75*n.distance]
if len(good)>=4:
    src=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H,inliers=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    h,w=ref.shape[:2]
    corners=np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    box=cv2.perspectiveTransform(corners,H)
    scene=cv2.polylines(scene,[np.int32(box)],True,(0,255,0),3)
match_vis=cv2.drawMatches(ref,kp1,scene,kp2,good[:50],None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

## FLANN alternative for SIFT

```python
index_params=dict(algorithm=1, trees=5)     # KD-tree for SIFT float descriptors
search_params=dict(checks=50)
flann=cv2.FlannBasedMatcher(index_params,search_params)
pairs=flann.knnMatch(d1,d2,k=2)
good=[m for m,n in pairs if m.distance < .75*n.distance]
```

## Task - computer screen detection using Hough lines

```python
img=cv2.imread("computer_lab.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,150)
lines=cv2.HoughLinesP(edge,1,np.pi/180,60,minLineLength=80,maxLineGap=15)
if lines is not None:
    for x1,y1,x2,y2 in lines[:,0]:
        angle=abs(np.degrees(np.arctan2(y2-y1,x2-x1)))
        if angle<10 or angle>80:              # roughly horizontal/vertical screen borders
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
```

## Task - asset tracking in lab using SIFT

Use the SIFT matching + homography block above with each known monitor/keyboard reference image. A robust identification = enough Lowe-ratio matches + enough RANSAC inliers.

```python
inlier_count = int(inliers.ravel().sum()) if inliers is not None else 0
if len(good)>=10 and inlier_count>=8:
    print("Asset recognized")
```

## Task - object recognition in VIDEO using SIFT

```python
ref=cv2.imread("reference.jpg"); refg=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create(); kp1,d1=sift.detectAndCompute(refg,None)
bf=cv2.BFMatcher(cv2.NORM_L2)
cap=cv2.VideoCapture("test.mp4")
while True:
    ok,frame=cap.read()
    if not ok: break
    kp2,d2=sift.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),None)
    if d2 is not None:
        pairs=bf.knnMatch(d1,d2,k=2)
        good=[m for m,n in pairs if m.distance<.75*n.distance]
        if len(good)>=8:
            src=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5)
            if H is not None:
                h,w=ref.shape[:2]
                box=cv2.perspectiveTransform(np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2),H)
                cv2.polylines(frame,[np.int32(box)],True,(0,255,0),3)
    # cv2.imshow("Recognition",frame); if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release(); cv2.destroyAllWindows()
```

## Task - panoramic image using SIFT + homography

```python
left=cv2.imread("left.jpg"); right=cv2.imread("right.jpg")
sift=cv2.SIFT_create(); bf=cv2.BFMatcher(cv2.NORM_L2)
kl,dl=sift.detectAndCompute(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),None)
kr,dr=sift.detectAndCompute(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),None)
pairs=bf.knnMatch(dr,dl,k=2)                 # right(query) -> left(train)
good=[m for m,n in pairs if m.distance<.75*n.distance]
src=np.float32([kr[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst=np.float32([kl[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H,_=cv2.findHomography(src,dst,cv2.RANSAC,5)
pano=cv2.warpPerspective(right,H,(left.shape[1]+right.shape[1],max(left.shape[0],right.shape[0])))
pano[:left.shape[0],:left.shape[1]]=left
```

## Task - lane detection using HoughLinesP

```python
frame=cv2.imread("road.jpg")
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,150)
h,w=edge.shape
mask=np.zeros_like(edge)
poly=np.array([[(0,h),(w//2-60,h//2+60),(w//2+60,h//2+60),(w,h)]],np.int32)
cv2.fillPoly(mask,poly,255)
roi=cv2.bitwise_and(edge,mask)
lines=cv2.HoughLinesP(roi,1,np.pi/180,40,minLineLength=50,maxLineGap=100)
if lines is not None:
    for x1,y1,x2,y2 in lines[:,0]:
        slope=(y2-y1)/(x2-x1+1e-9)
        if abs(slope)>0.4:
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),5)
```

## Task - coin detection/counting using Hough circles

Use the HoughCircles block above. Tune minDist, param2 and radius limits. param2 lower -> more circles/false positives; higher -> stricter.

## Task - smart security system: boundary + zone alarm

```python
cap=cv2.VideoCapture("security.mp4")
zone=(200,150,500,420)        # x1,y1,x2,y2
while True:
    ok,frame=cap.read()
    if not ok: break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,150)
    contours,_=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    zx1,zy1,zx2,zy2=zone; alarm=False
    for c in contours:
        if cv2.contourArea(c)<500: continue
        x,y,w,h=cv2.boundingRect(c)
        intersects = (x<zx2 and x+w>zx1 and y<zy2 and y+h>zy1)
        if intersects:
            alarm=True; cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.rectangle(frame,(zx1,zy1),(zx2,zy2),(255,0,0),2)
    if alarm: cv2.putText(frame,"ALARM",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
cap.release()
```

# EXAM DECISION MAP - question wording -> code

| Question says / symptom | Use |
| --- | --- |
| Uneven lighting / document text | Adaptive Gaussian threshold (local) |
| Need automatic single threshold | Otsu |
| Specific color object | BGR->HSV -> inRange -> bitwise_and |
| Touching objects need separation | Otsu -> opening -> distanceTransform -> markers -> watershed |
| Cluster image by colors | reshape pixels -> float32 -> cv2.kmeans -> reshape back |
| Grow a connected region from a seed | Region growing stack/queue + intensity tolerance |
| Straight lines / lanes / screen borders | Canny -> HoughLinesP |
| Coins / circular objects | Blur -> HoughCircles |
| Object invariant to scale/rotation | SIFT detectAndCompute -> matcher -> Lowe ratio |
| Locate SIFT object in scene | good matches -> findHomography(RANSAC) -> perspectiveTransform corners |
| Stitch images | SIFT matches -> homography -> warpPerspective -> overlay/blend |
| Rotate/scale/translate | getRotationMatrix2D + warpAffine |
| 3 point correspondences | getAffineTransform |
| 4 corner perspective correction | getPerspectiveTransform + warpPerspective |
| More than 4 noisy correspondences | findHomography(..., RANSAC) |
| Sensor anomalies / multiscale signal | DWT denoise -> residual -> threshold/z-score |

## Parameter cheat table

| Function | Key parameters / effect |
| --- | --- |
| GaussianBlur(img,(k,k),sigma) | k odd; larger k/sigma = more smoothing |
| threshold(gray,T,255,type) | global T; BINARY vs BINARY_INV |
| adaptiveThreshold(...,blockSize,C) | blockSize odd; local threshold = local statistic - C |
| Canny(gray,low,high) | low links weak edges; high selects strong edges; often high ~2-3x low |
| morphologyEx(...,OPEN,K) | erosion then dilation; removes small foreground noise |
| dilate | expands white foreground |
| distanceTransform | distance from foreground pixels to nearest zero/background |
| HoughLinesP | threshold votes; minLineLength; maxLineGap |
| HoughCircles | dp, minDist, param1=Canny high, param2=accumulator threshold, radii |
| SIFT Lowe ratio | 0.7-0.8 common; lower = stricter |
| findHomography RANSAC threshold | ~3-5 px common; rejects outlier matches |
| K-means K | more K = more clusters/detail |
| watershed distance factor | low may merge markers; high may lose markers |

## Morphology mini-reference

```python
K=np.ones((3,3),np.uint8)
eroded=cv2.erode(mask,K,iterations=1)                     # shrinks white
Dilated=cv2.dilate(mask,K,iterations=1)                   # grows white
opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,K)           # remove small white noise
closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,K)          # fill small black holes
gradient=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,K)      # boundary
```

## Contour mini-reference

```python
contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    area=cv2.contourArea(c)
    peri=cv2.arcLength(c,True)
    x,y,w,h=cv2.boundingRect(c)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    cv2.drawContours(img,[c],-1,(0,255,0),2)
```

## NumPy/OpenCV shape + dtype rules

- Color image shape = (height, width, channels); grayscale = (height, width). cv2.resize dsize is (width, height).

- uint8 range 0..255. Arithmetic on uint8 can saturate/wrap depending on operation; cast to float for transforms/gradients.

- Mask used by bitwise_and must be single-channel uint8 and same height/width as source.

- OpenCV draw colors are BGR, e.g. red=(0,0,255), green=(0,255,0), blue=(255,0,0).

- When plotting a BGR image: plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)). For gray: plt.imshow(gray,cmap="gray").

## Last-minute failure checklist

- img is None? Check path/case/working directory before debugging the algorithm.

- cv2 assertion failed in threshold/Canny? Convert to grayscale uint8 first.

- adaptiveThreshold error? blockSize must be odd and >1.

- bitwise mask error? mask must be uint8, one channel, same HxW.

- warp output clipped? Increase destination canvas and adjust translation, especially after rotation/shear.

- No SIFT matches? Check descriptors are not None; use grayscale; lower image blur; relax ratio slightly; need texture.

- Homography bad? Need >=4 well-spread correspondences; use RANSAC and inspect inliers.

- Hough detects too much? Increase vote/param2 threshold or minLineLength; preprocess/ROI.

- Video file empty? writer.release(); codec supported; frame size exactly equals writer size.

- Colors wrong in Matplotlib? BGR->RGB.

## Source/task completion checklist

LAB 01 tasks 1-9: included. LAB 02 tasks 1-3: included. LAB 03 tasks 1-10: included. LAB 04 manual: HOG/LBP/color-Hist/HED/HIG/texture, filtering/convolution/padding/stride, edge operators/Canny: included. LAB 05 manual + tasks 1-10: included. LAB 06 manual + all 8 task scenarios: included.
