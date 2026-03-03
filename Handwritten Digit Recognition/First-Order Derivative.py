import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1️⃣ Read image (grayscale)
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Convert to float to avoid overflow
img = img.astype(np.float32)

# 2️⃣ Sobel kernels (dx and dy)
dx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

dy = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# 3️⃣ Apply dx → Gx
Gx = cv2.filter2D(img, -1, dx)

# 4️⃣ Apply dy → Gy
Gy = cv2.filter2D(img, -1, dy)

# 5️⃣ Gradient Magnitude
magnitude = np.sqrt(Gx**2 + Gy**2)

# 6️⃣ Gradient Direction (optional)
direction = np.arctan2(Gy, Gx)

# 7️⃣ Apply Threshold
threshold = 100
edges = np.zeros_like(magnitude)
edges[magnitude >= threshold] = 255

# 8️⃣ Display results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Gradient Magnitude")
plt.imshow(magnitude, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Binary Edge Image")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.show()