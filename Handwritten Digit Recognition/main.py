import numpy as np
from tensorflow.keras.datasets import mnist

# --------------------------------------------------
# 1) Load MNIST dataset
# --------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# هنشتغل على الداتا كلها (train + test)
images = np.concatenate((x_train, x_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

# --------------------------------------------------
# 2) Convert images to binary
# --------------------------------------------------
def to_binary(images, threshold=128):
    return (images > threshold).astype(int)

binary_images = to_binary(images)

# --------------------------------------------------
# 3) Divide image into blocks and calculate centroids
# --------------------------------------------------
def calculate_block_centroids(image, num_blocks_h, num_blocks_v):
    """
    image: 2D numpy array (binary image 28x28)
    num_blocks_h: number of horizontal blocks
    num_blocks_v: number of vertical blocks
    """
    h, w = image.shape
    block_h = h // num_blocks_v
    block_w = w // num_blocks_h

    centroids = []

    for i in range(num_blocks_v):
        for j in range(num_blocks_h):
            block = image[
                i * block_h:(i + 1) * block_h,
                j * block_w:(j + 1) * block_w
            ]

            ones_positions = np.argwhere(block == 1)

            if len(ones_positions) == 0:
                # لو البلوك فاضي
                centroid_x, centroid_y = 0.0, 0.0
            else:
                centroid_y = np.mean(ones_positions[:, 0])
                centroid_x = np.mean(ones_positions[:, 1])

            centroids.extend([centroid_x, centroid_y])

    return np.array(centroids)

# --------------------------------------------------
# 4) Create feature vectors for all images
# --------------------------------------------------
def extract_features(images, num_blocks_h, num_blocks_v):
    feature_vectors = []
    for img in images:
        fv = calculate_block_centroids(img, num_blocks_h, num_blocks_v)
        feature_vectors.append(fv)
    return np.array(feature_vectors)

# --------------------------------------------------
# User Input for number of blocks
# --------------------------------------------------
num_blocks_h = int(input("Enter number of horizontal blocks: "))
num_blocks_v = int(input("Enter number of vertical blocks: "))

# --------------------------------------------------
# Feature Extraction
# --------------------------------------------------
features = extract_features(binary_images, num_blocks_h, num_blocks_v)

# --------------------------------------------------
# Expected Output
# --------------------------------------------------
print("Feature vectors shape:", features.shape)
print("Labels shape:", labels.shape)

# مثال على أول صورة
print("\nExample Feature Vector (first image):")
print(features[0])

print("\nCorresponding Label:")
print(labels[0])