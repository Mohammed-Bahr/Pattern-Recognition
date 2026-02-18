"""
Image Grayscale Histogram Analysis
===================================
Loads an image, converts to grayscale, computes and plots its histogram,
and performs basic statistical analysis using a Pandas DataFrame.

Dependencies: opencv-python, numpy, matplotlib, pandas
Install: pip install opencv-python numpy matplotlib pandas
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────
# 1. Load Image
# ──────────────────────────────────────────────
def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk. Raises FileNotFoundError if path is invalid."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")

    print(f"[✓] Loaded image: {path.name}  |  Shape: {image.shape}")
    return image


# ──────────────────────────────────────────────
# 2. Convert to Grayscale
# ──────────────────────────────────────────────
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image (OpenCV default) to grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"[✓] Converted to grayscale  |  Shape: {gray.shape}")
    return gray


# ──────────────────────────────────────────────
# 3. Compute Histogram
# ──────────────────────────────────────────────
def compute_histogram(gray: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the pixel intensity histogram of a grayscale image.

    Returns:
        pixel_values : array of intensity levels [0 … bins-1]
        frequencies  : array of pixel counts per intensity level
    """
    # calcHist returns shape (256, 1); flatten to 1-D
    hist = cv2.calcHist([gray], [0], None, [bins], [0, bins]).flatten()
    pixel_values = np.arange(bins)
    print(f"[✓] Histogram computed  |  Bins: {bins}  |  Total pixels: {int(hist.sum()):,}")
    return pixel_values, hist


# ──────────────────────────────────────────────
# 4. Plot Histogram
# ──────────────────────────────────────────────
def plot_histogram(
    gray: np.ndarray,
    pixel_values: np.ndarray,
    frequencies: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    Plot the grayscale image alongside its pixel intensity histogram.

    Args:
        gray         : Grayscale image array.
        pixel_values : Intensity levels (0–255).
        frequencies  : Pixel counts per level.
        save_path    : Optional path to save the figure as a PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grayscale Image & Pixel Intensity Histogram", fontsize=14, fontweight="bold")

    # — Left panel: grayscale image —
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale Image")
    axes[0].axis("off")

    # — Right panel: histogram —
    axes[1].fill_between(pixel_values, frequencies, alpha=0.6, color="steelblue")
    axes[1].plot(pixel_values, frequencies, color="navy", linewidth=0.8)
    axes[1].set_title("Pixel Intensity Histogram")
    axes[1].set_xlabel("Pixel Intensity (0 = Black, 255 = White)")
    axes[1].set_ylabel("Frequency (# of Pixels)")
    axes[1].set_xlim([0, 255])
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Plot saved to: {save_path}")

    plt.show()


# ──────────────────────────────────────────────
# 5. Build Histogram DataFrame
# ──────────────────────────────────────────────
def build_dataframe(pixel_values: np.ndarray, frequencies: np.ndarray) -> pd.DataFrame:
    """
    Store histogram data in a Pandas DataFrame.

    Columns:
        pixel_value  : Intensity level (0–255).
        frequency    : Raw pixel count.
        normalized   : Frequency as a proportion of total pixels [0, 1].
        cumulative   : Cumulative distribution (CDF), useful for equalization.
    """
    total_pixels = frequencies.sum()

    df = pd.DataFrame({
        "pixel_value": pixel_values.astype(int),
        "frequency":   frequencies.astype(int),
        "normalized":  frequencies / total_pixels,          # probability mass
        "cumulative":  np.cumsum(frequencies) / total_pixels,  # CDF
    })

    print(f"[✓] DataFrame created  |  Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────
# 6. Statistical Analysis on the DataFrame
# ──────────────────────────────────────────────
def analyze_histogram(df: pd.DataFrame) -> dict:
    """
    Compute basic statistical measures from the histogram DataFrame.

    Returns a dict of key metrics and prints a formatted summary.
    """
    # Weighted statistics (each intensity weighted by its frequency)
    weights = df["frequency"]
    values  = df["pixel_value"]

    mean_intensity   = np.average(values, weights=weights)
    variance         = np.average((values - mean_intensity) ** 2, weights=weights)
    std_intensity    = np.sqrt(variance)

    # Median: first intensity where cumulative frequency >= 50 %
    median_intensity = df.loc[df["cumulative"] >= 0.50, "pixel_value"].iloc[0]

    # Mode: intensity with the highest frequency
    mode_intensity   = df.loc[df["frequency"].idxmax(), "pixel_value"]

    # Filter: pixels in the mid-tone range [85, 170]
    mid_tones        = df.query("85 <= pixel_value <= 170")
    mid_tone_pct     = mid_tones["normalized"].sum() * 100

    # Dark (<85) / mid (85–170) / bright (>170) pixel share
    dark_pct   = df.query("pixel_value < 85")["normalized"].sum()   * 100
    bright_pct = df.query("pixel_value > 170")["normalized"].sum()  * 100

    stats = {
        "mean_intensity":   round(mean_intensity,  2),
        "std_intensity":    round(std_intensity,   2),
        "median_intensity": int(median_intensity),
        "mode_intensity":   int(mode_intensity),
        "dark_pct":         round(dark_pct,        2),
        "mid_tone_pct":     round(mid_tone_pct,    2),
        "bright_pct":       round(bright_pct,      2),
    }

    print("\n" + "=" * 44)
    print("         HISTOGRAM STATISTICS SUMMARY")
    print("=" * 44)
    print(f"  Mean intensity      : {stats['mean_intensity']}")
    print(f"  Std deviation       : {stats['std_intensity']}")
    print(f"  Median intensity    : {stats['median_intensity']}")
    print(f"  Mode (most common)  : {stats['mode_intensity']}")
    print(f"  Dark pixels  (<85)  : {stats['dark_pct']} %")
    print(f"  Mid-tone    (85-170): {stats['mid_tone_pct']} %")
    print(f"  Bright pixels (>170): {stats['bright_pct']} %")
    print("=" * 44 + "\n")

    return stats


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────
def run_pipeline(image_path: str, save_plot: str | None = "histogram_plot.png") -> pd.DataFrame:
    """
    Full pipeline: load → grayscale → histogram → plot → DataFrame → analysis.

    Args:
        image_path : Path to the input image file.
        save_plot  : Path to save the histogram plot (None to skip saving).

    Returns:
        df : Histogram DataFrame for downstream use.
    """
    image         = load_image(image_path)
    gray          = to_grayscale(image)
    px, freq      = compute_histogram(gray)
    plot_histogram(gray, px, freq, save_path=save_plot)
    df            = build_dataframe(px, freq)
    stats         = analyze_histogram(df)

    # Preview the DataFrame
    print("DataFrame head (first 5 rows):")
    print(df.head().to_string(index=False))
    print("\nDataFrame tail (last 5 rows):")
    print(df.tail().to_string(index=False))

    return df


# ──────────────────────────────────────────────
# Entry Point — replace the path with your image
# ──────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = "sample.jpg"   # ← change this to your image path

    # ── Quick demo using a generated test image if no file is present ──
    if not Path(IMAGE_PATH).exists():
        print(f"[!] '{IMAGE_PATH}' not found. Generating a synthetic test image instead.\n")
        # Create a 400×400 gradient image for demonstration
        synthetic = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
        cv2.imwrite("sample.jpg", synthetic)

    df = run_pipeline(IMAGE_PATH, save_plot="histogram_plot.png")

    # ── Example: extend for ML — export histogram features as a feature vector ──
    feature_vector = df["normalized"].values   # shape: (256,)
    print(f"\n[ML-ready] Normalised histogram feature vector shape: {feature_vector.shape}")