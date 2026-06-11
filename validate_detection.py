"""
Standalone validation script for multi-signal garage door detection.

Runs the 3-signal detection pipeline on all images in calibration_images/
and images/ directories. Produces a CSV report with all signal values.
"""
import cv2
import csv
import os
import sys
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim
from garage_monitor import (
    compute_ssim,
    crop_to_roi,
    detect_camera_mode,
    compute_roi_features,
    SSIM_THRESHOLD,
    LAPLACIAN_THRESHOLD,
    BRIGHTNESS_IR_OPEN_MAX,
    BRIGHTNESS_DAY_OPEN_MIN,
    ROI_STD_THRESHOLD,
)

load_dotenv()

REFERENCE_IMAGES = os.getenv("REFERENCE_IMAGES", "").split(",")
SCAN_DIRS = ["calibration_images", "images"]
OUTPUT_CSV = "validate_detection_results.csv"


def load_references():
    refs = {}
    for path in REFERENCE_IMAGES:
        path = path.strip()
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                refs[os.path.basename(path)] = img
    return refs


def main():
    refs = load_references()
    if not refs:
        print("Error: No reference images found. Check REFERENCE_IMAGES in .env.")
        sys.exit(1)

    ref_names = list(refs.keys())
    print(f"Loaded {len(refs)} reference images: {ref_names}")
    print(f"SSIM threshold: {SSIM_THRESHOLD}")
    print(f"Laplacian threshold: {LAPLACIAN_THRESHOLD}")
    print(f"Brightness thresholds: IR<{BRIGHTNESS_IR_OPEN_MAX}, Day>{BRIGHTNESS_DAY_OPEN_MIN}")
    print()

    # Collect all image files
    image_files = []
    for scan_dir in SCAN_DIRS:
        if not os.path.isdir(scan_dir):
            print(f"  Directory not found: {scan_dir} (skipping)")
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append((scan_dir, fname))

    if not image_files:
        print("No images found in any scan directory.")
        sys.exit(1)

    print(f"Found {len(image_files)} images across {SCAN_DIRS}")
    print(f"Writing results to {OUTPUT_CSV}...")
    print("-" * 90)

    fieldnames = [
        "directory",
        "image_file",
        "best_ssim",
        "best_ref",
        "ssim_vote",
        "laplacian_var",
        "lap_vote",
        "roi_brightness",
        "bright_vote",
        "camera_mode",
        "open_votes",
        "roi_std",
        "roi_std_override",
        "classification",
    ]

    open_count = 0
    closed_count = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (scan_dir, fname) in enumerate(image_files):
            image_path = os.path.join(scan_dir, fname)

            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            color = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if gray is None:
                print(f"  [{i+1}/{len(image_files)}] SKIP {scan_dir}/{fname} (unreadable)")
                continue

            # Signal 1: SSIM
            scores = {}
            for name, ref in refs.items():
                scores[name] = compute_ssim(gray, ref)
            best_ref = max(scores, key=scores.get)
            best_ssim = scores[best_ref]
            ssim_vote = "OPEN" if best_ssim < SSIM_THRESHOLD else "CLOSED"

            # Signal 2: Laplacian variance
            laplacian_var, mean_brightness, roi_std = compute_roi_features(gray)
            lap_vote = "OPEN" if laplacian_var > LAPLACIAN_THRESHOLD else "CLOSED"

            # Signal 3: Brightness
            camera_mode = detect_camera_mode(color) if color is not None else "daylight"
            if camera_mode == "ir":
                bright_vote = "OPEN" if mean_brightness < BRIGHTNESS_IR_OPEN_MAX else "CLOSED"
            else:
                bright_vote = "OPEN" if mean_brightness > BRIGHTNESS_DAY_OPEN_MIN else "CLOSED"

            # 2-of-3 voting
            open_votes = sum(1 for v in [ssim_vote, lap_vote, bright_vote] if v == "OPEN")
            classification = "OPEN" if open_votes >= 2 else "CLOSED"

            # Override: SSIM OPEN + low ROI std dev → night open door
            roi_std_override = False
            if ssim_vote == "OPEN" and roi_std < ROI_STD_THRESHOLD:
                classification = "OPEN"
                roi_std_override = True

            if classification == "OPEN":
                open_count += 1
            else:
                closed_count += 1

            writer.writerow({
                "directory": scan_dir,
                "image_file": fname,
                "best_ssim": f"{best_ssim:.4f}",
                "best_ref": best_ref,
                "ssim_vote": ssim_vote,
                "laplacian_var": f"{laplacian_var:.1f}",
                "lap_vote": lap_vote,
                "roi_brightness": f"{mean_brightness:.1f}",
                "bright_vote": bright_vote,
                "camera_mode": camera_mode,
                "open_votes": f"{open_votes}/3",
                "roi_std": f"{roi_std:.1f}",
                "roi_std_override": "YES" if roi_std_override else "",
                "classification": classification,
            })

            override_tag = " [STD_OVERRIDE]" if roi_std_override else ""
            print(
                f"  [{i+1}/{len(image_files)}] {scan_dir}/{fname}  "
                f"SSIM:{best_ssim:.4f}({ssim_vote}) "
                f"Lap:{laplacian_var:.0f}({lap_vote}) "
                f"Brt:{mean_brightness:.0f}({bright_vote},{camera_mode}) "
                f"Std:{roi_std:.1f} "
                f"-> {classification}{override_tag}"
            )

    print()
    print("=" * 60)
    print(f"  RESULTS: {len(image_files)} images processed")
    print(f"  Classified OPEN:   {open_count}")
    print(f"  Classified CLOSED: {closed_count}")
    print(f"  Output: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
