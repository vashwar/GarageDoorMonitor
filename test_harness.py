import cv2
import csv
import os
import sys
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim
from garage_monitor import (
    crop_to_roi,
    detect_camera_mode,
    compute_roi_features,
    LAPLACIAN_THRESHOLD,
    BRIGHTNESS_IR_OPEN_MAX,
    BRIGHTNESS_DAY_OPEN_MIN,
    ROI_STD_THRESHOLD,
)

# =============================================================================
# Configuration (loaded from .env)
# =============================================================================
load_dotenv()

SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", "0.55"))
REFERENCE_IMAGES = os.getenv("REFERENCE_IMAGES", "").split(",")

CALIBRATION_LOG = "calibration_log.csv"
IMAGES_DIR = "calibration_images"
OUTPUT_CSV = "test_harness_results.csv"


def main():
    # Load reference images
    refs = {}
    for path in REFERENCE_IMAGES:
        path = path.strip()
        if not path:
            continue
        if not os.path.exists(path):
            print(f"Warning: reference image not found: {path}")
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            refs[os.path.basename(path)] = img

    if not refs:
        print("Error: No reference images found. Check REFERENCE_IMAGES in .env file.")
        sys.exit(1)

    ref_names = list(refs.keys())
    print(f"Loaded {len(refs)} reference images: {ref_names}")
    print(f"SSIM threshold: {SSIM_THRESHOLD}")
    print(f"Laplacian threshold: {LAPLACIAN_THRESHOLD}")

    # Read calibration log
    if not os.path.exists(CALIBRATION_LOG):
        print(f"Error: {CALIBRATION_LOG} not found.")
        sys.exit(1)

    with open(CALIBRATION_LOG, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} entries in {CALIBRATION_LOG}")
    print(f"Writing results to {OUTPUT_CSV}...")

    # Process each image and write output
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = [
            "timestamp",
            "image_file",
            "original_status",
            "original_ssim",
        ] + [f"ssim_{name}" for name in ref_names] + [
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
            "new_status",
            "match",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        match_count = 0
        mismatch_count = 0
        skipped = 0

        for i, row in enumerate(rows):
            image_file = row["image_file"]
            image_path = os.path.join(IMAGES_DIR, image_file)

            if not os.path.exists(image_path):
                print(f"  Skipping {image_file} (file not found)")
                skipped += 1
                continue

            frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            color_frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"  Skipping {image_file} (could not read)")
                skipped += 1
                continue

            # Signal 1: SSIM against each reference
            scores = {}
            for name, ref in refs.items():
                f_resized = frame
                if f_resized.shape != ref.shape:
                    f_resized = cv2.resize(f_resized, (ref.shape[1], ref.shape[0]))
                f_blur = cv2.GaussianBlur(f_resized, (7, 7), 0)
                ref_blur = cv2.GaussianBlur(ref, (7, 7), 0)
                score, _ = ssim(ref_blur, f_blur, full=True)
                scores[name] = score

            best_ref = max(scores, key=scores.get)
            best_ssim = scores[best_ref]
            ssim_vote = "OPEN" if best_ssim < SSIM_THRESHOLD else "CLOSED"

            # Signal 2: Laplacian variance
            laplacian_var, mean_brightness, roi_std = compute_roi_features(frame)
            lap_vote = "OPEN" if laplacian_var > LAPLACIAN_THRESHOLD else "CLOSED"

            # Signal 3: ROI brightness
            camera_mode = detect_camera_mode(color_frame) if color_frame is not None else "daylight"
            if camera_mode == "ir":
                bright_vote = "OPEN" if mean_brightness < BRIGHTNESS_IR_OPEN_MAX else "CLOSED"
            else:
                bright_vote = "OPEN" if mean_brightness > BRIGHTNESS_DAY_OPEN_MIN else "CLOSED"

            # 2-of-3 voting
            open_votes = sum(1 for v in [ssim_vote, lap_vote, bright_vote] if v == "OPEN")
            new_status = "OPEN" if open_votes >= 2 else "CLOSED"

            # Override: SSIM OPEN + low ROI std dev → night open door
            roi_std_override = False
            if ssim_vote == "OPEN" and roi_std < ROI_STD_THRESHOLD:
                new_status = "OPEN"
                roi_std_override = True

            original_status = row["door_status"]
            matched = "YES" if new_status == original_status else "NO"

            if matched == "YES":
                match_count += 1
            else:
                mismatch_count += 1

            out_row = {
                "timestamp": row["timestamp"],
                "image_file": image_file,
                "original_status": original_status,
                "original_ssim": row["ssim"],
            }
            for name in ref_names:
                out_row[f"ssim_{name}"] = f"{scores[name]:.4f}"
            out_row["best_ssim"] = f"{best_ssim:.4f}"
            out_row["best_ref"] = best_ref
            out_row["ssim_vote"] = ssim_vote
            out_row["laplacian_var"] = f"{laplacian_var:.1f}"
            out_row["lap_vote"] = lap_vote
            out_row["roi_brightness"] = f"{mean_brightness:.1f}"
            out_row["bright_vote"] = bright_vote
            out_row["camera_mode"] = camera_mode
            out_row["open_votes"] = f"{open_votes}/3"
            out_row["roi_std"] = f"{roi_std:.1f}"
            out_row["roi_std_override"] = "YES" if roi_std_override else ""
            out_row["new_status"] = new_status
            out_row["match"] = matched

            writer.writerow(out_row)

            override_tag = " [STD_OVERRIDE]" if roi_std_override else ""
            print(
                f"  [{i+1}/{len(rows)}] {image_file}  |  "
                f"SSIM:{best_ssim:.4f}({ssim_vote}) "
                f"Lap:{laplacian_var:.0f}({lap_vote}) "
                f"Brt:{mean_brightness:.0f}({bright_vote},{camera_mode}) "
                f"Std:{roi_std:.1f}  |  "
                f"{original_status} -> {new_status}  {matched}{override_tag}"
            )

    # Summary
    total = match_count + mismatch_count
    print()
    print("=" * 60)
    print(f"  RESULTS: {total} images processed, {skipped} skipped")
    print(f"  Matches:    {match_count}/{total}")
    print(f"  Mismatches: {mismatch_count}/{total}")
    if total > 0:
        print(f"  Accuracy:   {match_count/total*100:.1f}%")
    print(f"  Output:     {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
