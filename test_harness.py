import cv2
import csv
import os
import sys
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim

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
            if frame is None:
                print(f"  Skipping {image_file} (could not read)")
                skipped += 1
                continue

            # Compute SSIM against each reference
            scores = {}
            for name, ref in refs.items():
                f_resized = frame
                if f_resized.shape != ref.shape:
                    f_resized = cv2.resize(f_resized, (ref.shape[1], ref.shape[0]))
                # Gaussian blur to match garage_monitor.py pipeline
                f_blur = cv2.GaussianBlur(f_resized, (7, 7), 0)
                ref_blur = cv2.GaussianBlur(ref, (7, 7), 0)
                score, _ = ssim(ref_blur, f_blur, full=True)
                scores[name] = score

            best_ref = max(scores, key=scores.get)
            best_ssim = scores[best_ref]
            new_status = "OPEN" if best_ssim < SSIM_THRESHOLD else "CLOSED"
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
            out_row["new_status"] = new_status
            out_row["match"] = matched

            writer.writerow(out_row)

            print(
                f"  [{i+1}/{len(rows)}] {image_file}  |  "
                f"Best: {best_ssim:.4f} ({best_ref})  |  "
                f"{original_status} -> {new_status}  {matched}"
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
