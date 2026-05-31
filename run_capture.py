"""Single-run capture script for Windows Task Scheduler.

Captures one frame, computes SSIM against all references, logs to CSV, and exits.
Task Scheduler handles the repeat interval.
"""
import cv2
import sys
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim

# =============================================================================
# Configuration (loaded from .env)
# =============================================================================
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

RTSP_URL = os.getenv("RTSP_URL")
SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", "0.55"))
REFERENCE_IMAGES = os.getenv("REFERENCE_IMAGES", "").split(",")

WARMUP_FRAMES = 10
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_images")
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_log.csv")


def main():
    if not RTSP_URL:
        print("Error: RTSP_URL not set in .env file.")
        sys.exit(1)

    # Load reference images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    refs = {}
    for path in REFERENCE_IMAGES:
        path = path.strip()
        full_path = os.path.join(script_dir, path) if not os.path.isabs(path) else path
        if path and os.path.exists(full_path):
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                refs[full_path] = img

    if not refs:
        print("Error: No reference images found.")
        sys.exit(1)

    # Capture frame
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        sys.exit(1)

    for _ in range(WARMUP_FRAMES):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not capture frame.")
        sys.exit(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute SSIM against each reference
    scores = {}
    for name, ref in refs.items():
        f = gray
        if f.shape != ref.shape:
            f = cv2.resize(f, (ref.shape[1], ref.shape[0]))
        score, _ = ssim(ref, f, full=True)
        scores[name] = score

    best_ref = max(scores, key=scores.get)
    best_ssim = scores[best_ref]
    door_status = "OPEN" if best_ssim < SSIM_THRESHOLD else "CLOSED"

    # Save image
    os.makedirs(IMAGES_DIR, exist_ok=True)
    now = datetime.now()
    image_filename = f"capture_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    cv2.imwrite(os.path.join(IMAGES_DIR, image_filename), frame)

    # Append to CSV
    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "best_ssim", "best_ref",
                "ssim_threshold", "door_status", "image_file",
            ])
        writer.writerow([
            now.strftime("%Y-%m-%d %H:%M:%S"),
            f"{best_ssim:.4f}",
            os.path.basename(best_ref),
            SSIM_THRESHOLD,
            door_status,
            image_filename,
        ])

    print(
        f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"SSIM: {best_ssim:.4f} ({os.path.basename(best_ref)})  |  "
        f"Door: {door_status}"
    )


if __name__ == "__main__":
    main()
