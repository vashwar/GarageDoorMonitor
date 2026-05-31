import cv2
import sys
import os
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim

# =============================================================================
# Configuration (loaded from .env)
# =============================================================================
load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", "0.55"))
REFERENCE_IMAGES = os.getenv("REFERENCE_IMAGES", "").split(",")

WARMUP_FRAMES = 10

# Output paths
IMAGES_DIR = "calibration_images"
CSV_FILE = "calibration_log.csv"

# Schedule
INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", "15"))


def load_references(paths):
    refs = {}
    for path in paths:
        path = path.strip()
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                refs[path] = img
    return refs


def capture_and_log(references):
    """Capture a frame, compute best SSIM across references, save and log."""
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    for _ in range(WARMUP_FRAMES):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not capture frame.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute SSIM against each reference
    scores = {}
    for name, ref in references.items():
        f = gray
        if f.shape != ref.shape:
            f = cv2.resize(f, (ref.shape[1], ref.shape[0]))
        score, _ = ssim(ref, f, full=True)
        scores[name] = score

    best_ref = max(scores, key=scores.get)
    best_ssim = scores[best_ref]
    door_status = "OPEN" if best_ssim < SSIM_THRESHOLD else "CLOSED"

    # Save image with timestamp
    os.makedirs(IMAGES_DIR, exist_ok=True)
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    image_filename = f"capture_{timestamp_str}.jpg"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    cv2.imwrite(image_path, frame)

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
        f"Door: {door_status}  |  Image: {image_filename}"
    )


def main():
    if not RTSP_URL:
        print("Error: RTSP_URL not set in .env file.")
        sys.exit(1)

    references = load_references(REFERENCE_IMAGES)
    if not references:
        print("Error: No reference images found. Check REFERENCE_IMAGES in .env file.")
        sys.exit(1)

    print(f"Scheduled calibration - capturing every {INTERVAL_MINUTES} minutes")
    print(f"SSIM threshold: {SSIM_THRESHOLD}")
    print(f"Reference images: {len(references)}")
    for name in references:
        print(f"  - {name}")
    print(f"Images saved to: {IMAGES_DIR}/")
    print(f"Log saved to: {CSV_FILE}")
    print("-" * 70)

    while True:
        try:
            capture_and_log(references)
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}")
        time.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
