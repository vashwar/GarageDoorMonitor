import cv2
import sys
import os
import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# =============================================================================
# Configuration (loaded from .env)
# =============================================================================
load_dotenv()

# Bound how long a single open/read may block, in milliseconds.
CAPTURE_OPEN_TIMEOUT_MS = int(os.getenv("CAPTURE_OPEN_TIMEOUT_MS", "8000"))
CAPTURE_READ_TIMEOUT_MS = int(os.getenv("CAPTURE_READ_TIMEOUT_MS", "8000"))
CAPTURE_RETRIES = int(os.getenv("CAPTURE_RETRIES", "3"))
CAPTURE_RETRY_BACKOFF = int(os.getenv("CAPTURE_RETRY_BACKOFF", "5"))

# Force RTSP over TCP before any VideoCapture is created. OpenCV reads this env
# var when the ffmpeg backend opens the stream. TCP is far more stable than the
# ffmpeg UDP default, which stalls and triggers read timeouts on Tapo. The
# open/read timeouts themselves are bounded via constructor params in
# _open_capture() (they must be set before the connect starts).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

RTSP_URL = os.getenv("RTSP_URL")
SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", "0.55"))
REFERENCE_IMAGES = os.getenv("REFERENCE_IMAGES", "").split(",")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [
    int(cid.strip()) for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid.strip()
]
OPEN_ALERT_MINUTES = int(os.getenv("OPEN_ALERT_MINUTES", "5"))
REPEAT_ALERT_MINUTES = int(os.getenv("REPEAT_ALERT_MINUTES", "15"))

# How often to check (seconds) — reads INTERVAL_MINUTES from .env
MONITOR_INTERVAL = int(os.getenv("INTERVAL_MINUTES", "15")) * 60

# Number of warmup frames to skip for camera auto-exposure
WARMUP_FRAMES = 10

# Image storage
IMAGES_DIR = "images"
MAX_IMAGES = 30

# ROI (Region of Interest) — door-panel gap between cars on 2304x1296 frame
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 900, 100, 1400, 700
FRAME_W, FRAME_H = 2304, 1296

# Multi-signal detection thresholds
LAPLACIAN_THRESHOLD = float(os.getenv("LAPLACIAN_THRESHOLD", "700"))
BRIGHTNESS_IR_OPEN_MAX = float(os.getenv("BRIGHTNESS_IR_OPEN_MAX", "85"))
BRIGHTNESS_DAY_OPEN_MIN = float(os.getenv("BRIGHTNESS_DAY_OPEN_MIN", "165"))
IR_SATURATION_THRESHOLD = 20.0
CONSECUTIVE_OPEN_REQUIRED = int(os.getenv("CONSECUTIVE_OPEN_REQUIRED", "2"))
ROI_STD_THRESHOLD = float(os.getenv("ROI_STD_THRESHOLD", "55"))


# =============================================================================
# Core Functions
# =============================================================================

def cleanup_images():
    """Delete oldest images if folder exceeds MAX_IMAGES."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    files = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(files) > MAX_IMAGES:
        files.sort(key=os.path.getmtime)
        for f in files[: len(files) - MAX_IMAGES]:
            os.remove(f)


def save_image(color_frame, prefix="capture"):
    """Save a frame to the images/ folder with a timestamp name. Returns the path."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    cleanup_images()
    filename = f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    path = os.path.join(IMAGES_DIR, filename)
    cv2.imwrite(path, color_frame)
    return path


def load_references(paths):
    """Load all available reference images as grayscale. Skip missing files."""
    refs = {}
    for path in paths:
        path = path.strip()
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                refs[path] = img
    return refs


def _open_capture():
    """Open the RTSP stream over TCP with bounded open/read timeouts.

    The timeouts must be passed as constructor params: OpenCV's ffmpeg
    interrupt callback reads them before the connect starts, so a dead camera
    fails in ~CAPTURE_OPEN_TIMEOUT_MS instead of the 30s default. Setting them
    via cap.set() after construction is too late — the connect already ran.
    """
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAPTURE_OPEN_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAPTURE_READ_TIMEOUT_MS,
    ]
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG, params)
    # Keep the buffer shallow so we read the latest frame, not a stale backlog.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_frame_from_camera():
    """Connect to camera, grab a single frame, return (color, grayscale) or (None, None).

    Retries a few times with short backoff on transient RTSP failures so a
    single dropped connection doesn't blind the monitor for a full interval.
    """
    for attempt in range(1, CAPTURE_RETRIES + 1):
        cap = _open_capture()
        try:
            if not cap.isOpened():
                raise RuntimeError("stream did not open")

            # Warm up with cheap grab() calls so auto-exposure settles without
            # decoding every frame (and without stacking read timeouts).
            for _ in range(WARMUP_FRAMES):
                if not cap.grab():
                    raise RuntimeError("grab failed during warmup")

            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("read returned no frame")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame, gray
        except Exception as e:
            print(
                f"[{time.strftime('%H:%M:%S')}] [WARN] capture attempt "
                f"{attempt}/{CAPTURE_RETRIES} failed: {e}"
            )
            if attempt < CAPTURE_RETRIES:
                time.sleep(CAPTURE_RETRY_BACKOFF)
        finally:
            cap.release()

    return None, None


def compute_ssim(frame, reference):
    """Compute SSIM between current frame and a reference image."""
    if frame.shape != reference.shape:
        frame = cv2.resize(frame, (reference.shape[1], reference.shape[0]))
    # Gaussian blur to reduce noise and smooth minor IR/daylight artifacts
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    reference = cv2.GaussianBlur(reference, (7, 7), 0)
    score, _ = ssim(reference, frame, full=True)
    return score


def crop_to_roi(gray_frame):
    """Crop the grayscale frame to the door-panel ROI, resizing to standard size first."""
    h, w = gray_frame.shape[:2]
    if w != FRAME_W or h != FRAME_H:
        scale_x = w / FRAME_W
        scale_y = h / FRAME_H
        x1 = int(ROI_X1 * scale_x)
        y1 = int(ROI_Y1 * scale_y)
        x2 = int(ROI_X2 * scale_x)
        y2 = int(ROI_Y2 * scale_y)
    else:
        x1, y1, x2, y2 = ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
    return gray_frame[y1:y2, x1:x2]


def detect_camera_mode(color_frame):
    """Detect IR/night vs daylight mode based on mean color saturation."""
    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
    mean_saturation = hsv[:, :, 1].mean()
    return "ir" if mean_saturation < IR_SATURATION_THRESHOLD else "daylight"


def compute_roi_features(gray_frame):
    """Compute Laplacian variance, mean brightness, and std dev on the door ROI."""
    roi = crop_to_roi(gray_frame)
    laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
    mean_brightness = float(roi.mean())
    roi_std = float(roi.std())
    return laplacian_var, mean_brightness, roi_std


def check_door(gray_frame, references, color_frame=None):
    """
    Multi-signal door detection using 2-of-3 voting.

    Signal 1: SSIM against reference images (existing)
    Signal 2: ROI Laplacian variance (texture complexity)
    Signal 3: ROI brightness (IR reflectance / outdoor light)

    Returns (is_open, best_ssim, best_ref_name, signals_detail).
    signals_detail is a dict with all signal values for logging.
    """
    # Signal 1: SSIM (existing logic)
    scores = {}
    for name, ref in references.items():
        scores[name] = compute_ssim(gray_frame, ref)

    best_ref = max(scores, key=scores.get)
    best_ssim = scores[best_ref]
    ssim_vote = "OPEN" if best_ssim < SSIM_THRESHOLD else "CLOSED"

    # Signal 2: Laplacian variance on ROI
    laplacian_var, mean_brightness, roi_std = compute_roi_features(gray_frame)
    lap_vote = "OPEN" if laplacian_var > LAPLACIAN_THRESHOLD else "CLOSED"

    # Signal 3: ROI brightness (mode-dependent)
    camera_mode = detect_camera_mode(color_frame) if color_frame is not None else "daylight"
    if camera_mode == "ir":
        bright_vote = "OPEN" if mean_brightness < BRIGHTNESS_IR_OPEN_MAX else "CLOSED"
    else:
        bright_vote = "OPEN" if mean_brightness > BRIGHTNESS_DAY_OPEN_MIN else "CLOSED"

    # 2-of-3 voting
    open_votes = sum(1 for v in [ssim_vote, lap_vote, bright_vote] if v == "OPEN")
    is_open = open_votes >= 2

    # Override: SSIM OPEN + low ROI std dev → uniform dark void = door open at night
    roi_std_override = False
    if ssim_vote == "OPEN" and roi_std < ROI_STD_THRESHOLD:
        is_open = True
        roi_std_override = True

    signals_detail = {
        "ssim_vote": ssim_vote,
        "laplacian_var": laplacian_var,
        "lap_vote": lap_vote,
        "mean_brightness": mean_brightness,
        "bright_vote": bright_vote,
        "camera_mode": camera_mode,
        "open_votes": open_votes,
        "roi_std": roi_std,
        "roi_std_override": roi_std_override,
    }

    return is_open, best_ssim, os.path.basename(best_ref), signals_detail


# =============================================================================
# Telegram Bot Handlers
# =============================================================================

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command — capture a photo, analyze, and reply."""
    if update.effective_chat.id not in TELEGRAM_CHAT_IDS:
        return

    await update.message.reply_text("Capturing garage image...")

    references = context.bot_data.get("references", {})
    if not references:
        await update.message.reply_text("Error: No reference images loaded.")
        return

    color_frame, gray_frame = await asyncio.to_thread(capture_frame_from_camera)
    if gray_frame is None:
        await update.message.reply_text("Error: Could not capture frame from camera.")
        return

    is_open, best_ssim, best_ref, signals = await asyncio.to_thread(
        check_door, gray_frame, references, color_frame
    )
    status = "OPEN" if is_open else "CLOSED"

    image_path = await asyncio.to_thread(save_image, color_frame, "status")

    override_note = " [STD_OVERRIDE]" if signals.get('roi_std_override') else ""
    caption = (
        f"Garage is {status}{override_note}\n"
        f"SSIM: {best_ssim:.4f} ({signals['ssim_vote']})\n"
        f"LapVar: {signals['laplacian_var']:.0f} ({signals['lap_vote']})\n"
        f"Bright: {signals['mean_brightness']:.0f} ({signals['bright_vote']}, {signals['camera_mode']})\n"
        f"ROI Std: {signals['roi_std']:.1f}\n"
        f"Votes: {signals['open_votes']}/3 OPEN\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    with open(image_path, "rb") as photo:
        await update.message.reply_photo(photo=photo, caption=caption)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages — respond to 'status'."""
    if update.effective_chat.id not in TELEGRAM_CHAT_IDS:
        return

    text = update.message.text.strip().lower()
    if text == "status":
        await cmd_status(update, context)
    else:
        await update.message.reply_text(
            "Commands:\n"
            "  /status or 'status' - Check garage door\n"
        )


async def send_alert(bot, message, photo_path=None):
    """Send an alert message (and optional photo) to all whitelisted chat IDs."""
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            if photo_path and os.path.exists(photo_path):
                with open(photo_path, "rb") as photo:
                    await bot.send_photo(chat_id=chat_id, photo=photo, caption=message)
            else:
                await bot.send_message(chat_id=chat_id, text=message)
        except Exception as e:
            print(f"[WARN] Failed to send Telegram alert to {chat_id}: {e}")


# =============================================================================
# Monitoring Loop
# =============================================================================

async def monitor_loop(app: Application):
    """Continuous monitoring loop that sends alerts when door is open too long."""
    references = app.bot_data["references"]
    bot = app.bot

    door_open_since = None
    last_alert_time = None
    consecutive_open_count = 0

    # Wait for bot to be ready
    await asyncio.sleep(2)
    print(f"Monitoring started. Checking every {MONITOR_INTERVAL // 60} minutes.")
    print(f"SSIM threshold: {SSIM_THRESHOLD}")
    print(f"Laplacian threshold: {LAPLACIAN_THRESHOLD}")
    print(f"Consecutive OPEN readings required: {CONSECUTIVE_OPEN_REQUIRED}")
    print(f"Alert after {OPEN_ALERT_MINUTES} minutes open.")
    print(f"Repeat alert every {REPEAT_ALERT_MINUTES} minutes if still open.")
    print("-" * 70)

    while True:
        try:
            color_frame, gray_frame = await asyncio.to_thread(capture_frame_from_camera)
            if gray_frame is None:
                print(f"[{time.strftime('%H:%M:%S')}] [WARN] Failed to capture frame")
                await asyncio.sleep(MONITOR_INTERVAL)
                continue

            is_open, best_ssim, best_ref, signals = await asyncio.to_thread(
                check_door, gray_frame, references, color_frame
            )

            timestamp = time.strftime("%H:%M:%S")

            # Temporal filtering: require consecutive OPEN readings
            if is_open:
                consecutive_open_count += 1
            else:
                consecutive_open_count = 0

            # Only treat as truly open if enough consecutive readings
            confirmed_open = consecutive_open_count >= CONSECUTIVE_OPEN_REQUIRED

            status = "OPEN" if confirmed_open else "CLOSED"
            override_tag = " [STD_OVERRIDE]" if signals.get('roi_std_override') else ""
            print(
                f"[{timestamp}] Door: {status}  |  "
                f"SSIM: {best_ssim:.4f}({signals['ssim_vote']})  "
                f"Lap: {signals['laplacian_var']:.0f}({signals['lap_vote']})  "
                f"Brt: {signals['mean_brightness']:.0f}({signals['bright_vote']},{signals['camera_mode']})  "
                f"Std: {signals['roi_std']:.1f}  "
                f"Votes: {signals['open_votes']}/3  "
                f"Consec: {consecutive_open_count}{override_tag}"
            )

            if confirmed_open:
                if door_open_since is None:
                    door_open_since = datetime.now()
                    last_alert_time = None

                elapsed_total = (datetime.now() - door_open_since).total_seconds() / 60

                if elapsed_total >= OPEN_ALERT_MINUTES:
                    send_alert_now = False
                    if last_alert_time is None:
                        send_alert_now = True
                    else:
                        elapsed_since_last = (datetime.now() - last_alert_time).total_seconds() / 60
                        if elapsed_since_last >= REPEAT_ALERT_MINUTES:
                            send_alert_now = True

                    if send_alert_now:
                        alert_path = await asyncio.to_thread(
                            save_image, color_frame, "alert"
                        )
                        alert_msg = (
                            f"ALERT: Garage door has been OPEN for "
                            f"{int(elapsed_total)} minutes!\n"
                            f"SSIM: {best_ssim:.4f} ({signals['ssim_vote']})\n"
                            f"LapVar: {signals['laplacian_var']:.0f} ({signals['lap_vote']})\n"
                            f"Bright: {signals['mean_brightness']:.0f} ({signals['bright_vote']}, {signals['camera_mode']})\n"
                            f"Votes: {signals['open_votes']}/3 OPEN\n"
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        await send_alert(bot, alert_msg, alert_path)
                        last_alert_time = datetime.now()
                        print(f"[{timestamp}] *** ALERT SENT ***")
            else:
                if door_open_since is not None and last_alert_time is not None:
                    elapsed = (datetime.now() - door_open_since).total_seconds() / 60
                    clear_msg = (
                        f"Garage door is now CLOSED.\n"
                        f"Was open for {int(elapsed)} minutes."
                    )
                    await send_alert(bot, clear_msg)
                    print(f"[{timestamp}] *** ALL-CLEAR SENT ***")

                door_open_since = None
                last_alert_time = None

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Monitor error: {e}")

        await asyncio.sleep(MONITOR_INTERVAL)


# =============================================================================
# Main
# =============================================================================

def main():
    if not RTSP_URL:
        print("Error: RTSP_URL not set in .env file.")
        sys.exit(1)
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set in .env file.")
        sys.exit(1)
    if not TELEGRAM_CHAT_IDS:
        print("Error: TELEGRAM_CHAT_IDS not set in .env file.")
        sys.exit(1)

    # Load reference images
    references = load_references(REFERENCE_IMAGES)
    if not references:
        print("Error: No reference images found. Check REFERENCE_IMAGES in .env file.")
        sys.exit(1)

    print(f"Loaded {len(references)} reference image(s):")
    for name in references:
        print(f"  - {name}")
    print(f"Telegram alerts to: {TELEGRAM_CHAT_IDS}")

    # Build Telegram bot application
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.bot_data["references"] = references

    # Register handlers
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run bot polling + monitor loop together
    async def run():
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        try:
            await monitor_loop(app)
        finally:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()

    print("Starting Telegram bot + monitor...")
    asyncio.run(run())


if __name__ == "__main__":
    main()
