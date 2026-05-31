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

RTSP_URL = os.getenv("RTSP_URL")
SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", "0.55"))
REFERENCE_IMAGES = os.getenv("REFERENCE_IMAGES", "").split(",")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [
    int(cid.strip()) for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid.strip()
]
OPEN_ALERT_MINUTES = int(os.getenv("OPEN_ALERT_MINUTES", "5"))

# How often to check (seconds) — reads INTERVAL_MINUTES from .env
MONITOR_INTERVAL = int(os.getenv("INTERVAL_MINUTES", "15")) * 60

# Number of warmup frames to skip for camera auto-exposure
WARMUP_FRAMES = 10

# Image storage
IMAGES_DIR = "images"
MAX_IMAGES = 30


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


def capture_frame_from_camera():
    """Connect to camera, grab a single frame, return (color, grayscale) or (None, None)."""
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        return None, None

    for _ in range(WARMUP_FRAMES):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray


def compute_ssim(frame, reference):
    """Compute SSIM between current frame and a reference image."""
    if frame.shape != reference.shape:
        frame = cv2.resize(frame, (reference.shape[1], reference.shape[0]))
    # Gaussian blur to reduce noise and smooth minor IR/daylight artifacts
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    reference = cv2.GaussianBlur(reference, (7, 7), 0)
    score, _ = ssim(reference, frame, full=True)
    return score


def check_door(gray_frame, references):
    """
    Compare frame against all reference images.
    Returns (is_open, best_ssim, best_ref_name).
    """
    scores = {}
    for name, ref in references.items():
        scores[name] = compute_ssim(gray_frame, ref)

    best_ref = max(scores, key=scores.get)
    best_ssim = scores[best_ref]
    is_open = best_ssim < SSIM_THRESHOLD

    return is_open, best_ssim, os.path.basename(best_ref)


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

    is_open, best_ssim, best_ref = await asyncio.to_thread(check_door, gray_frame, references)
    status = "OPEN" if is_open else "CLOSED"

    image_path = await asyncio.to_thread(save_image, color_frame, "status")

    caption = (
        f"Garage is {status}\n"
        f"SSIM: {best_ssim:.4f}\n"
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
    alert_sent = False

    # Wait for bot to be ready
    await asyncio.sleep(2)
    print(f"Monitoring started. Checking every {MONITOR_INTERVAL // 60} minutes.")
    print(f"SSIM threshold: {SSIM_THRESHOLD}")
    print(f"Alert after {OPEN_ALERT_MINUTES} minutes open.")
    print("-" * 70)

    while True:
        try:
            color_frame, gray_frame = await asyncio.to_thread(capture_frame_from_camera)
            if gray_frame is None:
                print(f"[{time.strftime('%H:%M:%S')}] [WARN] Failed to capture frame")
                await asyncio.sleep(MONITOR_INTERVAL)
                continue

            is_open, best_ssim, best_ref = await asyncio.to_thread(
                check_door, gray_frame, references
            )

            status = "OPEN" if is_open else "CLOSED"
            timestamp = time.strftime("%H:%M:%S")
            print(
                f"[{timestamp}] Door: {status}  |  "
                f"Best SSIM: {best_ssim:.4f} ({best_ref})"
            )

            if is_open:
                if door_open_since is None:
                    door_open_since = datetime.now()
                    alert_sent = False

                elapsed = (datetime.now() - door_open_since).total_seconds() / 60

                if elapsed >= OPEN_ALERT_MINUTES and not alert_sent:
                    # Send alert with photo
                    alert_path = await asyncio.to_thread(
                        save_image, color_frame, "alert"
                    )
                    alert_msg = (
                        f"ALERT: Garage door has been OPEN for "
                        f"{int(elapsed)} minutes!\n"
                        f"SSIM: {best_ssim:.4f}\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    await send_alert(bot, alert_msg, alert_path)
                    alert_sent = True
                    print(f"[{timestamp}] *** ALERT SENT ***")
            else:
                if door_open_since is not None and alert_sent:
                    # Door closed after alert — send all-clear
                    elapsed = (datetime.now() - door_open_since).total_seconds() / 60
                    clear_msg = (
                        f"Garage door is now CLOSED.\n"
                        f"Was open for {int(elapsed)} minutes."
                    )
                    await send_alert(bot, clear_msg)
                    print(f"[{timestamp}] *** ALL-CLEAR SENT ***")

                door_open_since = None
                alert_sent = False

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
