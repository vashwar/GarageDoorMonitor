# Garage Door Vision Monitor

A Python-based system that watches your garage door using a TP-Link Tapo security camera. It captures frames from the camera's live video feed, compares them against reference images of the closed door, and sends you a Telegram alert if the door is left open.

## Key Features

- **Automatic monitoring** -- continuously checks your garage door at a configurable interval (default: 30 minutes)
- **Multi-reference matching** -- compares against multiple reference images (two cars, one car, no cars, day/night) to handle different garage configurations
- **Telegram alerts** -- sends a photo and message to your phone when the door is open, and an all-clear when it closes
- **On-demand status** -- send `/status` to the Telegram bot anytime to get a live photo and door state
- **Day/night support** -- handles the camera's automatic IR night mode and daytime color mode switching
- **Calibration tools** -- scripts to capture reference images, run test validation, and tune detection thresholds

## Tech Stack

- **Language:** Python 3.x
- **Vision:** OpenCV (`opencv-python`) for frame capture, `scikit-image` for SSIM (Structural Similarity Index) calculation
- **Messaging:** `python-telegram-bot` (v20+ async)
- **Camera:** TP-Link Tapo via RTSP stream
- **Config:** `python-dotenv` for environment variables

## How It Works

1. Connects to the Tapo camera's RTSP video stream
2. Captures a single frame and converts it to grayscale
3. Compares the frame against reference images of the closed door using SSIM (a score from 0 to 1 measuring how similar two images are)
4. If the best SSIM score falls below the threshold (default: 0.55), the door is classified as **OPEN**
5. If the door stays open, a Telegram alert with a photo is sent to your phone
6. When the door closes again, an all-clear message is sent

## Prerequisites

- Python 3.10+ ([download](https://www.python.org/downloads/))
- A TP-Link Tapo camera on your local network with RTSP enabled
- A Telegram bot token (create one via [BotFather](https://t.me/botfather))
- Your Telegram chat ID(s)

## Installation

```bash
# Step 1: Clone or download the project
cd C:\VashwarTests\GarageCamera

# Step 2: Install dependencies
pip install opencv-python scikit-image python-telegram-bot python-dotenv

# Step 3: Set up environment variables
# Create a .env file in the project root with the following:
```

## Environment Variables

Create a `.env` file in the project root:

| Variable | Description | Required |
|----------|-------------|----------|
| `RTSP_URL` | Your Tapo camera's RTSP stream URL | Yes |
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather | Yes |
| `TELEGRAM_CHAT_IDS` | Comma-separated Telegram chat IDs to receive alerts | Yes |
| `SSIM_THRESHOLD` | Similarity threshold (0-1). Below this = door open | No (default: 0.55) |
| `INTERVAL_MINUTES` | How often to check, in minutes | No (default: 30) |
| `OPEN_ALERT_MINUTES` | Minutes door must stay open before alerting | No (default: 0) |
| `REFERENCE_IMAGES` | Comma-separated paths to reference images | Yes |

## How to Run

```bash
# Start the monitor (runs continuously)
python garage_monitor.py

# Or use the batch files on Windows:
start_monitor.bat    # runs in background
stop_monitor.bat     # stops the background process
```

Once running, the bot listens for Telegram commands:

- `/status` or text `status` -- captures a live photo and reports door state

### Sample Telegram Message

When you send `/status`, the bot replies with a photo and caption like this:

![Sample status response](docs/sample_status.jpg)

```
Garage is CLOSED
SSIM: 0.8830
Time: 2026-05-25 12:11:47
```

When the door is left open, the alert message looks like:

```
ALERT: Garage door has been OPEN for 5 minutes!
SSIM: 0.3784
Time: 2026-05-25 17:30:58
```

And when it closes again:

```
Garage door is now CLOSED.
Was open for 6 minutes.
```

## Project Files

| File | Purpose |
|------|---------|
| `garage_monitor.py` | Main monitor -- runs continuously, sends Telegram alerts |
| `capture_reference.py` | Interactive tool to capture new reference images from the camera |
| `calibrate.py` | One-time calibration -- captures a frame and shows SSIM scores against all references |
| `scheduled_calibrate.py` | Continuous scheduled capture with CSV logging (for tuning thresholds) |
| `run_capture.py` | Single-frame capture for use with Windows Task Scheduler |
| `test_harness.py` | Validates all calibration images against current references and threshold |
| `refimage/` | Reference images of the closed garage door |
| `images/` | Captured frames from the monitor (auto-cleaned, keeps last 30) |
| `calibration_images/` | Captured frames from calibration runs |
| `calibration_log.csv` | Log of all calibration captures with SSIM scores |

## Adding Reference Images

If the monitor produces false positives (classifying a closed door as open), you likely need a new reference image for that lighting condition:

1. Save the misclassified image to `refimage/` with a descriptive name
2. Add the path to `REFERENCE_IMAGES` in `.env` (comma-separated)
3. Restart the monitor

Common scenarios that need their own reference: daytime vs nighttime (IR mode), different car configurations, seasonal lighting changes.
