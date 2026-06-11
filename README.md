# Garage Door Vision Monitor

A Python-based system that watches your garage door using a TP-Link Tapo security camera. It captures frames from the camera's live video feed, compares them against reference images of the closed door, and sends you a Telegram alert if the door is left open.

## Key Features

- **Automatic monitoring** -- continuously checks your garage door at a configurable interval (default: 30 minutes)
- **Multi-signal detection** -- uses three independent signals (image similarity, texture complexity, brightness) with 2-of-3 voting for robust classification
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

The monitor uses a **2-of-3 voting system** with three independent signals to decide if the door is open or closed:

1. **SSIM (image similarity)** -- compares the current frame against reference images of the closed door. A low score means the image looks different from the closed state. If below the threshold (default: 0.55), votes OPEN.
2. **Laplacian variance (texture complexity)** -- measures how much visual detail is in the door region. An open door reveals the outdoors, which has much more texture than a flat closed door. If above the threshold (default: 700), votes OPEN.
3. **ROI brightness** -- measures how bright the door region is. Behavior depends on camera mode:
   - **IR/night mode:** an open door appears darker (looking into darkness outside). If below threshold, votes OPEN.
   - **Daylight mode:** an open door appears brighter (outdoor light flooding in). If above threshold, votes OPEN.

If **2 or more signals** vote OPEN, the door is classified as open.

**Override:** If SSIM votes OPEN and the door region has very low variation (standard deviation), the door is forced to OPEN regardless of the other signals. This catches the case of a pitch-black open door at night where Laplacian and brightness can't distinguish it.

When the door is confirmed open, a Telegram alert with a photo is sent. When it closes again, an all-clear message is sent.

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
# Create a .env file in the project root (see Environment Variables below)
```

## Environment Variables

Create a `.env` file in the project root:

| Variable | Description | Required |
|----------|-------------|----------|
| `RTSP_URL` | Your Tapo camera's RTSP stream URL | Yes |
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather | Yes |
| `TELEGRAM_CHAT_IDS` | Comma-separated Telegram chat IDs to receive alerts | Yes |
| `REFERENCE_IMAGES` | Comma-separated paths to reference images | Yes |
| `SSIM_THRESHOLD` | Similarity threshold (0-1). Below this = door open | No (default: 0.55) |
| `INTERVAL_MINUTES` | How often to check, in minutes | No (default: 30) |
| `OPEN_ALERT_MINUTES` | Minutes door must stay open before alerting | No (default: 5) |
| `LAPLACIAN_THRESHOLD` | Texture complexity threshold. Above this = door open | No (default: 700) |
| `BRIGHTNESS_IR_OPEN_MAX` | IR mode brightness below this = door open | No (default: 85) |
| `BRIGHTNESS_DAY_OPEN_MIN` | Daylight brightness above this = door open | No (default: 165) |
| `CONSECUTIVE_OPEN_REQUIRED` | Number of consecutive OPEN readings before confirming | No (default: 2) |
| `ROI_STD_THRESHOLD` | Low ROI standard deviation override threshold | No (default: 55) |

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
SSIM: 0.8830 (CLOSED)
LapVar: 186 (CLOSED)
Bright: 138 (CLOSED, ir)
ROI Std: 41.0
Votes: 0/3 OPEN
Time: 2026-05-25 12:11:47
```

When the door is left open, the alert message looks like:

```
ALERT: Garage door has been OPEN for 5 minutes!
SSIM: 0.3784 (OPEN)
LapVar: 1075 (OPEN)
Bright: 180 (OPEN, daylight)
Votes: 3/3 OPEN
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
| `test_harness.py` | Validates all calibration images against current references and thresholds |
| `validate_detection.py` | Runs the 3-signal detection pipeline on all images and produces a CSV report |
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
