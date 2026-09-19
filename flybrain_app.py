# ============================================================
# FLYBRAIN GEOMETRY DASH AI TRAINER
# ============================================================
#
# Python 3.13+
#
# Install:
#   py -m pip install numpy pillow opencv-python mss pyautogui
#   py -m pip install keyboard pynput
#   py -m pip install flybrain
#
# Optional GPU:
#   flybrain can use CUDA/CuPy if your installation supports it.
#
# Controls:
#   F8 = EMERGENCY STOP
#
# Modes:
#   OBSERVE  = watches Geometry Dash
#   LEARN    = records your Space/mouse decisions
#   AI TEST  = AI predicts but DOES NOT press
#   AI PLAY  = AI actually controls the game
#
# ============================================================

import os
import sys
import json
import time
import math
import hashlib
import secrets
import threading
import traceback
import webbrowser
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mss
except Exception:
    mss = None

try:
    import pyautogui
    pyautogui.PAUSE = 0.0
except Exception:
    pyautogui = None

try:
    import keyboard
except Exception:
    keyboard = None

try:
    from pynput import mouse
except Exception:
    mouse = None

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    from flybrain import FlyBrain, Eyes, Blob
except Exception:
    FlyBrain = None
    Eyes = None
    Blob = None


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

ACCOUNT_FILE = APP_DIR / "flybrain_accounts.json"
DATA_FILE = APP_DIR / "training_data.json"
MODEL_FILE = APP_DIR / "jump_model.npz"
DEBUG_DIR = APP_DIR / "debug_frames"

DEBUG_DIR.mkdir(exist_ok=True)

# Geometry Dash capture area
GAME_LEFT = 0
GAME_TOP = 0
GAME_WIDTH = 1280
GAME_HEIGHT = 1024

# Default calibration
DEFAULT_PLAYER_X = 300
DEFAULT_PLAYER_Y = 720
DEFAULT_GROUND_Y = 760
DEFAULT_SCAN_DISTANCE = 550

# Training targets
MIN_TRAINING_SAMPLES = 5000
MIN_JUMP_SAMPLES = 250
MIN_NO_JUMP_SAMPLES = 1000

# AI
AI_THRESHOLD = 0.66
AI_COOLDOWN = 0.38

# Worker rates
VISION_HZ = 45.0
BRAIN_HZ = 30.0
LEARN_RECORD_HZ = 30.0

# Discord
DISCORD_CLIENT_ID = os.environ.get(
    "FLYBRAIN_DISCORD_CLIENT_ID",
    ""
)

DISCORD_REDIRECT = "http://127.0.0.1:8765/callback"


# ============================================================
# COLORS
# ============================================================

BG = "#101216"
PANEL = "#181b21"
PANEL2 = "#20242b"
TEXT = "#eeeeee"
SUBTEXT = "#9ca3af"
ACCENT = "#6ea8fe"
GREEN = "#62d38b"
RED = "#ff6868"
YELLOW = "#ffd166"
ORANGE = "#ff9f43"
CYAN = "#4dd0e1"


# ============================================================
# FEATURES
# ============================================================

FEATURE_NAMES = [
    "player_y_norm",
    "velocity_y_norm",
    "grounded",
    "obstacle_distance_norm",
    "obstacle_height_norm",
    "obstacle_width_norm",
    "obstacle_confidence",
    "player_confidence",
    "brain_activity_norm",
    "brain_fired_norm",
]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class VisionState:
    player_found: bool = False

    player_x: float = 0.0
    player_y: float = 0.0
    player_w: float = 0.0
    player_h: float = 0.0

    obstacle_found: bool = False

    obstacle_x: float = 0.0
    obstacle_y: float = 0.0
    obstacle_w: float = 0.0
    obstacle_h: float = 0.0

    obstacle_distance: float = 9999.0

    grounded: bool = False
    velocity_y: float = 0.0

    confidence: float = 0.0
    obstacle_confidence: float = 0.0

    frame_ms: float = 0.0
    fps: float = 0.0

    jump_probability: float = 0.0
    reason: str = "Waiting"


@dataclass
class BrainState:
    available: bool = False
    running: bool = False
    activity: float = 0.0
    fired: int = 0
    total_neurons: int = 0
    error: str = ""

    brain: object = None
    eyes: object = None


@dataclass
class AppState:
    running: bool = True

    emergency: bool = False

    mode: str = "OBSERVE"
    complexity: str = "Simple"
    control: str = "Keyboard"

    training_active: bool = False
    training_session_id: str = ""
    training_session_samples: int = 0

    ai_enabled: bool = False

    player_x_cal: int = DEFAULT_PLAYER_X
    player_y_cal: int = DEFAULT_PLAYER_Y
    ground_y_cal: int = DEFAULT_GROUND_Y
    scan_distance: int = DEFAULT_SCAN_DISTANCE

    vision: VisionState = field(default_factory=VisionState)
    brain: BrainState = field(default_factory=BrainState)

    frame: object = None
    debug_frame: object = None

    samples: list = field(default_factory=list)
    sessions: list = field(default_factory=list)

    model_weights: object = None
    model_mean: object = None
    model_std: object = None
    model_bias: float = 0.0

    model_ready: bool = False
    train_accuracy: float = 0.0
    validation_accuracy: float = 0.0

    last_jump_time: float = 0.0

    mouse_down: bool = False
    space_down: bool = False

    previous_y: float = 0.0
    previous_y_time: float = 0.0

    frame_times: deque = field(
        default_factory=lambda: deque(maxlen=60)
    )

    logs: deque = field(
        default_factory=lambda: deque(maxlen=250)
    )

    lock: threading.RLock = field(
        default_factory=threading.RLock
    )


STATE = AppState()


# ============================================================
# LOGGING
# ============================================================

def log(message):
    timestamp = time.strftime("%H:%M:%S")

    with STATE.lock:
        STATE.logs.append(
            f"[{timestamp}] {message}"
        )

    print(f"[{timestamp}] {message}")


# ============================================================
# ACCOUNT SYSTEM
# ============================================================

def load_accounts():
    if not ACCOUNT_FILE.exists():
        return {}

    try:
        with open(ACCOUNT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_accounts(accounts):
    tmp = ACCOUNT_FILE.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=2)

    tmp.replace(ACCOUNT_FILE)


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        200_000
    ).hex()


def create_account(username, password):
    accounts = load_accounts()

    username = username.strip()

    if not username:
        return False, "Username is empty."

    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    if username in accounts:
        return False, "Username already exists."

    salt = secrets.token_hex(16)

    accounts[username] = {
        "salt": salt,
        "hash": password_hash(password, salt),
        "role": "user",
    }

    save_accounts(accounts)

    return True, "Account created."


def verify_account(username, password):
    accounts = load_accounts()

    if username not in accounts:
        return False, None

    account = accounts[username]

    expected = password_hash(
        password,
        account["salt"]
    )

    if secrets.compare_digest(
        expected,
        account["hash"]
    ):
        return True, account

    return False, None


# ============================================================
# CAPTURE
# ============================================================

SCREEN = None


def setup_capture():
    global SCREEN

    if mss is None:
        log("mss is not installed.")
        return False

    try:
        SCREEN = mss.mss()

        log(
            f"Capture region: "
            f"{GAME_WIDTH}x{GAME_HEIGHT} "
            f"at ({GAME_LEFT},{GAME_TOP})"
        )

        return True

    except Exception as e:
        log(f"Screen capture failed: {e}")
        return False


def capture_screen():
    if SCREEN is None:
        return None

    try:
        raw = np.array(
            SCREEN.grab({
                "left": GAME_LEFT,
                "top": GAME_TOP,
                "width": GAME_WIDTH,
                "height": GAME_HEIGHT,
            })
        )

        # BGRA -> BGR
        frame = np.ascontiguousarray(
            raw[:, :, :3]
        )

        return frame

    except Exception as e:
        log(f"Capture error: {e}")
        return None


# ============================================================
# GEOMETRY DASH WINDOW SAFETY
# ============================================================

def foreground_is_geometry_dash():
    if sys.platform != "win32":
        return True

    try:
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()

        if not hwnd:
            return False

        length = ctypes.windll.user32.GetWindowTextLengthW(
            hwnd
        )

        if length <= 0:
            return False

        buffer = ctypes.create_unicode_buffer(
            length + 1
        )

        ctypes.windll.user32.GetWindowTextW(
            hwnd,
            buffer,
            length + 1
        )

        title = buffer.value.lower()

        return (
            "geometry dash" in title
            or "geometrydash" in title
        )

    except Exception:
        return True


# ============================================================
# PLAYER DETECTION
# ============================================================

def detect_player(frame):
    if cv2 is None or frame is None:
        return None, 0.0

    h, w = frame.shape[:2]

    expected_x = STATE.player_x_cal
    expected_y = STATE.player_y_cal

    # Small region around expected cube position.
    x1 = max(0, expected_x - 140)
    x2 = min(w, expected_x + 140)

    y1 = max(0, expected_y - 150)
    y2 = min(h, expected_y + 150)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # Hitboxes and bright cube edges tend to create
    # compact high-contrast regions.
    edges = cv2.Canny(
        gray,
        70,
        180
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = 0.0

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(contour)

        if cw < 6 or ch < 6:
            continue

        if cw > 80 or ch > 80:
            continue

        area = cw * ch

        if area < 50:
            continue

        absolute_x = x + x1
        absolute_y = y + y1

        center_x = absolute_x + cw / 2
        center_y = absolute_y + ch / 2

        distance = math.sqrt(
            (center_x - expected_x) ** 2
            +
            (center_y - expected_y) ** 2
        )

        proximity = max(
            0.0,
            1.0 - distance / 220.0
        )

        ratio = min(cw, ch) / max(cw, ch)

        square_score = ratio

        size_score = min(
            1.0,
            area / 500.0
        )

        score = (
            proximity * 0.55
            +
            square_score * 0.30
            +
            size_score * 0.15
        )

        if score > best_score:
            best_score = score

            best = (
                absolute_x,
                absolute_y,
                cw,
                ch
            )

    return best, float(best_score)


# ============================================================
# OBSTACLE DETECTION
# ============================================================

def detect_obstacle(frame, player):
    if cv2 is None or frame is None or player is None:
        return None, 0.0

    px, py, pw, ph = player

    h, w = frame.shape[:2]

    start_x = int(
        max(0, px + pw + 10)
    )

    end_x = int(
        min(
            w,
            px + STATE.scan_distance
        )
    )

    if end_x <= start_x:
        return None, 0.0

    # Focus on the lower-middle portion of the level.
    top_y = max(
        0,
        int(py - 130)
    )

    bottom_y = min(
        h,
        int(STATE.ground_y_cal + 40)
    )

    roi = frame[
        top_y:bottom_y,
        start_x:end_x
    ]

    if roi.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        70,
        180
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = 0.0

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw < 8 or ch < 8:
            continue

        if cw > 500 or ch > 500:
            continue

        area = cw * ch

        if area < 80:
            continue

        ax = x + start_x
        ay = y + top_y

        distance = ax - (px + pw)

        if distance < 0:
            continue

        if distance > STATE.scan_distance:
            continue

        # Ignore very large horizontal floor pieces.
        if cw > 250 and ch < 25:
            continue

        distance_score = max(
            0.0,
            1.0 - distance /
            max(1, STATE.scan_distance)
        )

        height_center = ay + ch / 2

        vertical_difference = abs(
            height_center - STATE.ground_y_cal
        )

        vertical_score = max(
            0.0,
            1.0 - vertical_difference / 250
        )

        size_score = min(
            1.0,
            area / 5000
        )

        score = (
            distance_score * 0.50
            +
            vertical_score * 0.30
            +
            size_score * 0.20
        )

        if score > best_score:
            best_score = score

            best = (
                ax,
                ay,
                cw,
                ch
            )

    return best, float(best_score)


# ============================================================
# VISION UPDATE
# ============================================================

def update_vision(frame):
    start = time.perf_counter()

    player, player_conf = detect_player(
        frame
    )

    obstacle = None
    obstacle_conf = 0.0

    if player is not None:
        obstacle, obstacle_conf = detect_obstacle(
            frame,
            player
        )

    now = time.time()

    with STATE.lock:

        old_y = STATE.vision.player_y
        old_time = STATE.previous_y_time

        if player is not None:

            px, py, pw, ph = player

            STATE.vision.player_found = True

            STATE.vision.player_x = px + pw / 2
            STATE.vision.player_y = py + ph / 2

            STATE.vision.player_w = pw
            STATE.vision.player_h = ph

            STATE.vision.confidence = player_conf

            if old_time > 0:
                dt = now - old_time

                if dt > 0:
                    STATE.vision.velocity_y = (
                        STATE.vision.player_y - old_y
                    ) / dt

            STATE.previous_y = (
                STATE.vision.player_y
            )

            STATE.previous_y_time = now

            # Ground estimation
            feet = py + ph

            STATE.vision.grounded = (
                abs(
                    feet -
                    STATE.ground_y_cal
                ) < 35
            )

        else:

            STATE.vision.player_found = False
            STATE.vision.confidence = 0.0

        if obstacle is not None:

            ox, oy, ow, oh = obstacle

            STATE.vision.obstacle_found = True

            STATE.vision.obstacle_x = ox
            STATE.vision.obstacle_y = oy

            STATE.vision.obstacle_w = ow
            STATE.vision.obstacle_h = oh

            STATE.vision.obstacle_distance = (
                ox -
                STATE.vision.player_x
            )

            STATE.vision.obstacle_confidence = (
                obstacle_conf
            )

        else:

            STATE.vision.obstacle_found = False
            STATE.vision.obstacle_confidence = 0.0
            STATE.vision.obstacle_distance = 9999

        elapsed = (
            time.perf_counter() - start
        )

        STATE.vision.frame_ms = (
            elapsed * 1000
        )

        STATE.frame_times.append(
            now
        )

        if len(STATE.frame_times) >= 2:

            duration = (
                STATE.frame_times[-1]
                -
                STATE.frame_times[0]
            )

            if duration > 0:

                STATE.vision.fps = (
                    len(STATE.frame_times) - 1
                ) / duration


# ============================================================
# DEBUG FRAME
# ============================================================

def draw_debug(frame):
    if frame is None:
        return None

    out = frame.copy()

    with STATE.lock:

        v = STATE.vision

        # Player
        if v.player_found:

            x = int(
                v.player_x -
                v.player_w / 2
            )

            y = int(
                v.player_y -
                v.player_h / 2
            )

            cv2.rectangle(
                out,
                (x, y),
                (
                    x + int(v.player_w),
                    y + int(v.player_h)
                ),
                (0, 255, 0),
                2
            )

            cv2.circle(
                out,
                (
                    int(v.player_x),
                    int(v.player_y)
                ),
                4,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                out,
                f"PLAYER {v.confidence:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # Obstacle
        if v.obstacle_found:

            ox = int(v.obstacle_x)
            oy = int(v.obstacle_y)

            ow = int(v.obstacle_w)
            oh = int(v.obstacle_h)

            cv2.rectangle(
                out,
                (ox, oy),
                (ox + ow, oy + oh),
                (0, 120, 255),
                2
            )

            cv2.putText(
                out,
                f"OBSTACLE {v.obstacle_confidence:.2f}",
                (ox, max(20, oy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 120, 255),
                2
            )

        # Calibration
        cv2.line(
            out,
            (
                STATE.player_x_cal,
                0
            ),
            (
                STATE.player_x_cal,
                GAME_HEIGHT
            ),
            (255, 255, 0),
            1
        )

        cv2.line(
            out,
            (
                0,
                STATE.ground_y_cal
            ),
            (
                GAME_WIDTH,
                STATE.ground_y_cal
            ),
            (255, 255, 0),
            1
        )

        # Scan line
        scan_x = (
            STATE.player_x_cal
            +
            STATE.scan_distance
        )

        cv2.line(
            out,
            (
                scan_x,
                0
            ),
            (
                scan_x,
                GAME_HEIGHT
            ),
            (255, 0, 255),
            1
        )

        # Info
        info = [
            f"Mode: {STATE.mode}",
            f"AI: {'ON' if STATE.ai_enabled else 'OFF'}",
            f"Player: {v.player_found}",
            f"Grounded: {v.grounded}",
            f"Velocity Y: {v.velocity_y:.1f}",
            f"Obstacle: {v.obstacle_found}",
            f"Distance: {v.obstacle_distance:.1f}",
            f"Jump P: {v.jump_probability:.3f}",
            f"FPS: {v.fps:.1f}",
        ]

        y = 25

        for line in info:

            cv2.putText(
                out,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            y += 25

    return out


# ============================================================
# FLYBRAIN
# ============================================================

def initialize_brain():
    if FlyBrain is None:

        with STATE.lock:
            STATE.brain.available = False
            STATE.brain.error = (
                "FlyBrain package unavailable."
            )

        log("FlyBrain import failed.")
        return

    try:

        data = Path(
            os.environ.get(
                "FLYBRAIN_DATA",
                str(Path.home() / "fly-data")
            )
        )

        log(
            f"Loading FlyBrain from {data}"
        )

        brain = FlyBrain(
            data=data,
            device="cuda"
        )

        eyes = Eyes(
            brain.azimuth
        )

        with STATE.lock:

            STATE.brain.brain = brain
            STATE.brain.eyes = eyes

            STATE.brain.available = True
            STATE.brain.running = True

            STATE.brain.total_neurons = (
                brain.n
            )

        log(
            f"FlyBrain loaded: "
            f"{brain.n:,} neurons"
        )

    except Exception as e:

        with STATE.lock:
            STATE.brain.available = False
            STATE.brain.error = str(e)

        log(
            "FlyBrain initialization failed:"
        )

        log(str(e))


def frame_to_blobs(frame):
    if frame is None:
        return []

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce the screen to a manageable visual
    # representation for the fly eyes.
    small = cv2.resize(
        gray,
        (40, 20),
        interpolation=cv2.INTER_AREA
    )

    darkness = (
        1.0 -
        small.astype(np.float32) / 255.0
    )

    blobs = []

    for x in range(40):

        column = darkness[:, x]

        strength = float(
            np.mean(column)
        )

        if strength < 0.05:
            continue

        azimuth = (
            x / 39.0
        ) * 2.0 - 1.0

        blobs.append(
            Blob(
                center=float(azimuth),
                half_width=0.08,
                darkness=min(
                    1.0,
                    strength
                )
            )
        )

    return blobs


def brain_step(frame):
    if not STATE.brain.available:
        return

    brain = STATE.brain.brain
    eyes = STATE.brain.eyes

    if brain is None or eyes is None:
        return

    try:

        blobs = frame_to_blobs(
            frame
        )

        drive = eyes.drive(
            blobs
        )

        brain.step(
            drive
        )

        fired = brain.fired

        if hasattr(fired, "get"):
            fired = fired.get()

        fired = np.asarray(
            fired
        )

        fired_count = len(
            fired
        )

        with STATE.lock:

            STATE.brain.fired = (
                fired_count
            )

            STATE.brain.activity = (
                fired_count /
                max(1, brain.n)
            )

    except Exception as e:

        with STATE.lock:
            STATE.brain.error = str(e)

        log(
            f"Brain step error: {e}"
        )


# ============================================================
# FEATURES
# ============================================================

def current_features():
    with STATE.lock:

        v = STATE.vision
        b = STATE.brain

        player_y = (
            v.player_y /
            GAME_HEIGHT
        )

        velocity = np.clip(
            v.velocity_y / 1500.0,
            -1.0,
            1.0
        )

        grounded = (
            1.0
            if v.grounded
            else 0.0
        )

        if v.obstacle_found:

            distance = np.clip(
                v.obstacle_distance /
                STATE.scan_distance,
                0.0,
                1.0
            )

            height = np.clip(
                v.obstacle_h /
                250.0,
                0.0,
                1.0
            )

            width = np.clip(
                v.obstacle_w /
                250.0,
                0.0,
                1.0
            )

        else:

            distance = 1.0
            height = 0.0
            width = 0.0

        brain_activity = np.clip(
            b.activity * 1000.0,
            0.0,
            1.0
        )

        brain_fired = np.clip(
            b.fired / 1000.0,
            0.0,
            1.0
        )

        return np.asarray(
            [
                player_y,
                velocity,
                grounded,
                distance,
                height,
                width,
                v.obstacle_confidence,
                v.confidence,
                brain_activity,
                brain_fired,
            ],
            dtype=np.float32
        )


# ============================================================
# SIMPLE AI
# ============================================================

def simple_prediction():
    with STATE.lock:

        v = STATE.vision

        if not v.player_found:
            return 0.0

        if not v.obstacle_found:
            return 0.0

        distance = v.obstacle_distance

        height = v.obstacle_h

        # Closer obstacle = greater urgency.
        closeness = np.clip(
            1.0 -
            distance /
            STATE.scan_distance,
            0.0,
            1.0
        )

        height_score = np.clip(
            height / 100.0,
            0.0,
            1.0
        )

        ground_score = (
            1.0
            if v.grounded
            else 0.35
        )

        velocity_score = np.clip(
            abs(v.velocity_y) /
            1000.0,
            0.0,
            1.0
        )

        p = (
            closeness * 0.50
            +
            height_score * 0.20
            +
            ground_score * 0.20
            +
            velocity_score * 0.10
        )

        # Do not react to distant objects too aggressively.
        if distance > 300:
            p *= 0.45

        if distance < 55:
            p *= 0.80

        return float(
            np.clip(p, 0.0, 1.0)
        )


# ============================================================
# COMPLEX AI
# ============================================================

def complex_prediction():
    features = current_features()

    if features is None:
        return 0.0

    with STATE.lock:

        if (
            not STATE.model_ready
            or STATE.model_weights is None
        ):
            # Before training, use the simple system
            # as a temporary fallback.
            return simple_prediction() * 0.75

        weights = STATE.model_weights
        mean = STATE.model_mean
        std = STATE.model_std
        bias = STATE.model_bias

    normalized = (
        features - mean
    ) / np.maximum(
        std,
        1e-6
    )

    z = float(
        np.dot(
            normalized,
            weights
        )
        +
        bias
    )

    # Stable sigmoid
    z = np.clip(
        z,
        -30,
        30
    )

    probability = (
        1.0 /
        (1.0 + math.exp(-z))
    )

    return float(
        probability
    )


def predict_jump():
    if STATE.complexity == "Complex":
        return complex_prediction()

    return simple_prediction()


# ============================================================
# MOUSE INPUT
# ============================================================

def on_mouse_click(x, y, button, pressed):
    if mouse is None:
        return

    if button == mouse.Button.left:

        with STATE.lock:
            STATE.mouse_down = pressed


def start_mouse_listener():
    if mouse is None:
        log(
            "pynput not installed. "
            "Mouse input unavailable."
        )
        return None

    try:

        listener = mouse.Listener(
            on_click=on_mouse_click
        )

        listener.daemon = True
        listener.start()

        log("Mouse listener started.")

        return listener

    except Exception as e:

        log(
            f"Mouse listener error: {e}"
        )

        return None


# ============================================================
# KEYBOARD INPUT
# ============================================================

def update_keyboard_state():
    if keyboard is None:
        return

    try:

        down = keyboard.is_pressed(
            "space"
        )

        with STATE.lock:
            STATE.space_down = down

    except Exception:
        pass


def start_keyboard_hotkey():
    if keyboard is None:
        log(
            "keyboard package unavailable."
        )
        return

    try:

        keyboard.add_hotkey(
            "f8",
            emergency_stop
        )

        log(
            "F8 emergency stop enabled."
        )

    except Exception as e:

        log(
            f"F8 hotkey failed: {e}"
        )


# ============================================================
# EMERGENCY STOP
# ============================================================

def emergency_stop():
    with STATE.lock:

        STATE.emergency = True
        STATE.ai_enabled = False

    log(
        "!!! EMERGENCY STOP !!!"
    )


def reset_emergency():
    with STATE.lock:
        STATE.emergency = False

    log(
        "Emergency stop reset."
    )


# ============================================================
# ACTUAL JUMP
# ============================================================

def perform_jump(source="AI"):
    with STATE.lock:

        if STATE.emergency:
            return False

        if time.time() - STATE.last_jump_time < AI_COOLDOWN:
            return False

        control = STATE.control

    # Never let AI press another window accidentally.
    if not foreground_is_geometry_dash():
        return False

    success = False

    try:

        if control in (
            "Keyboard",
            "Both"
        ):

            if pyautogui is not None:
                pyautogui.press(
                    "space"
                )

                success = True

        if control in (
            "Mouse",
            "Both"
        ):

            if pyautogui is not None:
                pyautogui.click()

                success = True

    except Exception as e:

        log(
            f"Input error: {e}"
        )

    if success:

        with STATE.lock:
            STATE.last_jump_time = time.time()

        log(
            f"JUMP [{source}]"
        )

    return success


# ============================================================
# TRAINING INPUT STATE
# ============================================================

def get_current_input():
    with STATE.lock:

        space = STATE.space_down
        mouse_pressed = STATE.mouse_down

    if space and mouse_pressed:
        return "both"

    if space:
        return "space"

    if mouse_pressed:
        return "mouse"

    return "none"


# ============================================================
# TRAINING DATA
# ============================================================

def load_training_data():
    if not DATA_FILE.exists():
        return

    try:

        with open(
            DATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, list):

            with STATE.lock:
                STATE.samples = data

        log(
            f"Loaded "
            f"{len(data):,} training samples."
        )

    except Exception as e:

        log(
            f"Training data load failed: {e}"
        )


def save_training_data():
    try:

        with STATE.lock:
            data = list(
                STATE.samples
            )

        tmp = DATA_FILE.with_suffix(
            ".tmp"
        )

        with open(
            tmp,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f
            )

        tmp.replace(
            DATA_FILE
        )

    except Exception as e:

        log(
            f"Training save failed: {e}"
        )


def dataset_stats():
    with STATE.lock:

        total = len(
            STATE.samples
        )

        positive = sum(
            1
            for s in STATE.samples
            if int(s.get("label", 0)) == 1
        )

        negative = (
            total - positive
        )

    remaining = max(
        0,
        MIN_TRAINING_SAMPLES -
        total
    )

    progress = min(
        100.0,
        total /
        MIN_TRAINING_SAMPLES *
        100.0
    )

    ready = (
        total >= MIN_TRAINING_SAMPLES
        and
        positive >= MIN_JUMP_SAMPLES
        and
        negative >= MIN_NO_JUMP_SAMPLES
    )

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "remaining": remaining,
        "progress": progress,
        "ready": ready,
    }


def record_training_sample(
    label,
    input_source="none"
):
    features = current_features()

    if features is None:
        return

    with STATE.lock:

        sample = {
            "features":
                features.astype(
                    float
                ).tolist(),

            "label":
                int(bool(label)),

            "input":
                input_source,

            "timestamp":
                time.time(),

            "session":
                STATE.training_session_id,
        }

        STATE.samples.append(
            sample
        )

        STATE.training_session_samples += 1


# ============================================================
# LEARNING WORKER
# ============================================================

def learning_worker():
    last_input = "none"
    last_negative = 0.0
    last_save = 0.0

    while STATE.running:

        time.sleep(
            1.0 /
            LEARN_RECORD_HZ
        )

        update_keyboard_state()

        with STATE.lock:
            active = STATE.training_active

        if not active:
            continue

        current = get_current_input()

        now = time.time()

        # Record every new jump input.
        if current != "none":

            if current != last_input:

                record_training_sample(
                    label=1,
                    input_source=current
                )

        last_input = current

        # Record negative examples periodically.
        if now - last_negative > 0.20:

            if current == "none":

                record_training_sample(
                    label=0,
                    input_source="none"
                )

                last_negative = now

        if now - last_save > 10:

            save_training_data()

            last_save = now


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def start_training_session():
    with STATE.lock:

        STATE.training_active = True

        STATE.training_session_id = (
            time.strftime(
                "%Y%m%d_%H%M%S"
            )
            +
            "_"
            +
            secrets.token_hex(3)
        )

        STATE.training_session_samples = 0

    log(
        f"Training session started: "
        f"{STATE.training_session_id}"
    )


def stop_training_session():
    with STATE.lock:

        STATE.training_active = False

        session = (
            STATE.training_session_id
        )

        count = (
            STATE.training_session_samples
        )

        if session:

            STATE.sessions.append({
                "id": session,
                "samples": count,
                "timestamp": time.time(),
            })

    save_training_data()

    log(
        f"Training session stopped: "
        f"{count:,} samples."
    )


def delete_last_session():
    with STATE.lock:

        if not STATE.sessions:
            return False

        session_id = (
            STATE.sessions[-1]["id"]
        )

        before = len(
            STATE.samples
        )

        STATE.samples = [
            s
            for s in STATE.samples
            if s.get("session")
            != session_id
        ]

        after = len(
            STATE.samples
        )

        STATE.sessions.pop()

    save_training_data()

    log(
        f"Deleted session "
        f"{session_id}: "
        f"{before-after:,} samples."
    )

    return True


def clear_dataset():
    with STATE.lock:
        STATE.samples.clear()
        STATE.sessions.clear()

        STATE.model_ready = False
        STATE.model_weights = None

    save_training_data()

    if MODEL_FILE.exists():

        try:
            MODEL_FILE.unlink()
        except Exception:
            pass

    log(
        "Training dataset cleared."
    )


def export_dataset():
    path = filedialog.asksaveasfilename(
        title="Export FlyBrain dataset",
        defaultextension=".json",
        filetypes=[
            ("JSON", "*.json")
        ]
    )

    if not path:
        return

    with STATE.lock:

        data = list(
            STATE.samples
        )

    try:

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )

        log(
            f"Dataset exported to {path}"
        )

    except Exception as e:

        messagebox.showerror(
            "Export failed",
            str(e)
        )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def sigmoid(x):
    x = np.clip(
        x,
        -30,
        30
    )

    return (
        1.0 /
        (1.0 + np.exp(-x))
    )


def train_model():
    with STATE.lock:

        samples = list(
            STATE.samples
        )

    if len(samples) < 20:

        return (
            False,
            "Not enough samples."
        )

    X = []
    y = []

    for sample in samples:

        try:

            features = np.asarray(
                sample["features"],
                dtype=np.float64
            )

            label = int(
                sample["label"]
            )

            if len(features) != len(
                FEATURE_NAMES
            ):
                continue

            X.append(features)
            y.append(label)

        except Exception:
            continue

    if len(X) < 20:

        return (
            False,
            "Not enough valid samples."
        )

    X = np.asarray(
        X,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    if len(np.unique(y)) < 2:

        return (
            False,
            "Dataset needs both jump and no-jump samples."
        )

    # Shuffle
    rng = np.random.default_rng(
        42
    )

    indices = np.arange(
        len(X)
    )

    rng.shuffle(
        indices
    )

    X = X[indices]
    y = y[indices]

    split = int(
        len(X) * 0.80
    )

    X_train = X[:split]
    y_train = y[:split]

    X_val = X[split:]
    y_val = y[split:]

    mean = X_train.mean(
        axis=0
    )

    std = X_train.std(
        axis=0
    )

    std[
        std < 1e-6
    ] = 1.0

    X_train = (
        X_train - mean
    ) / std

    X_val = (
        X_val - mean
    ) / std

    weights = np.zeros(
        X_train.shape[1],
        dtype=np.float64
    )

    bias = 0.0

    # Class balancing
    positives = max(
        1,
        np.sum(y_train == 1)
    )

    negatives = max(
        1,
        np.sum(y_train == 0)
    )

    pos_weight = (
        len(y_train) /
        (2 * positives)
    )

    neg_weight = (
        len(y_train) /
        (2 * negatives)
    )

    learning_rate = 0.05

    for epoch in range(700):

        logits = (
            X_train @ weights
            +
            bias
        )

        predictions = sigmoid(
            logits
        )

        sample_weights = np.where(
            y_train == 1,
            pos_weight,
            neg_weight
        )

        error = (
            predictions - y_train
        )

        weighted_error = (
            error *
            sample_weights
        )

        grad_w = (
            X_train.T @
            weighted_error
        ) / len(X_train)

        grad_b = (
            np.sum(
                weighted_error
            )
            /
            len(X_train)
        )

        weights -= (
            learning_rate *
            grad_w
        )

        bias -= (
            learning_rate *
            grad_b
        )

    train_pred = (
        sigmoid(
            X_train @ weights +
            bias
        ) >= 0.5
    )

    val_pred = (
        sigmoid(
            X_val @ weights +
            bias
        ) >= 0.5
    )

    train_accuracy = float(
        np.mean(
            train_pred ==
            y_train.astype(bool)
        )
    )

    validation_accuracy = float(
        np.mean(
            val_pred ==
            y_val.astype(bool)
        )
    )

    np.savez(
        MODEL_FILE,
        weights=weights,
        mean=mean,
        std=std,
        bias=bias,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
    )

    with STATE.lock:

        STATE.model_weights = weights
        STATE.model_mean = mean
        STATE.model_std = std
        STATE.model_bias = float(
            bias
        )

        STATE.model_ready = True

        STATE.train_accuracy = (
            train_accuracy
        )

        STATE.validation_accuracy = (
            validation_accuracy
        )

    return (
        True,
        (
            f"Training complete. "
            f"Train={train_accuracy*100:.1f}% "
            f"Validation={validation_accuracy*100:.1f}%"
        )
    )


def train_model_async():
    def worker():

        log(
            "Training model..."
        )

        ok, result = train_model()

        if ok:
            log(result)
        else:
            log(
                f"Training failed: {result}"
            )

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


def load_model():
    if not MODEL_FILE.exists():
        return

    try:

        data = np.load(
            MODEL_FILE
        )

        with STATE.lock:

            STATE.model_weights = (
                data["weights"]
            )

            STATE.model_mean = (
                data["mean"]
            )

            STATE.model_std = (
                data["std"]
            )

            STATE.model_bias = float(
                data["bias"]
            )

            STATE.train_accuracy = float(
                data.get(
                    "train_accuracy",
                    0
                )
            )

            STATE.validation_accuracy = float(
                data.get(
                    "validation_accuracy",
                    0
                )
            )

            STATE.model_ready = True

        log(
            "AI model loaded."
        )

    except Exception as e:

        log(
            f"Model load failed: {e}"
        )


# ============================================================
# AI CONTROL LOOP
# ============================================================

def ai_worker():
    while STATE.running:

        time.sleep(
            1.0 / 30.0
        )

        with STATE.lock:

            mode = STATE.mode
            enabled = STATE.ai_enabled
            emergency = STATE.emergency

        if emergency:
            continue

        if mode not in (
            "AI TEST",
            "AI PLAY"
        ):
            continue

        if not enabled:
            continue

        probability = predict_jump()

        with STATE.lock:
            STATE.vision.jump_probability = (
                probability
            )

        if probability >= AI_THRESHOLD:

            if mode == "AI TEST":

                with STATE.lock:
                    STATE.vision.reason = (
                        "AI WOULD JUMP"
                    )

            elif mode == "AI PLAY":

                perform_jump(
                    "AI"
                )

        else:

            with STATE.lock:
                STATE.vision.reason = (
                    "AI WAIT"
                )


# ============================================================
# MAIN VISION WORKER
# ============================================================

def vision_worker():
    while STATE.running:

        start = time.perf_counter()

        frame = capture_screen()

        if frame is not None:

            with STATE.lock:
                STATE.frame = frame

            update_vision(
                frame
            )

            debug = draw_debug(
                frame
            )

            with STATE.lock:
                STATE.debug_frame = debug

        elapsed = (
            time.perf_counter() -
            start
        )

        delay = max(
            0,
            (1.0 / VISION_HZ)
            -
            elapsed
        )

        time.sleep(
            delay
        )


# ============================================================
# BRAIN WORKER
# ============================================================

def brain_worker():
    while STATE.running:

        start = time.perf_counter()

        with STATE.lock:
            frame = STATE.frame

        if frame is not None:
            brain_step(
                frame
            )

        elapsed = (
            time.perf_counter()
            -
            start
        )

        time.sleep(
            max(
                0,
                1.0 / BRAIN_HZ -
                elapsed
            )
        )


# ============================================================
# UI HELPERS
# ============================================================

def make_button(parent, text, command):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=PANEL2,
        fg=TEXT,
        activebackground="#303640",
        activeforeground=TEXT,
        relief="flat",
        padx=10,
        pady=7,
        cursor="hand2"
    )


def make_label(parent, text="", size=10):
    return tk.Label(
        parent,
        text=text,
        bg=PANEL,
        fg=TEXT,
        font=("Segoe UI", size)
    )


def panel(parent, title):
    frame = tk.Frame(
        parent,
        bg=PANEL,
        bd=1,
        relief="solid"
    )

    title_label = tk.Label(
        frame,
        text=title,
        bg=PANEL2,
        fg=TEXT,
        font=(
            "Segoe UI",
            11,
            "bold"
        ),
        anchor="w",
        padx=10,
        pady=7
    )

    title_label.pack(
        fill="x"
    )

    body = tk.Frame(
        frame,
        bg=PANEL
    )

    body.pack(
        fill="both",
        expand=True,
        padx=8,
        pady=8
    )

    return frame, body


# ============================================================
# LOGIN WINDOW
# ============================================================

class LoginWindow:
    def __init__(self):
        self.root = tk.Tk()

        self.root.title(
            "FlyBrain - Login"
        )

        self.root.geometry(
            "440x430"
        )

        self.root.configure(
            bg=BG
        )

        self.username = tk.StringVar()
        self.password = tk.StringVar()

        self.build()

        self.root.mainloop()

    def build(self):

        title = tk.Label(
            self.root,
            text="FLYBRAIN",
            bg=BG,
            fg=ACCENT,
            font=(
                "Segoe UI",
                27,
                "bold"
            )
        )

        title.pack(
            pady=(40, 4)
        )

        subtitle = tk.Label(
            self.root,
            text="Geometry Dash AI Trainer",
            bg=BG,
            fg=SUBTEXT,
            font=(
                "Segoe UI",
                11
            )
        )

        subtitle.pack(
            pady=(0, 30)
        )

        box = tk.Frame(
            self.root,
            bg=PANEL,
            padx=25,
            pady=25
        )

        box.pack(
            padx=30,
            fill="x"
        )

        tk.Label(
            box,
            text="Username",
            bg=PANEL,
            fg=TEXT
        ).pack(
            anchor="w"
        )

        tk.Entry(
            box,
            textvariable=self.username,
            bg=PANEL2,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat"
        ).pack(
            fill="x",
            pady=(4, 15),
            ipady=7
        )

        tk.Label(
            box,
            text="Password",
            bg=PANEL,
            fg=TEXT
        ).pack(
            anchor="w"
        )

        tk.Entry(
            box,
            textvariable=self.password,
            show="*",
            bg=PANEL2,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat"
        ).pack(
            fill="x",
            pady=(4, 20),
            ipady=7
        )

        make_button(
            box,
            "LOGIN",
            self.login
        ).pack(
            fill="x",
            pady=4
        )

        make_button(
            box,
            "CREATE ACCOUNT",
            self.create
        ).pack(
            fill="x",
            pady=4
        )

        tk.Label(
            self.root,
            text=(
                "Local accounts are stored on this PC."
            ),
            bg=BG,
            fg=SUBTEXT,
            font=("Segoe UI", 9)
        ).pack(
            pady=20
        )

    def login(self):

        username = (
            self.username.get()
        )

        password = (
            self.password.get()
        )

        ok, account = verify_account(
            username,
            password
        )

        if not ok:

            messagebox.showerror(
                "Login failed",
                "Incorrect username or password."
            )

            return

        self.root.destroy()

        Dashboard(
            username=username,
            account=account
        )

    def create(self):

        ok, message = create_account(
            self.username.get(),
            self.password.get()
        )

        if ok:

            messagebox.showinfo(
                "Account",
                message
            )

        else:

            messagebox.showerror(
                "Account",
                message
            )


# ============================================================
# DASHBOARD
# ============================================================

class Dashboard:

    def __init__(
        self,
        username,
        account
    ):

        self.username = username
        self.account = account

        self.root = tk.Tk()

        self.root.title(
            "FlyBrain Geometry Dash AI"
        )

        self.root.geometry(
            "1450x900"
        )

        self.root.minsize(
            1100,
            700
        )

        self.root.configure(
            bg=BG
        )

        self.build()

        self.root.after(
            100,
            self.refresh
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.root.mainloop()

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    def build(self):

        header = tk.Frame(
            self.root,
            bg=PANEL2,
            height=55
        )

        header.pack(
            fill="x"
        )

        tk.Label(
            header,
            text="FLYBRAIN",
            bg=PANEL2,
            fg=ACCENT,
            font=(
                "Segoe UI",
                16,
                "bold"
            )
        ).pack(
            side="left",
            padx=15
        )

        tk.Label(
            header,
            text="Geometry Dash AI Trainer",
            bg=PANEL2,
            fg=SUBTEXT,
            font=("Segoe UI", 10)
        ).pack(
            side="left"
        )

        self.emergency_label = tk.Label(
            header,
            text="● READY",
            bg=PANEL2,
            fg=GREEN,
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        )

        self.emergency_label.pack(
            side="right",
            padx=15
        )

        # Main movable panes
        main = tk.PanedWindow(
            self.root,
            orient="horizontal",
            bg=BG,
            sashwidth=7
        )

        main.pack(
            fill="both",
            expand=True
        )

        left = tk.Frame(
            main,
            bg=BG
        )

        right = tk.Frame(
            main,
            bg=BG
        )

        main.add(
            left,
            minsize=400
        )

        main.add(
            right,
            minsize=500
        )

        self.build_controls(
            left
        )

        self.build_visuals(
            right
        )

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    def build_controls(self, parent):

        controls, body = panel(
            parent,
            "CONTROL CENTER"
        )

        controls.pack(
            fill="x",
            padx=8,
            pady=8
        )

        # Mode
        tk.Label(
            body,
            text="Mode",
            bg=PANEL,
            fg=TEXT
        ).pack(
            anchor="w"
        )

        self.mode_var = tk.StringVar(
            value="OBSERVE"
        )

        mode_box = ttk.Combobox(
            body,
            textvariable=self.mode_var,
            values=[
                "OBSERVE",
                "LEARN",
                "AI TEST",
                "AI PLAY"
            ],
            state="readonly"
        )

        mode_box.pack(
            fill="x",
            pady=(3, 10)
        )

        mode_box.bind(
            "<<ComboboxSelected>>",
            self.mode_changed
        )

        # Complexity
        tk.Label(
            body,
            text="AI Complexity",
            bg=PANEL,
            fg=TEXT
        ).pack(
            anchor="w"
        )

        self.complexity_var = tk.StringVar(
            value="Simple"
        )

        complexity_frame = tk.Frame(
            body,
            bg=PANEL
        )

        complexity_frame.pack(
            fill="x",
            pady=(3, 10)
        )

        tk.Radiobutton(
            complexity_frame,
            text="Simple",
            variable=self.complexity_var,
            value="Simple",
            command=self.complexity_changed,
            bg=PANEL,
            fg=TEXT,
            selectcolor=PANEL2,
            activebackground=PANEL,
            activeforeground=TEXT
        ).pack(
            side="left"
        )

        tk.Radiobutton(
            complexity_frame,
            text="Complex",
            variable=self.complexity_var,
            value="Complex",
            command=self.complexity_changed,
            bg=PANEL,
            fg=TEXT,
            selectcolor=PANEL2,
            activebackground=PANEL,
            activeforeground=TEXT
        ).pack(
            side="left"
        )

        # Input
        tk.Label(
            body,
            text="Input Method",
            bg=PANEL,
            fg=TEXT
        ).pack(
            anchor="w"
        )

        self.control_var = tk.StringVar(
            value="Keyboard"
        )

        ttk.Combobox(
            body,
            textvariable=self.control_var,
            values=[
                "Keyboard",
                "Mouse",
                "Both"
            ],
            state="readonly"
        ).pack(
            fill="x",
            pady=(3, 10)
        )

        # AI enable
        self.ai_button = make_button(
            body,
            "ENABLE AI",
            self.toggle_ai
        )

        self.ai_button.pack(
            fill="x",
            pady=4
        )

        make_button(
            body,
            "F8 EMERGENCY STOP",
            emergency_stop
        ).pack(
            fill="x",
            pady=4
        )

        make_button(
            body,
            "RESET EMERGENCY STOP",
            reset_emergency
        ).pack(
            fill="x",
            pady=4
        )

        # Calibration
        cal, calbody = panel(
            parent,
            "VISION CALIBRATION"
        )

        cal.pack(
            fill="x",
            padx=8,
            pady=8
        )

        self.player_x_var = tk.IntVar(
            value=DEFAULT_PLAYER_X
        )

        self.player_y_var = tk.IntVar(
            value=DEFAULT_PLAYER_Y
        )

        self.ground_y_var = tk.IntVar(
            value=DEFAULT_GROUND_Y
        )

        self.scan_var = tk.IntVar(
            value=DEFAULT_SCAN_DISTANCE
        )

        self.add_slider(
            calbody,
            "Player X",
            self.player_x_var,
            50,
            1200
        )

        self.add_slider(
            calbody,
            "Player Y",
            self.player_y_var,
            100,
            950
        )

        self.add_slider(
            calbody,
            "Ground Y",
            self.ground_y_var,
            300,
            1000
        )

        self.add_slider(
            calbody,
            "Scan Distance",
            self.scan_var,
            100,
            1000
        )

        # Training
        self.build_training(
            parent
        )

        # Developer
        self.build_developer(
            parent
        )

    def add_slider(
        self,
        parent,
        label,
        variable,
        low,
        high
    ):

        row = tk.Frame(
            parent,
            bg=PANEL
        )

        row.pack(
            fill="x",
            pady=3
        )

        tk.Label(
            row,
            text=label,
            bg=PANEL,
            fg=TEXT,
            width=15,
            anchor="w"
        ).pack(
            side="left"
        )

        scale = tk.Scale(
            row,
            from_=low,
            to=high,
            orient="horizontal",
            variable=variable,
            bg=PANEL,
            fg=TEXT,
            highlightthickness=0,
            troughcolor=PANEL2,
            activebackground=ACCENT
        )

        scale.pack(
            side="left",
            fill="x",
            expand=True
        )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def build_training(self, parent):

        frame, body = panel(
            parent,
            "LEARNING / TRAINING"
        )

        frame.pack(
            fill="x",
            padx=8,
            pady=8
        )

        self.total_label = make_label(
            body
        )

        self.total_label.pack(
            anchor="w"
        )

        self.jump_label = make_label(
            body
        )

        self.jump_label.pack(
            anchor="w"
        )

        self.nojump_label = make_label(
            body
        )

        self.nojump_label.pack(
            anchor="w"
        )

        self.remaining_label = make_label(
            body
        )

        self.remaining_label.pack(
            anchor="w"
        )

        self.ready_label = make_label(
            body,
            size=11
        )

        self.ready_label.pack(
            anchor="w",
            pady=4
        )

        self.progress = ttk.Progressbar(
            body,
            maximum=100
        )

        self.progress.pack(
            fill="x",
            pady=5
        )

        self.session_label = make_label(
            body
        )

        self.session_label.pack(
            anchor="w"
        )

        button_row = tk.Frame(
            body,
            bg=PANEL
        )

        button_row.pack(
            fill="x",
            pady=5
        )

        make_button(
            button_row,
            "START LEARNING",
            start_training_session
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=2
        )

        make_button(
            button_row,
            "STOP",
            stop_training_session
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=2
        )

        make_button(
            body,
            "TRAIN COMPLEX MODEL",
            train_model_async
        ).pack(
            fill="x",
            pady=3
        )

        make_button(
            body,
            "DELETE LAST SESSION",
            delete_last_session
        ).pack(
            fill="x",
            pady=3
        )

        make_button(
            body,
            "EXPORT DATASET",
            export_dataset
        ).pack(
            fill="x",
            pady=3
        )

        make_button(
            body,
            "CLEAR DATASET",
            self.confirm_clear
        ).pack(
            fill="x",
            pady=3
        )

        self.model_label = make_label(
            body
        )

        self.model_label.pack(
            anchor="w",
            pady=4
        )

    def confirm_clear(self):

        result = messagebox.askyesno(
            "Clear dataset?",
            "Delete all training samples?"
        )

        if result:
            clear_dataset()

    # --------------------------------------------------------
    # VISUALS
    # --------------------------------------------------------

    def build_visuals(self, parent):

        preview, body = panel(
            parent,
            "VISION DEBUGGER"
        )

        preview.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        self.preview = tk.Label(
            body,
            bg="black",
            fg=TEXT,
            text="Waiting for camera..."
        )

        self.preview.pack(
            fill="both",
            expand=True
        )

        stats, statsbody = panel(
            parent,
            "LIVE STATISTICS"
        )

        stats.pack(
            fill="x",
            padx=8,
            pady=8
        )

        self.stats_label = tk.Label(
            statsbody,
            text="",
            bg=PANEL,
            fg=TEXT,
            justify="left",
            anchor="w",
            font=(
                "Consolas",
                10
            )
        )

        self.stats_label.pack(
            fill="x"
        )

        discord, dbody = panel(
            parent,
            "DISCORD"
        )

        discord.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            dbody,
            text=(
                "Linking uses Discord OAuth2.\n"
                "Never enter your Discord password here."
            ),
            bg=PANEL,
            fg=SUBTEXT,
            justify="left"
        ).pack(
            anchor="w"
        )

        make_button(
            dbody,
            "OPEN DISCORD",
            self.discord_link
        ).pack(
            fill="x",
            pady=5
        )

        logs, logbody = panel(
            parent,
            "LOG"
        )

        logs.pack(
            fill="both",
            expand=False,
            padx=8,
            pady=8
        )

        self.log_text = tk.Text(
            logbody,
            height=8,
            bg="#0b0d10",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat"
        )

        self.log_text.pack(
            fill="both",
            expand=True
        )

    # --------------------------------------------------------
    # DEVELOPER
    # --------------------------------------------------------

    def build_developer(self, parent):

        frame, body = panel(
            parent,
            "DEVELOPER TOOLS"
        )

        frame.pack(
            fill="x",
            padx=8,
            pady=8
        )

        account_role = (
            self.account.get(
                "role",
                "user"
            )
        )

        if account_role != "developer":

            tk.Label(
                body,
                text=(
                    "Developer mode unavailable "
                    "for this account."
                ),
                bg=PANEL,
                fg=SUBTEXT
            ).pack(
                anchor="w"
            )

            return

        tk.Label(
            body,
            text="DEVELOPER MODE ENABLED",
            bg=PANEL,
            fg=YELLOW,
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=3
        )

        self.dev_info = tk.Label(
            body,
            text="",
            bg=PANEL,
            fg=TEXT,
            justify="left",
            anchor="w",
            font=("Consolas", 9)
        )

        self.dev_info.pack(
            fill="x",
            pady=5
        )

        make_button(
            body,
            "SAVE DEBUG FRAME",
            self.save_debug
        ).pack(
            fill="x",
            pady=2
        )

        make_button(
            body,
            "RESET FLYBRAIN",
            self.reset_brain
        ).pack(
            fill="x",
            pady=2
        )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    def mode_changed(self, event=None):

        mode = self.mode_var.get()

        with STATE.lock:

            STATE.mode = mode

            if mode == "LEARN":

                STATE.ai_enabled = False

            elif mode in (
                "AI TEST",
                "AI PLAY"
            ):

                STATE.ai_enabled = True

            else:

                STATE.ai_enabled = False

        log(
            f"Mode changed to {mode}"
        )

    def complexity_changed(self):

        with STATE.lock:

            STATE.complexity = (
                self.complexity_var.get()
            )

        log(
            f"AI complexity: "
            f"{STATE.complexity}"
        )

    def toggle_ai(self):

        with STATE.lock:

            STATE.ai_enabled = (
                not STATE.ai_enabled
            )

            enabled = STATE.ai_enabled

        log(
            f"AI {'enabled' if enabled else 'disabled'}"
        )

    # --------------------------------------------------------
    # DISCORD
    # --------------------------------------------------------

    def discord_link(self):

        if not DISCORD_CLIENT_ID:

            messagebox.showinfo(
                "Discord",
                (
                    "Discord OAuth is not configured yet.\n\n"
                    "Set the environment variable:\n"
                    "FLYBRAIN_DISCORD_CLIENT_ID"
                )
            )

            return

        url = (
            "https://discord.com/oauth2/authorize"
            f"?client_id={DISCORD_CLIENT_ID}"
            "&response_type=code"
            "&redirect_uri="
            +
            webbrowser.quote(DISCORD_REDIRECT)
            +
            "&scope=identify"
        )

        webbrowser.open(
            url
        )

    # --------------------------------------------------------
    # DEVELOPER
    # --------------------------------------------------------

    def save_debug(self):

        with STATE.lock:
            frame = STATE.debug_frame

        if frame is None:
            return

        filename = (
            DEBUG_DIR /
            (
                "debug_"
                +
                time.strftime(
                    "%Y%m%d_%H%M%S"
                )
                +
                ".png"
            )
        )

        if cv2.imwrite(
            str(filename),
            frame
        ):

            log(
                f"Debug frame saved: {filename}"
            )

    def reset_brain(self):

        with STATE.lock:

            brain = STATE.brain.brain

        if brain is None:
            return

        try:

            brain.reset()

            log(
                "FlyBrain reset."
            )

        except Exception as e:

            log(
                f"Brain reset failed: {e}"
            )

    # --------------------------------------------------------
    # REFRESH UI
    # --------------------------------------------------------

    def refresh(self):

        try:

            # Calibration
            with STATE.lock:

                STATE.player_x_cal = (
                    self.player_x_var.get()
                )

                STATE.player_y_cal = (
                    self.player_y_var.get()
                )

                STATE.ground_y_cal = (
                    self.ground_y_var.get()
                )

                STATE.scan_distance = (
                    self.scan_var.get()
                )

                v = STATE.vision
                b = STATE.brain

            # Training stats
            stats = dataset_stats()

            self.total_label.config(
                text=(
                    f"Total samples: "
                    f"{stats['total']:,}"
                )
            )

            self.jump_label.config(
                text=(
                    f"Jump samples: "
                    f"{stats['positive']:,} "
                    f"/ {MIN_JUMP_SAMPLES:,}"
                )
            )

            self.nojump_label.config(
                text=(
                    f"No-jump samples: "
                    f"{stats['negative']:,} "
                    f"/ {MIN_NO_JUMP_SAMPLES:,}"
                )
            )

            self.remaining_label.config(
                text=(
                    f"Samples needed: "
                    f"{stats['remaining']:,}"
                )
            )

            self.progress["value"] = (
                stats["progress"]
            )

            if stats["ready"]:

                self.ready_label.config(
                    text="● READY TO TRAIN",
                    fg=GREEN
                )

            else:

                self.ready_label.config(
                    text="● MORE DATA NEEDED",
                    fg=YELLOW
                )

            with STATE.lock:

                if STATE.training_active:

                    session = (
                        STATE.training_session_id
                    )

                    count = (
                        STATE.training_session_samples
                    )

                    self.session_label.config(
                        text=(
                            f"Recording: "
                            f"{session} "
                            f"({count:,} samples)"
                        )
                    )

                else:

                    self.session_label.config(
                        text="Not recording."
                    )

            # Model
            with STATE.lock:

                if STATE.model_ready:

                    self.model_label.config(
                        text=(
                            f"Model READY | "
                            f"Train: "
                            f"{STATE.train_accuracy*100:.1f}% | "
                            f"Validation: "
                            f"{STATE.validation_accuracy*100:.1f}%"
                        ),
                        fg=GREEN
                    )

                else:

                    self.model_label.config(
                        text=(
                            "Model not trained."
                        ),
                        fg=SUBTEXT
                    )

            # Emergency
            with STATE.lock:

                emergency = STATE.emergency
                ai = STATE.ai_enabled

            if emergency:

                self.emergency_label.config(
                    text="● EMERGENCY STOP",
                    fg=RED
                )

            elif ai:

                self.emergency_label.config(
                    text="● AI ACTIVE",
                    fg=YELLOW
                )

            else:

                self.emergency_label.config(
                    text="● READY",
                    fg=GREEN
                )

            # Statistics
            stats_text = (
                f"PLAYER\n"
                f"  Found:       {v.player_found}\n"
                f"  X:           {v.player_x:.1f}\n"
                f"  Y:           {v.player_y:.1f}\n"
                f"  Confidence:  {v.confidence:.3f}\n"
                f"  Grounded:    {v.grounded}\n"
                f"  Velocity Y:  {v.velocity_y:.1f}\n\n"

                f"OBSTACLE\n"
                f"  Found:       {v.obstacle_found}\n"
                f"  Distance:    {v.obstacle_distance:.1f}\n"
                f"  Width:       {v.obstacle_w:.1f}\n"
                f"  Height:      {v.obstacle_h:.1f}\n"
                f"  Confidence:  {v.obstacle_confidence:.3f}\n\n"

                f"AI\n"
                f"  Probability: {v.jump_probability:.3f}\n"
                f"  Reason:      {v.reason}\n\n"

                f"PERFORMANCE\n"
                f"  FPS:         {v.fps:.1f}\n"
                f"  Frame:       {v.frame_ms:.2f} ms\n\n"

                f"FLYBRAIN\n"
                f"  Available:   {b.available}\n"
                f"  Neurons:     {b.total_neurons:,}\n"
                f"  Fired:       {b.fired:,}\n"
                f"  Activity:    {b.activity:.6f}\n"
            )

            self.stats_label.config(
                text=stats_text
            )

            # Debug image
            self.update_preview()

            # Logs
            with STATE.lock:
                logs = list(
                    STATE.logs
                )

            self.log_text.delete(
                "1.0",
                "end"
            )

            self.log_text.insert(
                "end",
                "\n".join(logs)
            )

            self.log_text.see(
                "end"
            )

            # Developer
            if hasattr(
                self,
                "dev_info"
            ):

                self.dev_info.config(
                    text=(
                        f"Python: {sys.version.split()[0]}\n"
                        f"OpenCV: "
                        f"{cv2.__version__ if cv2 else 'N/A'}\n"
                        f"FlyBrain: "
                        f"{'YES' if FlyBrain else 'NO'}\n"
                        f"CUDA brain: "
                        f"{b.available}\n"
                        f"Capture: "
                        f"{GAME_WIDTH}x{GAME_HEIGHT}\n"
                        f"Model file: "
                        f"{MODEL_FILE.name}\n"
                        f"Training file: "
                        f"{DATA_FILE.name}"
                    )
                )

        except Exception as e:

            # Don't kill the UI because of a refresh issue.
            print(
                "UI refresh error:",
                e
            )

        if STATE.running:

            self.root.after(
                100,
                self.refresh
            )

    def update_preview(self):

        if Image is None or ImageTk is None:
            return

        with STATE.lock:
            frame = STATE.debug_frame

        if frame is None:
            return

        try:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                rgb
            )

            # Keep preview from becoming enormous.
            max_w = 850
            max_h = 570

            image.thumbnail(
                (
                    max_w,
                    max_h
                )
            )

            photo = ImageTk.PhotoImage(
                image
            )

            self.preview.config(
                image=photo,
                text=""
            )

            self.preview.image = photo

        except Exception:
            pass

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):

        with STATE.lock:

            STATE.running = False
            STATE.ai_enabled = False
            STATE.training_active = False

        try:

            if keyboard is not None:
                keyboard.unhook_all_hotkeys()

        except Exception:
            pass

        save_training_data()

        self.root.destroy()


# ============================================================
# STARTUP
# ============================================================

def initialize():

    log(
        "Starting FlyBrain Geometry Dash Trainer..."
    )

    log(
        f"Python: {sys.version.split()[0]}"
    )

    log(
        f"Capture: "
        f"{GAME_WIDTH}x{GAME_HEIGHT}"
    )

    if cv2 is None:
        log(
            "WARNING: OpenCV unavailable."
        )

    if pyautogui is None:
        log(
            "WARNING: PyAutoGUI unavailable."
        )

    if mss is None:
        log(
            "WARNING: MSS unavailable."
        )

    load_training_data()

    load_model()

    setup_capture()

    start_mouse_listener()

    start_keyboard_hotkey()

    # FlyBrain takes some time to initialize,
    # so do not freeze the UI.
    threading.Thread(
        target=initialize_brain,
        daemon=True
    ).start()

    threading.Thread(
        target=vision_worker,
        daemon=True
    ).start()

    threading.Thread(
        target=brain_worker,
        daemon=True
    ).start()

    threading.Thread(
        target=learning_worker,
        daemon=True
    ).start()

    threading.Thread(
        target=ai_worker,
        daemon=True
    ).start()

    log(
        "Workers started."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        initialize()

        LoginWindow()

    except KeyboardInterrupt:

        STATE.running = False

    except Exception:

        traceback.print_exc()

        try:
            messagebox.showerror(
                "FlyBrain crashed",
                traceback.format_exc()
            )
        except Exception:
            pass