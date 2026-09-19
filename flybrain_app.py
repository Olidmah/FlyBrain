"""
FlyBrain - Geometry Dash AI
Single-file desktop application.

Features:
- Local account system
- Robust handling of old/corrupt account JSON
- Geometry Dash screen capture
- Simple / Complex AI modes
- Observe / Learn / AI Test / AI Play
- Entertainment Mode
- F8 emergency stop
- Training dataset
- Logistic-regression decision model
- FlyBrain simulated-brain processing
- Vision debugger
- Training statistics
- Bottom-right entertainment window
- No GD password storage

IMPORTANT:
This program controls keyboard/mouse input when AI Play or
Entertainment Mode is active. F8 is the emergency stop.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

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
    from flybrain import FlyBrain, Eyes, Blob
    FLYBRAIN_AVAILABLE = True
except Exception:
    FlyBrain = None
    Eyes = None
    Blob = None
    FLYBRAIN_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path.home() / "FlyBrain"
APP_DIR.mkdir(exist_ok=True)

ACCOUNT_FILE = APP_DIR / "accounts.json"
DATASET_FILE = APP_DIR / "training_data.json"
MODEL_FILE = APP_DIR / "model.json"
SETTINGS_FILE = APP_DIR / "settings.json"

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 1024

GD_REGION = {
    "left": 0,
    "top": 0,
    "width": SCREEN_WIDTH,
    "height": SCREEN_HEIGHT,
}


# ============================================================
# COLORS / UI
# ============================================================

BG = "#111318"
PANEL = "#191c23"
PANEL2 = "#20242d"
TEXT = "#f1f3f5"
MUTED = "#9ca3af"
ACCENT = "#6ea8fe"
GREEN = "#64d98b"
RED = "#ff6b6b"
YELLOW = "#ffd166"


# ============================================================
# ACCOUNT SYSTEM
# ============================================================

def password_hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        180_000,
    ).hex()


def load_accounts() -> dict:
    """
    Loads accounts while safely ignoring malformed old entries.

    This specifically prevents:
        TypeError: string indices must be integers
    """
    if not ACCOUNT_FILE.exists():
        return {}

    try:
        with ACCOUNT_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    cleaned = {}

    for username, account in raw.items():
        if not isinstance(username, str):
            continue

        if not isinstance(account, dict):
            continue

        salt = account.get("salt")
        stored_hash = account.get("hash")

        if not isinstance(salt, str):
            continue

        if not isinstance(stored_hash, str):
            continue

        cleaned[username] = {
            "salt": salt,
            "hash": stored_hash,
            "role": account.get("role", "user"),
        }

    return cleaned


def save_accounts(accounts: dict):
    temp = ACCOUNT_FILE.with_suffix(".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=2)

    temp.replace(ACCOUNT_FILE)


def create_account(username: str, password: str):
    username = username.strip()

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 4:
        return False, "Password must contain at least 4 characters."

    accounts = load_accounts()

    if username in accounts:
        return False, "That username already exists."

    salt = secrets.token_hex(16)

    accounts[username] = {
        "salt": salt,
        "hash": password_hash(password, salt),
        "role": "user",
    }

    save_accounts(accounts)

    return True, "Account created."


def verify_account(username: str, password: str):
    accounts = load_accounts()

    if not isinstance(accounts, dict):
        return False, None

    account = accounts.get(username)

    if not isinstance(account, dict):
        return False, None

    salt = account.get("salt")
    stored_hash = account.get("hash")

    if not isinstance(salt, str):
        return False, None

    if not isinstance(stored_hash, str):
        return False, None

    try:
        expected = password_hash(password, salt)
    except Exception:
        return False, None

    if secrets.compare_digest(expected, stored_hash):
       if username.lower() == "olidmah":
        account["role"] = "developer"
    return True, account

    return False, None


# ============================================================
# DATASET
# ============================================================

def load_dataset() -> list:
    if not DATASET_FILE.exists():
        return []

    try:
        with DATASET_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def save_dataset(data: list):
    temp = DATASET_FILE.with_suffix(".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    temp.replace(DATASET_FILE)


def add_training_sample(features, jump):
    data = load_dataset()

    data.append({
        "features": [float(x) for x in features],
        "jump": int(bool(jump)),
        "time": time.time(),
    })

    save_dataset(data)


# ============================================================
# SIMPLE LOGISTIC MODEL
# ============================================================

class LogisticModel:
    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def predict_probability(self, features):
        if self.weights is None:
            return 0.5

        x = np.asarray(features, dtype=np.float64)

        if len(x) != len(self.weights):
            return 0.5

        z = float(np.dot(self.weights, x) + self.bias)

        z = max(-30.0, min(30.0, z))

        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, features):
        return self.predict_probability(features) >= 0.5

    def train(self, data, epochs=500, learning_rate=0.03):
        if len(data) < 20:
            return False

        X = np.array(
            [item["features"] for item in data],
            dtype=np.float64,
        )

        y = np.array(
            [item["jump"] for item in data],
            dtype=np.float64,
        )

        if len(X.shape) != 2:
            return False

        self.weights = np.zeros(X.shape[1], dtype=np.float64)
        self.bias = 0.0

        for _ in range(epochs):
            z = X @ self.weights + self.bias
            z = np.clip(z, -30, 30)

            p = 1.0 / (1.0 + np.exp(-z))

            grad_w = (X.T @ (p - y)) / len(X)
            grad_b = float(np.mean(p - y))

            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b

        return True

    def save(self):
        if self.weights is None:
            return

        with MODEL_FILE.open("w", encoding="utf-8") as f:
            json.dump({
                "weights": self.weights.tolist(),
                "bias": self.bias,
            }, f)

    def load(self):
        if not MODEL_FILE.exists():
            return

        try:
            with MODEL_FILE.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            weights = obj.get("weights")

            if isinstance(weights, list):
                self.weights = np.asarray(weights, dtype=np.float64)

            self.bias = float(obj.get("bias", 0.0))

        except Exception:
            self.weights = None
            self.bias = 0.0


# ============================================================
# VISION
# ============================================================

@dataclass
class VisionResult:
    player_x: float = 0.0
    player_y: float = 0.0
    player_confidence: float = 0.0
    obstacle_x: float = 0.0
    obstacle_y: float = 0.0
    obstacle_distance: float = 9999.0
    obstacle_width: float = 0.0
    ground_y: float = 0.0
    vertical_velocity: float = 0.0
    horizontal_velocity: float = 0.0
    airborne: bool = False
    jump_hint: bool = False


class GDVision:
    """
    Geometry Dash screen analyzer.

    This deliberately keeps the detector conservative. A generic
    contour is not automatically treated as the player.
    """

    def __init__(self):
        self.previous_gray = None
        self.previous_player_x = None
        self.previous_player_y = None
        self.last = VisionResult()

    def reset(self):
        self.previous_gray = None
        self.previous_player_x = None
        self.previous_player_y = None
        self.last = VisionResult()

    def analyze(self, frame):
        result = VisionResult()

        if frame is None or cv2 is None:
            self.last = result
            return result

        try:
            frame = np.ascontiguousarray(frame[:, :, :3])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Geometry Dash generally has a large amount of relatively
            # stable background. Motion is useful as one signal but
            # should never be treated as the player by itself.
            motion_score = 0.0

            if self.previous_gray is not None:
                diff = cv2.absdiff(gray, self.previous_gray)
                motion_score = float(np.mean(diff))

            self.previous_gray = gray

            h, w = gray.shape

            # Conservative gameplay region.
            # Ignore the top UI and very bottom edge.
            y1 = int(h * 0.20)
            y2 = int(h * 0.90)

            crop = frame[y1:y2, :]

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Bright/saturated objects.
            mask1 = cv2.inRange(
                hsv,
                np.array([0, 80, 80]),
                np.array([179, 255, 255]),
            )

            kernel = np.ones((3, 3), np.uint8)
            mask1 = cv2.morphologyEx(
                mask1,
                cv2.MORPH_OPEN,
                kernel,
            )

            contours, _ = cv2.findContours(
                mask1,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            candidates = []

            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)

                if cw < 4 or ch < 4:
                    continue

                if cw > 150 or ch > 150:
                    continue

                area = cw * ch

                if area < 20:
                    continue

                ratio = cw / max(ch, 1)

                if ratio > 5.0 or ratio < 0.2:
                    continue

                center_x = x + cw / 2
                center_y = y + ch / 2 + y1

                candidates.append(
                    (
                        center_x,
                        center_y,
                        cw,
                        ch,
                        area,
                    )
                )

            # Track near the previous player position when possible.
            player = None

            if self.previous_player_x is not None:
                best_score = float("inf")

                for c in candidates:
                    x, y, cw, ch, area = c

                    distance = math.hypot(
                        x - self.previous_player_x,
                        y - self.previous_player_y,
                    )

                    if distance < 100:
                        size_penalty = abs(cw - ch) * 3

                        score = distance + size_penalty

                        if score < best_score:
                            best_score = score
                            player = c

            # If tracking failed, use small square-ish candidates.
            if player is None:
                square_candidates = []

                for c in candidates:
                    x, y, cw, ch, area = c

                    ratio = cw / max(ch, 1)

                    if 0.65 <= ratio <= 1.5 and 8 <= cw <= 50:
                        square_candidates.append(c)

                if square_candidates:
                    # Prefer candidates around the usual GD play area.
                    player = min(
                        square_candidates,
                        key=lambda c: abs(c[0] - 300),
                    )

            if player is not None:
                px, py, pw, ph, _ = player

                result.player_x = px
                result.player_y = py
                result.player_confidence = 0.65

                if self.previous_player_x is not None:
                    result.horizontal_velocity = (
                        px - self.previous_player_x
                    )

                    result.vertical_velocity = (
                        py - self.previous_player_y
                    )

                self.previous_player_x = px
                self.previous_player_y = py

            # Obstacles are candidates ahead of the player.
            if player is not None:
                px = player[0]
                py = player[1]

                obstacles = []

                for c in candidates:
                    x, y, cw, ch, area = c

                    if x <= px + 15:
                        continue

                    if abs(y - py) > 260:
                        continue

                    distance = x - px

                    obstacles.append(
                        (
                            distance,
                            x,
                            y,
                            cw,
                            ch,
                        )
                    )

                if obstacles:
                    obstacles.sort(key=lambda item: item[0])

                    distance, ox, oy, ow, oh = obstacles[0]

                    result.obstacle_x = ox
                    result.obstacle_y = oy
                    result.obstacle_distance = distance
                    result.obstacle_width = ow

            # Estimate ground from lower image region.
            lower = gray[int(h * 0.72):int(h * 0.90)]

            if lower.size:
                row_variance = np.var(lower, axis=1)

                if len(row_variance):
                    row = int(np.argmax(row_variance))
                    result.ground_y = int(h * 0.72) + row

            if result.player_confidence > 0:
                result.airborne = (
                    result.ground_y > 0
                    and result.player_y < result.ground_y - 25
                )

            # Conservative jump hint.
            if result.obstacle_distance < 230:
                if result.vertical_velocity >= -1:
                    result.jump_hint = True

            self.last = result
            return result

        except Exception:
            self.last = result
            return result


# ============================================================
# FLYBRAIN MOTOR SYSTEM
# ============================================================

class BrainEngine:
    def __init__(self):
        self.available = FLYBRAIN_AVAILABLE
        self.brain = None
        self.eyes = None

        if not self.available:
            return

        try:
            self.brain = FlyBrain()

            self.eyes = Eyes(self.brain.azimuth)

        except Exception:
            self.available = False
            self.brain = None
            self.eyes = None

    def reset(self):
        if self.brain is None:
            return

        try:
            self.brain.reset()
        except Exception:
            pass

    def process(self, vision: VisionResult):
        if self.brain is None or self.eyes is None:
            return {}

        try:
            darkness = 1.0

            if vision.obstacle_distance < 300:
                darkness = max(
                    0.05,
                    min(
                        1.0,
                        1.0 - vision.obstacle_distance / 300.0,
                    ),
                )

            blob = Blob(
                center=0.0,
                half_width=0.1,
                darkness=darkness,
            )

            drive = self.eyes.drive([blob])

            self.brain.step(eye_drive=drive)

            fired = self.brain.fired

            if hasattr(fired, "get"):
                fired = fired.get()

            fired = np.asarray(fired)

            return {
                "fired": int(len(fired)),
                "neurons": fired,
            }

        except Exception:
            return {}


# ============================================================
# MAIN APPLICATION
# ============================================================

class FlyBrainApp:
    def __init__(self, root):
        self.root = root

        self.username = None
        self.account = None

        self.running = True
        self.ai_enabled = False
        self.emergency_stop = False

        self.mode = "Observe"
        self.ai_complexity = "Simple"

        self.entertainment = False
        self.entertainment_window = None

        self.training = False
        self.training_target = 5000

        self.attempts = 0
        self.deaths = 0
        self.jumps = 0
        self.frames = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        self.last_frame = None
        self.last_vision = VisionResult()
        self.last_brain = {}

        self.vision = GDVision()
        self.brain = BrainEngine()

        self.model = LogisticModel()
        self.model.load()

        self.capture_thread = None
        self.ai_thread = None

        self.lock = threading.Lock()

        self.sct = None

        if mss is not None:
            try:
                self.sct = mss.mss()
            except Exception:
                self.sct = None

        self.build_login()

        self.install_hotkey()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close,
        )

    # ========================================================
    # UI HELPERS
    # ========================================================

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def configure_window(self, width=1050, height=700):
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(800, 550)
        self.root.configure(bg=BG)

    def label(self, parent, text, size=10, bold=False):
        font = ("Segoe UI", size, "bold" if bold else "normal")

        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=TEXT,
            font=font,
        )

    def button(self, parent, text, command, width=18):
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=PANEL2,
            fg=TEXT,
            activebackground="#303642",
            activeforeground=TEXT,
            relief="flat",
            padx=8,
            pady=6,
        )

    # ========================================================
    # LOGIN
    # ========================================================

    def build_login(self):
        self.clear_root()
        self.configure_window(500, 430)

        frame = tk.Frame(self.root, bg=BG)
        frame.pack(fill="both", expand=True, padx=50, pady=40)

        self.label(
            frame,
            "🧠 FlyBrain",
            size=28,
            bold=True,
        ).pack(pady=(20, 5))

        self.label(
            frame,
            "Geometry Dash AI",
            size=12,
        ).pack(pady=(0, 30))

        self.label(frame, "Username").pack(anchor="w")

        self.login_username = tk.Entry(
            frame,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
        )
        self.login_username.pack(fill="x", pady=(5, 15))

        self.label(frame, "Password").pack(anchor="w")

        self.login_password = tk.Entry(
            frame,
            show="*",
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
        )
        self.login_password.pack(fill="x", pady=(5, 20))

        row = tk.Frame(frame, bg=BG)
        row.pack()

        self.button(
            row,
            "Login",
            self.login,
        ).pack(side="left", padx=5)

        self.button(
            row,
            "Create Account",
            self.register,
        ).pack(side="left", padx=5)

        self.label(
            frame,
            "Local account data is stored on this PC.",
            size=9,
        ).pack(pady=25)

    def login(self):
        username = self.login_username.get().strip()
        password = self.login_password.get()

        ok, account = verify_account(
            username,
            password,
        )

        if not ok:
            messagebox.showerror(
                "Login failed",
                "Username or password is incorrect.",
            )
            return

        self.username = username
        self.account = account

        self.start_application()

    def register(self):
        username = self.login_username.get().strip()
        password = self.login_password.get()

        ok, message = create_account(
            username,
            password,
        )

        if ok:
            messagebox.showinfo(
                "Account created",
                "Your account was created. You can now log in.",
            )
        else:
            messagebox.showerror(
                "Could not create account",
                message,
            )

    # ========================================================
    # MAIN APP
    # ========================================================

    def start_application(self):
        self.clear_root()
        self.configure_window(1150, 750)

        self.build_main_ui()

        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            daemon=True,
        )
        self.capture_thread.start()

        self.ai_thread = threading.Thread(
            target=self.ai_loop,
            daemon=True,
        )
        self.ai_thread.start()

        self.update_ui()

    def build_main_ui(self):
        top = tk.Frame(
            self.root,
            bg=PANEL,
            height=60,
        )
        top.pack(fill="x")

        tk.Label(
            top,
            text="🧠 FlyBrain",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 20, "bold"),
        ).pack(side="left", padx=18, pady=12)

        self.status_label = tk.Label(
            top,
            text="● Ready",
            bg=PANEL,
            fg=GREEN,
            font=("Segoe UI", 10, "bold"),
        )
        self.status_label.pack(side="left", padx=15)

        self.f8_label = tk.Label(
            top,
            text="F8 = EMERGENCY STOP",
            bg=PANEL,
            fg=YELLOW,
            font=("Segoe UI", 9, "bold"),
        )
        self.f8_label.pack(side="right", padx=18)

        body = tk.PanedWindow(
            self.root,
            orient="horizontal",
            bg=BG,
            sashwidth=6,
        )
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG)
        right = tk.Frame(body, bg=BG)

        body.add(left, minsize=350)
        body.add(right, minsize=450)

        # ----------------------------------------------
        # LEFT
        # ----------------------------------------------

        mode_panel = tk.LabelFrame(
            left,
            text="Mode",
            bg=PANEL,
            fg=TEXT,
            padx=12,
            pady=12,
        )
        mode_panel.pack(fill="x", padx=12, pady=12)

        self.mode_var = tk.StringVar(value="Observe")

        modes = [
            "Observe",
            "Learn",
            "AI Test",
            "AI Play",
            "Entertainment",
        ]

        self.mode_menu = ttk.Combobox(
            mode_panel,
            textvariable=self.mode_var,
            values=modes,
            state="readonly",
        )
        self.mode_menu.pack(fill="x")
        self.mode_menu.bind(
            "<<ComboboxSelected>>",
            self.change_mode,
        )

        complexity_panel = tk.LabelFrame(
            left,
            text="AI Complexity",
            bg=PANEL,
            fg=TEXT,
            padx=12,
            pady=12,
        )
        complexity_panel.pack(fill="x", padx=12, pady=5)

        self.complexity_var = tk.StringVar(
            value="Simple"
        )

        tk.Radiobutton(
            complexity_panel,
            text="Simple",
            variable=self.complexity_var,
            value="Simple",
            command=self.change_complexity,
            bg=PANEL,
            fg=TEXT,
            selectcolor=PANEL2,
            activebackground=PANEL,
            activeforeground=TEXT,
        ).pack(anchor="w")

        tk.Radiobutton(
            complexity_panel,
            text="Complex",
            variable=self.complexity_var,
            value="Complex",
            command=self.change_complexity,
            bg=PANEL,
            fg=TEXT,
            selectcolor=PANEL2,
            activebackground=PANEL,
            activeforeground=TEXT,
        ).pack(anchor="w")

        training_panel = tk.LabelFrame(
            left,
            text="Training",
            bg=PANEL,
            fg=TEXT,
            padx=12,
            pady=12,
        )
        training_panel.pack(fill="x", padx=12, pady=5)

        self.training_stats = tk.Label(
            training_panel,
            text="Loading...",
            bg=PANEL,
            fg=TEXT,
            justify="left",
            anchor="w",
        )
        self.training_stats.pack(fill="x")

        self.button(
            training_panel,
            "Train Complex AI",
            self.train_model,
            width=24,
        ).pack(pady=(10, 5))

        self.button(
            training_panel,
            "Clear Training Data",
            self.clear_training,
            width=24,
        ).pack(pady=5)

        controls = tk.LabelFrame(
            left,
            text="Controls",
            bg=PANEL,
            fg=TEXT,
            padx=12,
            pady=12,
        )
        controls.pack(fill="x", padx=12, pady=5)

        self.button(
            controls,
            "STOP AI",
            self.stop_ai,
            width=24,
        ).pack(pady=4)

        self.button(
            controls,
            "Vision Debugger",
            self.open_debugger,
            width=24,
        ).pack(pady=4)

        self.button(
            controls,
            "Developer Tools",
            self.open_developer,
            width=24,
        ).pack(pady=4)

        self.button(
            controls,
            "Link Geometry Dash",
            self.link_gd,
            width=24,
        ).pack(pady=4)

        self.button(
            controls,
            "Discord",
            self.link_discord,
            width=24,
        ).pack(pady=4)

        # ----------------------------------------------
        # RIGHT
        # ----------------------------------------------

        stats = tk.LabelFrame(
            right,
            text="Live Status",
            bg=PANEL,
            fg=TEXT,
            padx=15,
            pady=15,
        )
        stats.pack(fill="x", padx=12, pady=12)

        self.live_text = tk.Label(
            stats,
            text="Waiting...",
            bg=PANEL,
            fg=TEXT,
            justify="left",
            anchor="w",
            font=("Consolas", 11),
        )
        self.live_text.pack(fill="x")

        preview = tk.LabelFrame(
            right,
            text="Vision",
            bg=PANEL,
            fg=TEXT,
            padx=10,
            pady=10,
        )
        preview.pack(fill="both", expand=True, padx=12, pady=5)

        self.preview_label = tk.Label(
            preview,
            text="Vision preview",
            bg="#08090c",
            fg=MUTED,
        )
        self.preview_label.pack(
            fill="both",
            expand=True,
        )

        self.bottom_label = tk.Label(
            self.root,
            text=f"Logged in as {self.username}",
            bg=PANEL,
            fg=MUTED,
            anchor="w",
        )
        self.bottom_label.pack(fill="x", padx=10, pady=5)

    # ========================================================
    # MODE
    # ========================================================

    def change_mode(self, event=None):
        self.mode = self.mode_var.get()

        if self.mode == "Entertainment":
            self.start_entertainment()

        else:
            self.stop_entertainment()

        if self.mode in ("AI Play", "Entertainment"):
            self.ai_enabled = True
            self.emergency_stop = False

        else:
            self.ai_enabled = False

    def change_complexity(self):
        self.ai_complexity = self.complexity_var.get()

    # ========================================================
    # CAPTURE
    # ========================================================

    def capture_loop(self):
        while self.running:
            if self.sct is None:
                time.sleep(1)
                continue

            try:
                raw = np.array(
                    self.sct.grab(GD_REGION),
                    dtype=np.uint8,
                )

                frame = np.ascontiguousarray(
                    raw[:, :, :3]
                )

                vision = self.vision.analyze(frame)

                brain = self.brain.process(vision)

                with self.lock:
                    self.last_frame = frame
                    self.last_vision = vision
                    self.last_brain = brain

                    self.frames += 1

                now = time.time()

                if now - self.last_fps_time >= 1:
                    self.fps = self.frames / (
                        now - self.last_fps_time
                    )
                    self.frames = 0
                    self.last_fps_time = now

            except Exception:
                time.sleep(0.02)

    # ========================================================
    # AI LOOP
    # ========================================================

    def ai_loop(self):
        previous_jump = False

        while self.running:
            time.sleep(0.025)

            if not self.ai_enabled:
                continue

            if self.emergency_stop:
                continue

            try:
                with self.lock:
                    vision = self.last_vision

                features = self.features_from_vision(
                    vision
                )

                jump = self.decide_jump(
                    features,
                    vision,
                )

                if self.mode == "Learn":
                    self.record_learning(
                        features,
                        jump,
                    )

                if self.mode in (
                    "AI Play",
                    "Entertainment",
                ):
                    if jump and not previous_jump:
                        self.perform_jump()

                previous_jump = jump

            except Exception:
                pass

    def features_from_vision(self, v: VisionResult):
        distance = min(
            max(v.obstacle_distance, 0),
            1000,
        )

        return [
            v.player_x / SCREEN_WIDTH,
            v.player_y / SCREEN_HEIGHT,
            distance / 1000.0,
            min(v.obstacle_width / 200.0, 1.0),
            max(-1.0, min(1.0, v.horizontal_velocity / 20)),
            max(-1.0, min(1.0, v.vertical_velocity / 20)),
            1.0 if v.airborne else 0.0,
            v.player_confidence,
        ]

    def decide_jump(self, features, vision):
        if self.ai_complexity == "Complex":
            probability = self.model.predict_probability(
                features
            )

            if probability >= 0.70:
                return True

        # Simple fallback.
        if vision.obstacle_distance < 180:
            if not vision.airborne:
                return True

        return False

    def record_learning(self, features, jump):
        add_training_sample(
            features,
            jump,
        )

    # ========================================================
    # INPUT
    # ========================================================

    def perform_jump(self):
        if self.emergency_stop:
            return

        if pyautogui is None:
            return

        try:
            pyautogui.press("space")
            self.jumps += 1
        except Exception:
            pass

    def stop_ai(self):
        self.ai_enabled = False
        self.emergency_stop = True

        self.status_label.config(
            text="● AI STOPPED",
            fg=RED,
        )

    # ========================================================
    # F8
    # ========================================================

    def install_hotkey(self):
        if keyboard is None:
            return

        try:
            keyboard.add_hotkey(
                "f8",
                self.emergency_stop_all,
            )
        except Exception:
            pass

    def emergency_stop_all(self):
        self.emergency_stop = True
        self.ai_enabled = False

        try:
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="● EMERGENCY STOP",
                    fg=RED,
                ),
            )
        except Exception:
            pass

    # ========================================================
    # TRAINING
    # ========================================================

    def train_model(self):
        data = load_dataset()

        if len(data) < 20:
            messagebox.showwarning(
                "Not enough data",
                "Collect at least 20 training samples first.",
            )
            return

        ok = self.model.train(data)

        if ok:
            self.model.save()

            messagebox.showinfo(
                "Training complete",
                f"Trained on {len(data):,} samples.",
            )

    def clear_training(self):
        answer = messagebox.askyesno(
            "Clear training",
            "Delete all local training samples?",
        )

        if not answer:
            return

        try:
            if DATASET_FILE.exists():
                DATASET_FILE.unlink()

            self.model.weights = None
            self.model.bias = 0

            if MODEL_FILE.exists():
                MODEL_FILE.unlink()

        except Exception as e:
            messagebox.showerror(
                "Error",
                str(e),
            )

    # ========================================================
    # ENTERTAINMENT MODE
    # ========================================================

    def start_entertainment(self):
        if self.entertainment_window is not None:
            return

        self.entertainment = True

        self.entertainment_window = tk.Toplevel(
            self.root
        )

        win = self.entertainment_window

        win.title("FlyBrain - Entertainment")

        width = 320
        height = 230

        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()

        x = screen_w - width - 20
        y = screen_h - height - 60

        win.geometry(
            f"{width}x{height}+{x}+{y}"
        )

        win.minsize(260, 180)
        win.attributes("-topmost", True)

        frame = tk.Frame(
            win,
            bg=BG,
        )
        frame.pack(
            fill="both",
            expand=True,
        )

        tk.Label(
            frame,
            text="🧠 FlyBrain",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=(12, 2))

        tk.Label(
            frame,
            text="ENTERTAINMENT MODE",
            bg=BG,
            fg=ACCENT,
            font=("Segoe UI", 9, "bold"),
        ).pack()

        self.ent_status = tk.Label(
            frame,
            text="● PLAYING",
            bg=BG,
            fg=GREEN,
            font=("Segoe UI", 12, "bold"),
        )
        self.ent_status.pack(pady=8)

        self.ent_stats = tk.Label(
            frame,
            text="Starting...",
            bg=BG,
            fg=TEXT,
            justify="left",
        )
        self.ent_stats.pack()

        row = tk.Frame(
            frame,
            bg=BG,
        )
        row.pack(pady=12)

        tk.Button(
            row,
            text="PAUSE",
            command=self.pause_entertainment,
            bg=PANEL2,
            fg=TEXT,
            relief="flat",
            padx=15,
            pady=5,
        ).pack(side="left", padx=5)

        tk.Button(
            row,
            text="STOP",
            command=self.stop_entertainment,
            bg=PANEL2,
            fg=TEXT,
            relief="flat",
            padx=15,
            pady=5,
        ).pack(side="left", padx=5)

        win.protocol(
            "WM_DELETE_WINDOW",
            self.stop_entertainment,
        )

    def pause_entertainment(self):
        self.ai_enabled = not self.ai_enabled

        if self.ai_enabled:
            self.ent_status.config(
                text="● PLAYING",
                fg=GREEN,
            )
        else:
            self.ent_status.config(
                text="● PAUSED",
                fg=YELLOW,
            )

    def stop_entertainment(self):
        self.entertainment = False
        self.ai_enabled = False

        if self.entertainment_window is not None:
            try:
                self.entertainment_window.destroy()
            except Exception:
                pass

            self.entertainment_window = None

    # ========================================================
    # DEBUGGER
    # ========================================================

    def open_debugger(self):
        win = tk.Toplevel(self.root)

        win.title("FlyBrain Vision Debugger")
        win.geometry("600x500")
        win.configure(bg=BG)

        text = tk.Text(
            win,
            bg="#08090c",
            fg=TEXT,
            font=("Consolas", 10),
        )
        text.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )

        def update():
            if not win.winfo_exists():
                return

            v = self.last_vision

            text.delete("1.0", "end")

            text.insert(
                "end",
                "FLYBRAIN VISION DEBUGGER\n"
                "=========================\n\n"
            )

            text.insert(
                "end",
                f"Player X:          {v.player_x:.1f}\n"
                f"Player Y:          {v.player_y:.1f}\n"
                f"Confidence:        {v.player_confidence:.2f}\n\n"
                f"Obstacle X:        {v.obstacle_x:.1f}\n"
                f"Obstacle Y:        {v.obstacle_y:.1f}\n"
                f"Obstacle distance: {v.obstacle_distance:.1f}\n"
                f"Obstacle width:    {v.obstacle_width:.1f}\n\n"
                f"Ground Y:          {v.ground_y:.1f}\n"
                f"Horizontal speed:  {v.horizontal_velocity:.2f}\n"
                f"Vertical speed:    {v.vertical_velocity:.2f}\n"
                f"Airborne:          {v.airborne}\n"
                f"Jump hint:         {v.jump_hint}\n\n"
                f"FPS:               {self.fps:.1f}\n"
                f"Brain fired:       "
                f"{self.last_brain.get('fired', 0)}\n"
            )

            win.after(
                150,
                update,
            )

        update()

    # ========================================================
    # DEVELOPER
    # ========================================================

    def open_developer(self):
        if not self.account:
            return

        if self.account.get("role") != "developer":
            messagebox.showinfo(
                "Developer Mode",
                "Developer tools are not enabled for this account.",
            )
            return

        win = tk.Toplevel(self.root)

        win.title("FlyBrain Developer Tools")
        win.geometry("500x500")
        win.configure(bg=BG)

        frame = tk.Frame(
            win,
            bg=BG,
        )
        frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20,
        )

        self.label(
            frame,
            "Developer Tools",
            size=20,
            bold=True,
        ).pack(pady=10)

        self.button(
            frame,
            "Reset FlyBrain",
            self.reset_brain,
            width=30,
        ).pack(pady=5)

        self.button(
            frame,
            "Reset Vision",
            self.reset_vision,
            width=30,
        ).pack(pady=5)

        self.button(
            frame,
            "Show Dataset Stats",
            self.show_dataset_stats,
            width=30,
        ).pack(pady=5)

        self.button(
            frame,
            "Emergency Stop",
            self.emergency_stop_all,
            width=30,
        ).pack(pady=5)

    def reset_brain(self):
        self.brain.reset()

    def reset_vision(self):
        self.vision.reset()

    def show_dataset_stats(self):
        data = load_dataset()

        jumps = sum(
            1 for item in data
            if item.get("jump") == 1
        )

        no_jumps = len(data) - jumps

        messagebox.showinfo(
            "Dataset",
            f"Total samples: {len(data):,}\n"
            f"Jump: {jumps:,}\n"
            f"No jump: {no_jumps:,}\n"
            f"Recommended target: {self.training_target:,}",
        )

    # ========================================================
    # GEOMETRY DASH LINKING
    # ========================================================

    def link_gd(self):
        win = tk.Toplevel(self.root)

        win.title("Link Geometry Dash")
        win.geometry("480x420")
        win.configure(bg=BG)

        frame = tk.Frame(
            win,
            bg=BG,
        )
        frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=25,
        )

        self.label(
            frame,
            "🎮 Link Geometry Dash",
            size=20,
            bold=True,
        ).pack(pady=10)

        self.label(
            frame,
            "This does not ask for your Geometry Dash password.",
            size=10,
        ).pack(pady=5)

        self.label(
            frame,
            "GD Username",
        ).pack(anchor="w", pady=(20, 3))

        username_entry = tk.Entry(
            frame,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
        )
        username_entry.pack(fill="x")

        code = secrets.token_hex(4).upper()

        self.label(
            frame,
            f"Verification code: {code}",
            size=14,
            bold=True,
        ).pack(pady=20)

        self.label(
            frame,
            "GD does not provide a standard OAuth profile\n"
            "verification endpoint for custom codes.\n\n"
            "Therefore this version stores the requested\n"
            "username/code locally but does not falsely claim\n"
            "that ownership has been verified.",
            size=9,
        ).pack()

        def save_link():
            username = username_entry.get().strip()

            if not username:
                messagebox.showwarning(
                    "Missing username",
                    "Enter your Geometry Dash username.",
                    parent=win,
                )
                return

            settings = load_settings()

            settings["gd_username"] = username
            settings["gd_link_code"] = code
            settings["gd_link_status"] = "pending"

            save_settings(settings)

            messagebox.showinfo(
                "Saved",
                "The Geometry Dash username has been saved locally.\n\n"
                "Ownership is still pending verification.",
                parent=win,
            )

        self.button(
            frame,
            "Save GD Link",
            save_link,
            width=25,
        ).pack(pady=15)

    # ========================================================
    # DISCORD
    # ========================================================

    def link_discord(self):
        win = tk.Toplevel(self.root)

        win.title("Discord")
        win.geometry("460x300")
        win.configure(bg=BG)

        frame = tk.Frame(
            win,
            bg=BG,
        )
        frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=25,
        )

        self.label(
            frame,
            "💬 Discord",
            size=20,
            bold=True,
        ).pack(pady=10)

        self.label(
            frame,
            "Discord linking requires a Discord OAuth2\n"
            "application and redirect configuration.",
            size=10,
        ).pack(pady=10)

        self.label(
            frame,
            "The current app does not store a Discord password.",
            size=9,
        ).pack(pady=5)

        self.button(
            frame,
            "Open Discord Developer Portal",
            self.open_discord_portal,
            width=30,
        ).pack(pady=15)

    def open_discord_portal(self):
        import webbrowser

        webbrowser.open(
            "https://discord.com/developers/applications"
        )

    # ========================================================
    # SETTINGS
    # ========================================================

    def update_ui(self):
        if not self.running:
            return

        try:
            v = self.last_vision

            dataset = load_dataset()

            jumps = sum(
                1 for item in dataset
                if item.get("jump") == 1
            )

            no_jumps = len(dataset) - jumps

            self.training_stats.config(
                text=(
                    f"Samples: {len(dataset):,}\n"
                    f"Jump samples: {jumps:,}\n"
                    f"No-jump samples: {no_jumps:,}\n"
                    f"Target: {self.training_target:,}"
                )
            )

            status = (
                "PLAYING"
                if self.ai_enabled
                else "IDLE"
            )

            self.status_label.config(
                text=f"● {status}",
                fg=(
                    GREEN
                    if self.ai_enabled
                    else MUTED
                ),
            )

            self.live_text.config(
                text=(
                    f"Mode:             {self.mode}\n"
                    f"AI:               {self.ai_complexity}\n"
                    f"FPS:              {self.fps:.1f}\n"
                    f"Player X:         {v.player_x:.1f}\n"
                    f"Player Y:         {v.player_y:.1f}\n"
                    f"Player confidence:{v.player_confidence:.2f}\n"
                    f"Obstacle distance: {v.obstacle_distance:.1f}\n"
                    f"Airborne:         {v.airborne}\n"
                    f"Jumps:            {self.jumps}\n"
                    f"Attempts:         {self.attempts}\n"
                    f"Deaths:           {self.deaths}\n"
                )
            )

            if self.entertainment_window is not None:
                try:
                    self.ent_stats.config(
                        text=(
                            f"AI: {self.ai_complexity}\n"
                            f"FPS: {self.fps:.1f}\n"
                            f"Jumps: {self.jumps}\n"
                            f"Attempts: {self.attempts}\n"
                            f"Deaths: {self.deaths}"
                        )
                    )
                except Exception:
                    pass

        except Exception:
            pass

        self.root.after(
            250,
            self.update_ui,
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        self.running = False
        self.ai_enabled = False
        self.emergency_stop = True

        try:
            if keyboard is not None:
                keyboard.unhook_all_hotkeys()
        except Exception:
            pass

        self.stop_entertainment()

        try:
            if self.sct is not None:
                self.sct.close()
        except Exception:
            pass

        self.root.destroy()


# ============================================================
# SETTINGS HELPERS
# ============================================================

def load_settings():
    if not SETTINGS_FILE.exists():
        return {}

    try:
        with SETTINGS_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


def save_settings(data):
    with SETTINGS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, indent=2)


# ============================================================
# STARTUP
# ============================================================

def main():
    root = tk.Tk()

    root.title("FlyBrain")

    app = FlyBrainApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
