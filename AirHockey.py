"""Air Hockey - OpenCV + MediaPipe. Runs even without camera / model.

Controls:
    M       Mode: AI opponent <-> 2-player
    T       Time limit (cycles; restarts match)
    P       Pause / resume
    H       Help overlay
    D       Debug view (skeleton overlay on camera preview)
    R       Restart match
    Q/ESC   Quit

Camera preview is always visible in the top-right corner when a camera
is available. Debug (D) adds the hand-skeleton overlay on top of it.
"""

import math
import os
import random
import sys
import threading
import time
import traceback

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"FATAL: opencv-python / numpy missing ({e})")
    print("  python -m pip install opencv-python numpy")
    input("Press Enter to exit...")
    sys.exit(1)

HAS_MEDIAPIPE = False
mp = None
mp_vision = None
mp_python = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError as e:
    print(f"[AirHockey] mediapipe unavailable ({e}) - using mouse only")


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
PALM_IDS = (0, 5, 9, 13, 17)


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
CAM_INDEX = 0
CAM_W, CAM_H = 320, 240
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

WIDTH, HEIGHT = 1280, 820
COURT_W = 420
COURT_H = 570
CX = WIDTH / 2.0
CY = 435
LEFT = CX - COURT_W / 2.0
RIGHT = CX + COURT_W / 2.0
TOP = CY - COURT_H / 2.0
BOTTOM = CY + COURT_H / 2.0

PANEL_L_X0, PANEL_L_X1 = 60, 400
PANEL_R_X0, PANEL_R_X1 = 880, 1220

# Small always-on camera preview, top-right corner
CAM_PREV_W, CAM_PREV_H = 180, 130
CAM_PREV_X = WIDTH - CAM_PREV_W - 20     # 1080
CAM_PREV_Y = 12

MALLET_R, PUCK_R = 24, 12
GOAL_HALF = 60

MAX_PLAYER_SPEED = 3000.0
SPEED_AI = 560.0
SPEED_MAX = 1500.0
SPEED_MIN = 220.0
SERVE_SPEED = 750.0
PUSH = 220.0
DAMPING = 0.92
SMOOTH_K = 14.0
MIN_VX_ON_WALL_BOUNCE = 100.0
MAP_LO, MAP_HI = 0.15, 0.85

GOAL_FREEZE_DURATION = 0.8
READY_DELAY = 0.4

TARGET_FPS = 60
FRAME_DT = 1.0 / TARGET_FPS
PHYS_STEP = 1.0 / 120.0

#                   no limit  1 min  2 min  3 min  5 min
MATCH_OPTIONS =     (0,       60,    120,   180,   300)
MATCH_DEFAULT_I = 1                     # 1 minute

BG = (28, 18, 12)
BOARD = (60, 40, 30)
LINE = (200, 200, 200)
GOAL = (255, 200, 0)
PLAYER_C = (60, 220, 60)
AI_C = (220, 60, 60)
PUCK_C = (240, 240, 240)
WARN_C = (60, 200, 255)
OK_C = (60, 220, 60)
ERR_C = (60, 60, 220)
RED_C = (0, 0, 255)
DIM = (120, 120, 120)
HELP_BG = (18, 12, 8)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_DUP = cv2.FONT_HERSHEY_DUPLEX
WIN = "Air Hockey"


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# --------------------------------------------------------------------------- #
#  Model download (optional)
# --------------------------------------------------------------------------- #
MODEL_OK = False
if HAS_MEDIAPIPE:
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[AirHockey] downloading {MODEL_PATH} (~8 MB, one time)...")
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        try:
            _test = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            _test.close()
            MODEL_OK = True
            print("[AirHockey] hand model ready.")
        except Exception as e:
            print(f"[AirHockey] model load failed ({e}) - using mouse only")
    except Exception as e:
        print(f"[AirHockey] model download failed ({e}) - using mouse only")


def make_landmarker(num_hands):
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


# --------------------------------------------------------------------------- #
#  Camera (optional)
# --------------------------------------------------------------------------- #
def try_open_camera():
    try:
        c = cv2.VideoCapture(CAM_INDEX)
        if not c.isOpened():
            c.release()
            print(f"[AirHockey] camera {CAM_INDEX} not available - using mouse")
            return None
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        c.set(cv2.CAP_PROP_FPS, 30)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, _ = c.read()
        if not ret:
            c.release()
            print(f"[AirHockey] camera {CAM_INDEX} opened but no frames - using mouse")
            return None
        return c
    except Exception as e:
        print(f"[AirHockey] camera error ({e}) - using mouse")
        return None


cap = try_open_camera()
TRACKER_ACTIVE = HAS_MEDIAPIPE and MODEL_OK and cap is not None

stop_ev = threading.Event()
debug_ev = threading.Event()
shared = {"data": ([], None, 0.0)}
want_hands = {"n": 1}
tracker = None


def tracker_loop(landmarker, cur_hands):
    start_t = time.perf_counter()
    last_ts = -1
    fails = 0
    n, t0, tfps = 0, start_t, 0.0
    try:
        while not stop_ev.is_set():
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 100:
                    break
                time.sleep(0.01)
                continue
            fails = 0

            wanted = want_hands["n"]
            if wanted != cur_hands:
                new_lm = make_landmarker(wanted)
                landmarker.close()
                landmarker, cur_hands = new_lm, wanted

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.perf_counter() - start_t) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            want_dbg = debug_ev.is_set()
            hands = []
            for lm in result.hand_landmarks:
                nx = 1.0 - sum(lm[i].x for i in PALM_IDS) / len(PALM_IDS)
                ny = sum(lm[i].y for i in PALM_IDS) / len(PALM_IDS)
                pts = [(1.0 - p.x, p.y) for p in lm] if want_dbg else None
                hands.append((nx, ny, pts))
            hands.sort(key=lambda h: h[0])

            n += 1
            now = time.perf_counter()
            if now - t0 >= 0.5:
                tfps = n / (now - t0)
                n, t0 = 0, now

            # Frame is ALWAYS delivered (needed for the always-on preview)
            shared["data"] = (hands, frame, tfps)
    except Exception as e:
        print(f"[AirHockey] tracker thread stopped: {e}")


if TRACKER_ACTIVE:
    try:
        _init_lm = make_landmarker(1)
        tracker = threading.Thread(target=tracker_loop,
                                   args=(_init_lm, 1), daemon=True)
        tracker.start()
        print("[AirHockey] hand tracking active.")
    except Exception as e:
        print(f"[AirHockey] could not start tracker ({e}) - using mouse")
        tracker = None
        TRACKER_ACTIVE = False
else:
    print("[AirHockey] running in mouse-only mode.")


# --------------------------------------------------------------------------- #
#  Hand -> game mapping
# --------------------------------------------------------------------------- #
def hand_to_game(nx, ny, bottom_half):
    tx = clamp((nx - MAP_LO) / (MAP_HI - MAP_LO), 0.0, 1.0)
    ty = clamp((ny - MAP_LO) / (MAP_HI - MAP_LO), 0.0, 1.0)
    gx = (LEFT + MALLET_R) + tx * ((RIGHT - MALLET_R) - (LEFT + MALLET_R))
    if bottom_half:
        y0, y1 = CY + MALLET_R, BOTTOM - MALLET_R
    else:
        y0, y1 = TOP + MALLET_R, CY - MALLET_R
    gy = y0 + ty * (y1 - y0)
    return gx, gy


# --------------------------------------------------------------------------- #
#  Game
# --------------------------------------------------------------------------- #
class AirHockey:
    def __init__(self):
        self.ai_mode = True
        self.paused = False
        self.help_open = False
        self.match_idx = MATCH_DEFAULT_I
        self.match_len = MATCH_OPTIONS[self.match_idx]
        self.elapsed_accum = 0.0
        self.player = [CX, BOTTOM - 120.0]
        self.opponent = [CX, TOP + 120.0]
        self.puck = [CX, CY]
        self.velocity = [0.0, 0.0]
        self.smooth_player = [self.player[0], self.player[1]]
        self.smooth_opponent = [self.opponent[0], self.opponent[1]]
        self.player_target = (self.player[0], self.player[1])
        self.opponent_target = (self.opponent[0], self.opponent[1])
        self.player_target_from_hand = False
        self.opponent_target_from_hand = False
        self.mouse_x = self.player[0]
        self.mouse_y = self.player[1]
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.end_reason = ""
        self.winner = ""
        self.goal_freeze = 0.0
        self.goal_msg = ""
        self.goal_col = OK_C
        self.pending_game_over = False
        self.pending_winner = ""
        self.pending_reason = ""
        self.ready_timer = READY_DELAY
        self.acc = 0.0
        self._reset_positions()
        self.puck[0], self.puck[1] = CX, CY
        self.velocity[0] = self.velocity[1] = 0.0

    def _reset_positions(self):
        self.player[0], self.player[1] = CX, BOTTOM - 120.0
        self.opponent[0], self.opponent[1] = CX, TOP + 120.0
        self.smooth_player[0], self.smooth_player[1] = self.player
        self.smooth_opponent[0], self.smooth_opponent[1] = self.opponent

    def _serve(self, direction):
        self.puck[0], self.puck[1] = CX, CY
        angle = random.uniform(-0.55, 0.55)
        speed = random.uniform(0.75, 1.0) * SERVE_SPEED
        self.velocity[1] = direction * math.cos(angle) * speed
        self.velocity[0] = math.sin(angle) * speed

    def restart(self):
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.end_reason = ""
        self.winner = ""
        self.elapsed_accum = 0.0
        self.acc = 0.0
        self.goal_freeze = 0.0
        self.goal_msg = ""
        self.pending_game_over = False
        self.ready_timer = READY_DELAY
        self._reset_positions()
        self.puck[0], self.puck[1] = CX, CY
        self.velocity[0] = self.velocity[1] = 0.0

    def cycle_match_length(self):
        self.match_idx = (self.match_idx + 1) % len(MATCH_OPTIONS)
        self.match_len = MATCH_OPTIONS[self.match_idx]
        self.restart()

    def _trigger_time_up(self):
        self.game_over = True
        self.end_reason = "time"
        if self.player_score > self.ai_score:
            self.winner = "YOU WIN!" if self.ai_mode else "P1 WINS!"
        elif self.ai_score > self.player_score:
            self.winner = "AI WINS" if self.ai_mode else "P2 WINS!"
        else:
            self.winner = "DRAW"

    def update_elapsed(self, dt):
        if self.paused or self.help_open or self.game_over:
            return
        if self.ready_timer > 0:
            return
        self.elapsed_accum += dt
        if self.match_len > 0 and self.elapsed_accum >= self.match_len:
            self.elapsed_accum = float(self.match_len)
            self._trigger_time_up()

    def _on_goal(self, scorer):
        if scorer == "ai":
            self.ai_score += 1
            self.goal_msg = "AI SCORED!" if self.ai_mode else "P2 SCORED!"
            self.goal_col = AI_C
            if self.ai_score >= 5:
                self.pending_game_over = True
                self.pending_winner = "AI WINS" if self.ai_mode else "P2 WINS!"
                self.pending_reason = "score"
        else:
            self.player_score += 1
            self.goal_msg = "GOAL!" if self.ai_mode else "P1 SCORED!"
            self.goal_col = PLAYER_C
            if self.player_score >= 5:
                self.pending_game_over = True
                self.pending_winner = "YOU WIN!" if self.ai_mode else "P1 WINS!"
                self.pending_reason = "score"
        self.goal_freeze = GOAL_FREEZE_DURATION
        self._reset_positions()

    def _end_freeze(self):
        self.goal_freeze = 0.0
        self.goal_msg = ""
        if self.pending_game_over:
            self.game_over = True
            self.winner = self.pending_winner
            self.end_reason = self.pending_reason
            self.pending_game_over = False
        else:
            self._serve(random.choice([-1, 1]))

    def compute_targets(self, hands):
        self.player_target_from_hand = False
        self.opponent_target_from_hand = False

        if self.ai_mode:
            if hands:
                nx, ny, _ = hands[0]
                self.player_target = hand_to_game(nx, ny, bottom_half=True)
                self.player_target_from_hand = True
            else:
                self.player_target = (
                    clamp(self.mouse_x, LEFT + MALLET_R, RIGHT - MALLET_R),
                    clamp(self.mouse_y, CY + MALLET_R, BOTTOM - MALLET_R),
                )
        else:
            if len(hands) >= 2:
                hs = sorted(hands, key=lambda h: h[1])
                top_nx, top_ny, _ = hs[0]
                bot_nx, bot_ny, _ = hs[-1]
                self.opponent_target = hand_to_game(top_nx, top_ny, bottom_half=False)
                self.player_target = hand_to_game(bot_nx, bot_ny, bottom_half=True)
                self.opponent_target_from_hand = True
                self.player_target_from_hand = True
            elif len(hands) == 1:
                nx, ny, _ = hands[0]
                if ny < 0.5:
                    self.opponent_target = hand_to_game(nx, ny, bottom_half=False)
                    self.opponent_target_from_hand = True
                else:
                    self.player_target = hand_to_game(nx, ny, bottom_half=True)
                    self.player_target_from_hand = True

        if self.ai_mode:
            if self.velocity[1] < 0:
                tx = self.puck[0] + self.velocity[0] * 0.12 + random.uniform(-30, 30)
                ty = self.puck[1] + self.velocity[1] * 0.12
            else:
                tx = CX + (self.puck[0] - CX) * 0.35
                ty = TOP + 120.0
            self.opponent_target = (tx, ty)

    def update(self, dt):
        prev_px, prev_py = self.player[0], self.player[1]
        prev_ox, prev_oy = self.opponent[0], self.opponent[1]
        k = 1.0 - math.exp(-SMOOTH_K * dt)

        if self.player_target_from_hand:
            self.smooth_player[0] += (self.player_target[0] - self.smooth_player[0]) * k
            self.smooth_player[1] += (self.player_target[1] - self.smooth_player[1]) * k
        else:
            self.smooth_player[0] = self.player_target[0]
            self.smooth_player[1] = self.player_target[1]
        dx = self.smooth_player[0] - self.player[0]
        dy = self.smooth_player[1] - self.player[1]
        d = math.hypot(dx, dy)
        if d > 1e-4:
            step = min(d, MAX_PLAYER_SPEED * dt)
            self.player[0] += dx / d * step
            self.player[1] += dy / d * step
        self.player[0] = clamp(self.player[0], LEFT + MALLET_R, RIGHT - MALLET_R)
        self.player[1] = clamp(self.player[1], CY + MALLET_R, BOTTOM - MALLET_R)

        if self.ai_mode:
            tx = clamp(self.opponent_target[0], LEFT + MALLET_R, RIGHT - MALLET_R)
            ty = clamp(self.opponent_target[1], TOP + MALLET_R, CY - MALLET_R)
            ax, ay = tx - self.opponent[0], ty - self.opponent[1]
            d_ai = math.hypot(ax, ay)
            if d_ai > 0.5:
                step = min(SPEED_AI * dt, d_ai)
                self.opponent[0] += ax / d_ai * step
                self.opponent[1] += ay / d_ai * step
        else:
            if self.opponent_target_from_hand:
                self.smooth_opponent[0] += (self.opponent_target[0] - self.smooth_opponent[0]) * k
                self.smooth_opponent[1] += (self.opponent_target[1] - self.smooth_opponent[1]) * k
            ox = self.smooth_opponent[0] - self.opponent[0]
            oy = self.smooth_opponent[1] - self.opponent[1]
            d = math.hypot(ox, oy)
            if d > 1e-4:
                step = min(d, MAX_PLAYER_SPEED * dt)
                self.opponent[0] += ox / d * step
                self.opponent[1] += oy / d * step
            self.opponent[0] = clamp(self.opponent[0], LEFT + MALLET_R, RIGHT - MALLET_R)
            self.opponent[1] = clamp(self.opponent[1], TOP + MALLET_R, CY - MALLET_R)

        if self.ready_timer > 0:
            self.ready_timer -= dt
            if self.ready_timer <= 0:
                self.ready_timer = 0
                self._serve(random.choice([-1, 1]))
            return

        if self.goal_freeze > 0:
            self.goal_freeze -= dt
            if self.goal_freeze <= 0:
                self._end_freeze()
            return

        self.puck[0] += self.velocity[0] * dt
        self.puck[1] += self.velocity[1] * dt
        damp = DAMPING ** dt
        self.velocity[0] *= damp
        self.velocity[1] *= damp

        if self.puck[0] < LEFT + PUCK_R:
            self.puck[0] = LEFT + PUCK_R
            self.velocity[0] = abs(self.velocity[0])
        elif self.puck[0] > RIGHT - PUCK_R:
            self.puck[0] = RIGHT - PUCK_R
            self.velocity[0] = -abs(self.velocity[0])

        in_goal = abs(self.puck[0] - CX) <= GOAL_HALF
        if self.puck[1] < TOP + PUCK_R and not in_goal:
            self.puck[1] = TOP + PUCK_R
            self.velocity[1] = abs(self.velocity[1])
            self._break_wall_bounce()
        elif self.puck[1] > BOTTOM - PUCK_R and not in_goal:
            self.puck[1] = BOTTOM - PUCK_R
            self.velocity[1] = -abs(self.velocity[1])
            self._break_wall_bounce()

        r_sum = MALLET_R + PUCK_R
        p_mvx = (self.player[0] - prev_px) / max(dt, 1e-4)
        p_mvy = (self.player[1] - prev_py) / max(dt, 1e-4)
        o_mvx = (self.opponent[0] - prev_ox) / max(dt, 1e-4)
        o_mvy = (self.opponent[1] - prev_oy) / max(dt, 1e-4)
        for mallet, mvx, mvy in ((self.player, p_mvx, p_mvy),
                                 (self.opponent, o_mvx, o_mvy)):
            ox = self.puck[0] - mallet[0]
            oy = self.puck[1] - mallet[1]
            d = math.hypot(ox, oy)
            if d < 1e-4:
                nx, ny = 1.0, 0.0
            else:
                nx, ny = ox / d, oy / d
            if d < r_sum:
                self.puck[0] = mallet[0] + nx * r_sum
                self.puck[1] = mallet[1] + ny * r_sum
                dot = self.velocity[0] * nx + self.velocity[1] * ny
                if dot < 0:
                    self.velocity[0] -= 2 * dot * nx
                    self.velocity[1] -= 2 * dot * ny
                self.velocity[0] += nx * PUSH * dt
                self.velocity[1] += ny * PUSH * dt
                mdot = mvx * nx + mvy * ny
                if mdot > 0:
                    boost = min(mdot * 0.4, SPEED_MAX * 0.5)
                    self.velocity[0] += nx * boost
                    self.velocity[1] += ny * boost

        speed = math.hypot(self.velocity[0], self.velocity[1])
        if speed > SPEED_MAX:
            kk = SPEED_MAX / speed
            self.velocity[0] *= kk
            self.velocity[1] *= kk
        elif 0 < speed < SPEED_MIN:
            kk = 1.0 + 1.2 * dt
            self.velocity[0] *= kk
            self.velocity[1] *= kk

        in_goal = abs(self.puck[0] - CX) <= GOAL_HALF
        if self.puck[1] > BOTTOM and in_goal:
            self._on_goal("ai")
        elif self.puck[1] < TOP and in_goal:
            self._on_goal("player")

    def _break_wall_bounce(self):
        if abs(self.velocity[0]) < MIN_VX_ON_WALL_BOUNCE:
            sign = 1 if random.random() < 0.5 else -1
            self.velocity[0] = sign * (
                MIN_VX_ON_WALL_BOUNCE + random.uniform(0.0, 80.0))

    # ------------------------------------------------------------------ #
    #  Drawing
    # ------------------------------------------------------------------ #
    def _center(self, canvas, text, cy, scale, color, thick, font=None):
        font = font or FONT
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(canvas, text, (int((WIDTH - w) // 2), int(cy + h // 2)),
                    font, scale, color, thick, cv2.LINE_AA)

    def _draw_court(self, canvas):
        cv2.rectangle(canvas, (int(LEFT), int(TOP)),
                      (int(RIGHT), int(BOTTOM)), BOARD, -1)
        cv2.rectangle(canvas, (int(LEFT), int(TOP)),
                      (int(RIGHT), int(BOTTOM)), LINE, 2)
        cyi = int(CY)
        x = int(LEFT) + 10
        while x < int(RIGHT) - 10:
            cv2.line(canvas, (x, cyi),
                     (min(x + 12, int(RIGHT) - 10), cyi), LINE, 2)
            x += 24
        cv2.circle(canvas, (int(CX), cyi), 70, LINE, 2)
        cv2.rectangle(canvas, (int(CX - GOAL_HALF), int(TOP) - 6),
                      (int(CX + GOAL_HALF), int(TOP) + 4), GOAL, 3)
        cv2.rectangle(canvas, (int(CX - GOAL_HALF), int(BOTTOM) - 4),
                      (int(CX + GOAL_HALF), int(BOTTOM) + 6), GOAL, 3)

    def _draw_header(self, canvas):
        title = "AIR HOCKEY"
        (tw, _), _ = cv2.getTextSize(title, FONT_DUP, 1.0, 2)
        cv2.putText(canvas, title, (int(CX - tw // 2), 50),
                    FONT_DUP, 1.0, LINE, 2, cv2.LINE_AA)
        rule = "FIRST TO 5"
        (rw, _), _ = cv2.getTextSize(rule, FONT, 0.55, 1)
        cv2.putText(canvas, rule, (int(CX - rw // 2), 76),
                    FONT, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        if self.match_len > 0:
            show = max(0.0, self.match_len - self.elapsed_accum)
            timer_col = RED_C if show <= 10.0 else LINE
        else:
            show = self.elapsed_accum
            timer_col = LINE
        mm = int(show // 60)
        ss = show - mm * 60
        timer_txt = f"{mm:02d}:{ss:04.1f}"
        (tw, _), _ = cv2.getTextSize(timer_txt, FONT_DUP, 1.2, 2)
        cv2.putText(canvas, timer_txt, (int(CX - tw // 2), 118),
                    FONT_DUP, 1.2, timer_col, 2, cv2.LINE_AA)

    def _draw_panel(self, canvas, x0, x1, label, score, color):
        cx = (x0 + x1) // 2
        cv2.rectangle(canvas, (x0, int(TOP)), (x1, int(BOTTOM)),
                      (30, 22, 16), -1)
        cv2.rectangle(canvas, (x0, int(TOP)), (x1, int(BOTTOM)),
                      (80, 80, 80), 1)
        (lw, _), _ = cv2.getTextSize(label, FONT_DUP, 1.0, 2)
        cv2.putText(canvas, label, (cx - lw // 2, int(TOP) + 60),
                    FONT_DUP, 1.0, color, 2, cv2.LINE_AA)
        st = str(score)
        (sw, sh), _ = cv2.getTextSize(st, FONT_DUP, 5.0, 8)
        cv2.putText(canvas, st, (cx - sw // 2, int(CY) + sh // 2),
                    FONT_DUP, 5.0, color, 8, cv2.LINE_AA)
        ot = "of 5"
        (ow, _), _ = cv2.getTextSize(ot, FONT, 0.7, 1)
        cv2.putText(canvas, ot, (cx - ow // 2, int(CY) + 110),
                    FONT, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    def _draw_mallets_and_puck(self, canvas):
        for pos, col in ((self.player, PLAYER_C), (self.opponent, AI_C)):
            p = (int(pos[0]), int(pos[1]))
            cv2.circle(canvas, (p[0] + 3, p[1] + 3), MALLET_R,
                       (8, 6, 4), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, MALLET_R, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, MALLET_R, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, p, MALLET_R // 3, (255, 255, 255), -1, cv2.LINE_AA)
        if self.goal_freeze <= 0 and self.ready_timer <= 0:
            pp = (int(self.puck[0]), int(self.puck[1]))
            cv2.circle(canvas, (pp[0] + 2, pp[1] + 2), PUCK_R,
                       (8, 6, 4), -1, cv2.LINE_AA)
            cv2.circle(canvas, pp, PUCK_R, PUCK_C, -1, cv2.LINE_AA)
            cv2.circle(canvas, pp, PUCK_R, (20, 20, 20), 1, cv2.LINE_AA)

    def _draw_hud(self, canvas, hands, tfps, fps, debug):
        y_strip = HEIGHT - 62
        cv2.line(canvas, (40, y_strip - 25), (WIDTH - 40, y_strip - 25),
                 (60, 60, 60), 1)
        hint = ("M: mode    T: time limit    P: pause    H: help    "
                "D: debug    R: restart    Q/ESC: quit")
        (hw, _), _ = cv2.getTextSize(hint, FONT, 0.62, 1)
        cv2.putText(canvas, hint, ((WIDTH - hw) // 2, y_strip),
                    FONT, 0.62, (210, 210, 210), 1, cv2.LINE_AA)

        y_stat = HEIGHT - 20
        if not TRACKER_ACTIVE:
            dot_col = DIM
            hand_txt = "MOUSE"
        elif hands:
            dot_col = OK_C
            hand_txt = f"HAND ({len(hands)})"
        else:
            dot_col = WARN_C
            hand_txt = "NO HAND"
        cv2.circle(canvas, (46, y_stat - 5), 6, dot_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, hand_txt, (62, y_stat),
                    FONT, 0.55, dot_col, 1, cv2.LINE_AA)

        badges = [("AI" if self.ai_mode else "2P", LINE),
                  ("No limit" if not self.match_len
                   else f"{self.match_len // 60} min", LINE)]
        if self.ready_timer > 0:
            badges.append(("READY", WARN_C))
        elif self.goal_freeze > 0:
            badges.append(("GOAL", WARN_C))
        if debug:
            badges.append(("DEBUG", WARN_C))
        if self.help_open:
            badges.append(("HELP", WARN_C))
        if self.paused:
            badges.append(("PAUSED", WARN_C))

        x = WIDTH - 40
        for label, col in reversed(badges):
            (bw, _), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            x -= bw
            cv2.putText(canvas, label, (x, y_stat),
                        FONT, 0.55, col, 1, cv2.LINE_AA)
            x -= 20

    def _draw_cam_preview(self, canvas, cam_frame, hands, tfps, fps, debug):
        """Always-on small camera window in the top-right corner."""
        x, y = CAM_PREV_X, CAM_PREV_Y
        w, h = CAM_PREV_W, CAM_PREV_H

        if cam_frame is not None:
            small = cv2.resize(cv2.flip(cam_frame, 1), (w, h))
            canvas[y:y+h, x:x+w] = small

            for (nx, ny, pts) in hands:
                # Skeleton in debug only
                if debug and pts is not None:
                    for (sx, sy) in pts:
                        px = x + int(sx * w)
                        py = y + int(sy * h)
                        if x <= px < x + w and y <= py < y + h:
                            cv2.circle(canvas, (px, py), 2, WARN_C, -1)
                # Palm marker always
                px = x + int(nx * w)
                py = y + int(ny * h)
                if x <= px < x + w and y <= py < y + h:
                    cv2.circle(canvas, (px, py), 5, OK_C, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

            # "LIVE" indicator
            cv2.circle(canvas, (x + 8, y + 8), 4, (0, 0, 255), -1)
            cv2.putText(canvas, "LIVE", (x + 16, y + 13),
                        FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            if debug:
                cv2.putText(canvas, f"g{fps:.0f}/t{tfps:.0f}",
                            (x, y + h + 14),
                            FONT, 0.42, DIM, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), -1)
            msg = "MOUSE MODE" if not TRACKER_ACTIVE else "waiting"
            (mw, _), _ = cv2.getTextSize(msg, FONT, 0.45, 1)
            cv2.putText(canvas, msg, (x + (w - mw) // 2, y + h // 2 + 4),
                        FONT, 0.45, DIM, 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), LINE, 1)

    def _draw_overlays(self, canvas):
        if self.help_open:
            self._draw_help(canvas)
            return
        if self.paused:
            self._center(canvas, "PAUSED", int(CY) - 10, 2.2,
                         (255, 255, 255), 4)
            self._center(canvas, "P: resume    H: help    Q: quit",
                         int(CY) + 50, 0.7, (180, 180, 180), 2)
        elif self.game_over:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
            canvas[:] = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
            if self.end_reason == "time":
                self._center(canvas, "TIME UP", int(CY) - 70, 2.0, WARN_C, 3)
            self._center(canvas, self.winner, int(CY) - 10, 2.2,
                         (0, 230, 255), 3)
            sub = (f"{self.player_score} - {self.ai_score}     "
                   "R restart     Q quit")
            self._center(canvas, sub, int(CY) + 60, 0.8,
                         (220, 220, 220), 2)
            return
        if self.ready_timer > 0:
            self._center(canvas, "READY!", int(CY) - 10, 2.0, WARN_C, 3)
            self._center(canvas, "Position your mallet",
                         int(CY) + 40, 0.7, DIM, 2)
        elif self.goal_freeze > 0 and self.goal_msg:
            self._center(canvas, self.goal_msg, int(CY) - 10, 2.0,
                         self.goal_col, 3)

    def _draw_help(self, canvas):
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        canvas[:] = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)
        bx0, by0, bx1, by1 = 220, 140, WIDTH - 220, HEIGHT - 160
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), HELP_BG, -1)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (100, 100, 100), 2)
        self._center(canvas, "CONTROLS", by0 + 40, 1.4, WARN_C, 3)
        rows = [
            ("M",       "Mode:  AI opponent  <->  2-player"),
            ("T",       "Time limit  (no limit / 1 / 2 / 3 / 5 min)"),
            ("P",       "Pause / resume"),
            ("D",       "Debug view  (hand skeleton on camera)"),
            ("H",       "Show / hide this help"),
            ("R",       "Restart match"),
            ("Q / ESC", "Quit"),
        ]
        y = by0 + 90
        for key, desc in rows:
            cv2.putText(canvas, key, (bx0 + 40, y),
                        FONT_DUP, 0.75, PLAYER_C, 2, cv2.LINE_AA)
            cv2.putText(canvas, desc, (bx0 + 170, y),
                        FONT, 0.7, LINE, 1, cv2.LINE_AA)
            y += 38
        y += 10
        cv2.putText(canvas, "PLAY", (bx0 + 40, y),
                    FONT_DUP, 0.75, WARN_C, 2, cv2.LINE_AA)
        y += 30
        tips = [
            "You defend the bottom goal; AI defends the top.",
            "Show palm ~40 cm from camera to steer the mallet.",
            "Camera preview is always visible in the top-right.",
            "No hand / no camera? Use the mouse - it always works.",
        ]
        for tip in tips:
            cv2.putText(canvas, "- " + tip, (bx0 + 50, y),
                        FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 26
        self._center(canvas, "H or P to close", by1 - 30, 0.7,
                     (180, 180, 180), 2)

    def draw(self, canvas, cam_frame, hands, tfps, fps, debug):
        canvas[:] = BG
        self._draw_court(canvas)
        self._draw_mallets_and_puck(canvas)
        self._draw_header(canvas)
        you_label = "YOU" if self.ai_mode else "P1"
        ai_label = "AI" if self.ai_mode else "P2"
        self._draw_panel(canvas, PANEL_L_X0, PANEL_L_X1,
                         you_label, self.player_score, PLAYER_C)
        self._draw_panel(canvas, PANEL_R_X0, PANEL_R_X1,
                         ai_label, self.ai_score, AI_C)
        self._draw_hud(canvas, hands, tfps, fps, debug)
        self._draw_cam_preview(canvas, cam_frame, hands, tfps, fps, debug)
        self._draw_overlays(canvas)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def run():
    game = AirHockey()

    def on_mouse(event, x, y, flags, param):
        if event in (cv2.EVENT_MOUSEMOVE,
                     cv2.EVENT_LBUTTONDOWN,
                     cv2.EVENT_LBUTTONUP):
            game.mouse_x = float(x)
            game.mouse_y = float(y)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WIDTH, HEIGHT)
    cv2.setMouseCallback(WIN, on_mouse)

    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    t_prev = time.perf_counter()
    fps_t, fps_n, fps = t_prev, 0, 0.0

    try:
        while not stop_ev.is_set():
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0.05:
                dt = 0.05
            elif dt <= 0.0:
                dt = FRAME_DT

            hands, cam_frame, tfps = shared["data"]
            game.compute_targets(hands)

            physics_on = (not game.paused and not game.help_open
                          and not game.game_over)
            if physics_on:
                game.acc += dt
                steps = 0
                while game.acc >= PHYS_STEP and steps < 24:
                    game.update(PHYS_STEP)
                    game.acc -= PHYS_STEP
                    steps += 1
                if steps >= 24:
                    game.acc = 0.0
                game.update_elapsed(dt)

            fps_n += 1
            if now - fps_t >= 0.5:
                fps = fps_n / (now - fps_t)
                fps_n, fps_t = 0, now

            debug_on = debug_ev.is_set()
            game.draw(canvas, cam_frame, hands, tfps, fps, debug_on)
            cv2.imshow(WIN, canvas)

            wait = max(1, int((FRAME_DT - (time.perf_counter() - now)) * 1000))
            key = cv2.waitKey(wait) & 0xFF

            if (key == ord('q') or key == 27
                    or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1):
                break
            elif key == ord('m'):
                game.ai_mode = not game.ai_mode
                want_hands["n"] = 1 if game.ai_mode else 2
            elif key == ord('t'):
                game.cycle_match_length()
            elif key == ord('p'):
                if game.help_open:
                    game.help_open = False
                    game.paused = False
                else:
                    game.paused = not game.paused
            elif key == ord('h'):
                if game.help_open:
                    game.help_open = False
                    game.paused = False
                else:
                    game.help_open = True
                    game.paused = True
            elif key == ord('r'):
                game.restart()
            elif key == ord('d'):
                if debug_ev.is_set():
                    debug_ev.clear()
                else:
                    debug_ev.set()
    finally:
        stop_ev.set()
        if tracker is not None:
            tracker.join(timeout=2.0)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("\n=== FATAL ===")
        traceback.print_exc()
        print(f"\n{exc}")
        try:
            input("Press Enter to close...")
        except Exception:
            pass