import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="AI Gym Trainer PRO", layout="wide")

USER_WEIGHT = 65

# ─────────────────────────────────────────────
#  ANGLE HELPERS
# ─────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def smooth(angle, buf, window=5):
    buf.append(angle)
    if len(buf) > window:
        buf.pop(0)
    return sum(buf) / len(buf)


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────
def neon_text(img, text, pos, color, scale=0.65, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0),       thickness+3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness+1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,          thickness)


def solid_dashboard(img, x1, y1, x2, y2, bg=(20,20,20), border=(60,60,60)):
    cv2.rectangle(img, (x1,y1), (x2,y2), bg, -1)
    cv2.rectangle(img, (x1,y1), (x2,y2), border, 1)


def warning_bottom(img, message):
    """
    Red warning banner drawn at the BOTTOM of the frame,
    just above the dashboard panel. Solid fill — no addWeighted.
    """
    h, w = img.shape[:2]
    DASH_H    = 80   # height of dashboard at top
    WARN_H    = 44   # height of warning strip
    # Place it just below the top dashboard
    y1 = DASH_H + 4
    y2 = y1 + WARN_H
    cv2.rectangle(img, (0, y1), (w, y2), (0, 0, 150), -1)
    cv2.rectangle(img, (0, y1), (w, y2), (0, 0, 220), 1)
    cv2.putText(img, f"!  {message}", (10, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)


def draw_line(img, lm, a, b, color, w, h, thick=2):
    x1, y1 = int(lm[a].x*w), int(lm[a].y*h)
    x2, y2 = int(lm[b].x*w), int(lm[b].y*h)
    cv2.line(img, (x1,y1), (x2,y2), color, thick, cv2.LINE_AA)


EXERCISES = ["Dumbbell Curl", "Push-Up", "Squat"]


# ─────────────────────────────────────────────
#  VIDEO PROCESSOR
# ─────────────────────────────────────────────
class PoseProcessor(VideoProcessorBase):

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )
        self.pose     = vision.PoseLandmarker.create_from_options(options)
        self.exercise = "Dumbbell Curl"

        # ── Curl ──
        self.left_counter     = 0
        self.right_counter    = 0
        self.left_stage       = None
        self.right_stage      = None
        self.left_angle_buf   = []
        self.right_angle_buf  = []
        self.left_angle_hist  = []   # (ts, angle) rolling 2-s window
        self.right_angle_hist = []
        self.left_warn_until  = 0.0
        self.right_warn_until = 0.0

        # ── Push-Up ──
        self.pushup_counter         = 0
        self.pushup_stage           = None
        self.pushup_elbow_buf       = []
        self.pushup_back_bad_frames = 0
        self.pushup_back_warn_until = 0.0

        # ── Squat ──
        self.squat_counter         = 0
        self.squat_stage           = None
        self.squat_lk_buf          = []
        self.squat_rk_buf          = []
        # spine: only checked at bottom of squat — no sustained counter needed
        self.squat_back_warn_until = 0.0
        # knee valgus: sustained counter kept (it's a valid mid-movement check)
        self.squat_knee_bad_frames = 0
        self.squat_knee_warn_until = 0.0

        self.start_time        = time.time()
        self.last_process_time = 0.0

    # ─────────────────────────────────────────
    #  CURL
    # ─────────────────────────────────────────
    def process_curl(self, img, lm, h, w, ts):
        SPEED_THRESH = 250
        WINDOW_SECS  = 2.0
        WARN_HOLD    = 1.5

        ls  = [lm[11].x, lm[11].y]; le  = [lm[13].x, lm[13].y]; lw_ = [lm[15].x, lm[15].y]
        rs  = [lm[12].x, lm[12].y]; re  = [lm[14].x, lm[14].y]; rw_ = [lm[16].x, lm[16].y]
        la  = smooth(calculate_angle(ls, le, lw_), self.left_angle_buf)
        ra  = smooth(calculate_angle(rs, re, rw_), self.right_angle_buf)

        self.left_angle_hist.append((ts, la))
        self.right_angle_hist.append((ts, ra))
        self.left_angle_hist  = [(t,a) for t,a in self.left_angle_hist  if ts-t <= WINDOW_SECS]
        self.right_angle_hist = [(t,a) for t,a in self.right_angle_hist if ts-t <= WINDOW_SECS]

        def total_change(hist):
            if len(hist) < 2: return 0.0
            return sum(abs(hist[i][1]-hist[i-1][1]) for i in range(1, len(hist)))

        if total_change(self.left_angle_hist)  > SPEED_THRESH: self.left_warn_until  = ts + WARN_HOLD
        if total_change(self.right_angle_hist) > SPEED_THRESH: self.right_warn_until = ts + WARN_HOLD

        left_bad  = ts < self.left_warn_until
        right_bad = ts < self.right_warn_until

        lc = (0,0,255) if left_bad  else (255,255,255)
        rc = (0,0,255) if right_bad else (255,255,255)
        draw_line(img, lm, 11, 13, lc, w, h, 3)
        draw_line(img, lm, 13, 15, lc, w, h, 3)
        draw_line(img, lm, 12, 14, rc, w, h, 3)
        draw_line(img, lm, 14, 16, rc, w, h, 3)
        draw_line(img, lm, 11, 12, (255,255,255), w, h, 2)
        draw_line(img, lm, 23, 24, (255,255,255), w, h, 2)

        if la > 160: self.left_stage = "Down"
        if la < 35 and self.left_stage == "Down" and not left_bad:
            self.left_stage = "Up"; self.left_counter += 1

        if ra > 160: self.right_stage = "Down"
        if ra < 35 and self.right_stage == "Down" and not right_bad:
            self.right_stage = "Up"; self.right_counter += 1

        neon_text(img, f"{int(la)}", (int(lm[13].x*w)-18, int(lm[13].y*h)-10), (0,255,0),   0.5)
        neon_text(img, f"{int(ra)}", (int(lm[14].x*w)-18, int(lm[14].y*h)-10), (0,200,255), 0.5)

        elapsed  = int(ts - self.start_time)
        total    = self.left_counter + self.right_counter
        calories = round(total * 0.5 * (USER_WEIGHT/65), 2)
        self._draw_dashboard(img, w, [
            ("L REPS", str(self.left_counter),  (0,255,0)),
            ("R REPS", str(self.right_counter), (0,200,255)),
            ("TIME",   f"{elapsed}s",            (255,255,255)),
            ("CAL",    str(calories),             (255,150,0)),
        ])

        # ── warning AFTER dashboard ──
        bad_sides = []
        if left_bad:  bad_sides.append("LEFT")
        if right_bad: bad_sides.append("RIGHT")
        if bad_sides:
            warning_bottom(img, f"SLOW DOWN — {' & '.join(bad_sides)} ARM!  Rep not counted.")

    # ─────────────────────────────────────────
    #  PUSH-UP
    # ─────────────────────────────────────────
    def process_pushup(self, img, lm, h, w, ts):
        SPINE_MIN        = 150
        SPINE_MAX        = 210
        BAD_FRAME_THRESH = 8

        ls  = [lm[11].x, lm[11].y]; le  = [lm[13].x, lm[13].y]; lw_ = [lm[15].x, lm[15].y]
        rs  = [lm[12].x, lm[12].y]; re  = [lm[14].x, lm[14].y]; rw_ = [lm[16].x, lm[16].y]
        elbow_avg = smooth(
            (calculate_angle(ls, le, lw_) + calculate_angle(rs, re, rw_)) / 2,
            self.pushup_elbow_buf)

        hip_vis       = lm[23].visibility if hasattr(lm[23], 'visibility') else 1.0
        knee_vis      = lm[25].visibility if hasattr(lm[25], 'visibility') else 1.0
        spine_visible = (hip_vis > 0.4 and knee_vis > 0.4)
        spine_a       = 0.0

        if spine_visible:
            spine_a = calculate_angle([lm[11].x, lm[11].y],
                                      [lm[23].x, lm[23].y],
                                      [lm[25].x, lm[25].y])
            if not (SPINE_MIN <= spine_a <= SPINE_MAX):
                self.pushup_back_bad_frames += 1
            else:
                self.pushup_back_bad_frames = max(0, self.pushup_back_bad_frames - 2)
            if self.pushup_back_bad_frames >= BAD_FRAME_THRESH:
                self.pushup_back_warn_until = ts + 1.2

        back_bad    = ts < self.pushup_back_warn_until
        spine_color = (0,0,255) if back_bad else (255,255,255)

        draw_line(img, lm, 11, 13, (255,255,255), w, h, 2)
        draw_line(img, lm, 13, 15, (255,255,255), w, h, 2)
        draw_line(img, lm, 12, 14, (255,255,255), w, h, 2)
        draw_line(img, lm, 14, 16, (255,255,255), w, h, 2)
        draw_line(img, lm, 11, 12, (255,255,255), w, h, 2)
        if spine_visible:
            draw_line(img, lm, 11, 23, spine_color, w, h, 3)
            draw_line(img, lm, 23, 25, spine_color, w, h, 3)

        if elbow_avg > 155: self.pushup_stage = "Up"
        if elbow_avg < 85 and self.pushup_stage == "Up":
            self.pushup_stage = "Down"
            if not back_bad:
                self.pushup_counter += 1

        neon_text(img, f"E:{int(elbow_avg)}",
                  (int(lm[13].x*w)-30, int(lm[13].y*h)-12), (0,200,255), 0.5)
        if spine_visible:
            neon_text(img, f"B:{int(spine_a)}",
                      (int(lm[23].x*w)-30, int(lm[23].y*h)-12),
                      (0,0,255) if back_bad else (0,255,0), 0.5)

        elapsed  = int(ts - self.start_time)
        calories = round(self.pushup_counter * 0.7 * (USER_WEIGHT/65), 2)
        self._draw_dashboard(img, w, [
            ("PUSH-UPS", str(self.pushup_counter),       (0,255,0)),
            ("STAGE",    str(self.pushup_stage or "—"),  (0,200,255)),
            ("TIME",     f"{elapsed}s",                  (255,255,255)),
            ("CAL",      str(calories),                  (255,150,0)),
        ])

        # ── warning AFTER dashboard ──
        if back_bad:
            warning_bottom(img,
                f"STRAIGHTEN YOUR BACK! ({int(spine_a)} — keep {SPINE_MIN}-{SPINE_MAX})")

    # ─────────────────────────────────────────
    #  SQUAT
    # ─────────────────────────────────────────
    def process_squat(self, img, lm, h, w, ts):
        """
        Spine correction — ONLY at the bottom of the squat (avg_knee < 100°).
        Rationale: during the descent and ascent the torso naturally leans
        forward — that is normal. We only care if the back is dangerously
        rounded/overextended at the deepest point of the rep.

        Knee valgus is still checked throughout the whole movement because
        knees caving inward at ANY point during a squat is harmful.
        """
        # Spine thresholds — checked ONLY when deeply squatting
        SPINE_LOW        = 100   # below this = dangerously folded forward
        SPINE_HIGH       = 240   # above this = dangerously hyperextended back
        KNEE_INWARD      = 0.22
        KNEE_BAD_THRESH  = 10
        SQUAT_DEEP_ANGLE = 100   # knee angle must be below this to check spine

        lh  = [lm[23].x, lm[23].y]; lk  = [lm[25].x, lm[25].y]; la_ = [lm[27].x, lm[27].y]
        rh  = [lm[24].x, lm[24].y]; rk  = [lm[26].x, lm[26].y]; ra_ = [lm[28].x, lm[28].y]
        lka      = smooth(calculate_angle(lh, lk, la_), self.squat_lk_buf)
        rka      = smooth(calculate_angle(rh, rk, ra_), self.squat_rk_buf)
        avg_knee = (lka + rka) / 2

        spine_a = calculate_angle([lm[11].x, lm[11].y],
                                  [lm[23].x, lm[23].y],
                                  [lm[25].x, lm[25].y])

        # ── spine check: ONLY when deeply bent ──
        at_bottom = avg_knee < SQUAT_DEEP_ANGLE
        if at_bottom and (spine_a < SPINE_LOW or spine_a > SPINE_HIGH):
            # fire immediately at bottom — no frame accumulation needed
            self.squat_back_warn_until = ts + 1.5

        back_bad = ts < self.squat_back_warn_until

        # ── knee valgus: checked throughout (sustained counter) ──
        hip_width  = abs(lm[23].x - lm[24].x)
        left_cave  = (lm[23].x - lm[25].x) > KNEE_INWARD * hip_width
        right_cave = (lm[26].x - lm[24].x) > KNEE_INWARD * hip_width
        if left_cave or right_cave:
            self.squat_knee_bad_frames += 1
        else:
            self.squat_knee_bad_frames = max(0, self.squat_knee_bad_frames - 2)
        if self.squat_knee_bad_frames >= KNEE_BAD_THRESH:
            self.squat_knee_warn_until = ts + 1.5

        knee_bad = ts < self.squat_knee_warn_until

        # ── skeleton colors ──
        sc  = (0,0,255) if back_bad else (255,255,255)
        lkc = (0,0,255) if (left_cave  and knee_bad) else (255,255,255)
        rkc = (0,0,255) if (right_cave and knee_bad) else (255,255,255)

        draw_line(img, lm, 11, 12, (255,255,255), w, h, 2)
        draw_line(img, lm, 23, 24, (255,255,255), w, h, 2)
        draw_line(img, lm, 11, 23, sc,  w, h, 3)
        draw_line(img, lm, 12, 24, sc,  w, h, 3)
        draw_line(img, lm, 23, 25, lkc, w, h, 3)
        draw_line(img, lm, 25, 27, lkc, w, h, 3)
        draw_line(img, lm, 24, 26, rkc, w, h, 3)
        draw_line(img, lm, 26, 28, rkc, w, h, 3)

        # ── rep counting ──
        posture_ok = not back_bad and not knee_bad
        if avg_knee > 160: self.squat_stage = "Up"
        if avg_knee < 90 and self.squat_stage == "Up":
            self.squat_stage = "Down"
            if posture_ok:
                self.squat_counter += 1

        # ── angle label ──
        lkx, lky = int(lm[25].x*w), int(lm[25].y*h)
        neon_text(img, f"{int(lka)}", (lkx-18, lky-10),
                  (0,0,255) if (left_cave and knee_bad) else (0,255,0), 0.5)

        elapsed  = int(ts - self.start_time)
        calories = round(self.squat_counter * 0.6 * (USER_WEIGHT/65), 2)
        self._draw_dashboard(img, w, [
            ("SQUATS", str(self.squat_counter),       (0,255,0)),
            ("STAGE",  str(self.squat_stage or "—"),  (0,200,255)),
            ("TIME",   f"{elapsed}s",                 (255,255,255)),
            ("CAL",    str(calories),                 (255,150,0)),
        ])

        # ── warning AFTER dashboard ──
        warns = []
        if back_bad: warns.append(f"BACK TOO BENT! ({int(spine_a)})")
        if knee_bad: warns.append("KNEES CAVING IN!")
        if warns:
            warning_bottom(img, " | ".join(warns))

    # ─────────────────────────────────────────
    #  DASHBOARD  (top centre solid box)
    # ─────────────────────────────────────────
    def _draw_dashboard(self, img, w, metrics):
        box_w, box_h = 500, 78
        x1 = (w - box_w) // 2; y1 = 4
        solid_dashboard(img, x1, y1, x1+box_w, y1+box_h)
        col_w = box_w // len(metrics)
        for i, (label, value, color) in enumerate(metrics):
            cx = x1 + i*col_w + 10
            neon_text(img, label, (cx, y1+24), (160,160,160), 0.38, 1)
            neon_text(img, value, (cx, y1+58), color, 0.72, 2)

    def _draw_landmarks(self, img, lm, h, w):
        for l in lm:
            cv2.circle(img, (int(l.x*w), int(l.y*h)), 3, (0,255,180), -1)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ts  = time.time()
        if ts - self.last_process_time < 0.04:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        self.last_process_time = ts

        rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = self.pose.detect_for_video(mp_image, int(ts*1000))

        if results.pose_landmarks:
            lm      = results.pose_landmarks[0]
            h, w, _ = img.shape
            self._draw_landmarks(img, lm, h, w)
            ex = self.exercise
            if   ex == "Dumbbell Curl": self.process_curl(img, lm, h, w, ts)
            elif ex == "Push-Up":       self.process_pushup(img, lm, h, w, ts)
            elif ex == "Squat":         self.process_squat(img, lm, h, w, ts)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@300;400;600&display=swap');
body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f; color: #e0e0e0; font-family: 'Inter', sans-serif;
}
h1 { font-family: 'Orbitron', sans-serif; letter-spacing: 3px; color: #00ffb3; }
.stButton>button {
    background: linear-gradient(135deg, #00ffb3, #00b3ff);
    color: #0a0a0f; font-weight: 700; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;
}
.stButton>button:hover { opacity: 0.8; }
div[data-baseweb="select"] { background: #14141f; border-radius: 8px; }
.stMarkdown hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)

st.title("🏋️ AI GYM TRAINER PRO")
st.markdown("*Real-time pose detection · Posture correction · 3 exercises*")
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    exercise_choice = st.selectbox("Exercise", EXERCISES)
    st.markdown("---")
    st.markdown("### 📋 Posture Rules")
    if exercise_choice == "Dumbbell Curl":
        st.info("🔴 **Speed** — total elbow change > 250° in 2 s = too fast. Warning shown bottom of video, rep not counted.")
    elif exercise_choice == "Push-Up":
        st.info("✅ **Rep by elbow** — works front-facing.\n\n🔴 **Spine** — only when hips visible. Warning bottom of video.")
    elif exercise_choice == "Squat":
        st.info("🔴 **Back** — only checked at deepest point (knee < 100°). Ignored during descent/ascent.\n\n🔴 **Knee cave** — checked throughout.")

col1, col2 = st.columns([3, 1])
with col1:
    ctx = webrtc_streamer(
        key="pose",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True
    )
    if ctx.video_processor:
        ctx.video_processor.exercise = exercise_choice

with col2:
    st.subheader("📊 Info")
    st.write(f"**Active:** `{exercise_choice}`")
    st.markdown("---")
    st.markdown("**🟢 Green** — Good form\n\n**🔴 Red** — Bad posture\n\n**⚠️ Bottom banner** — Fix now!")

st.markdown("---")
st.subheader("📁 Workout History")
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("💾 Save Workout"):
    st.session_state.history.append({
        "Time": time.strftime("%H:%M:%S"),
        "Exercise": exercise_choice,
        "Note": "Session saved"
    })
    st.success("Saved!")

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)