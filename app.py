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
#  DRAWING HELPERS  (video-only — no dashboard/warning on frame)
# ─────────────────────────────────────────────
def neon_text(img, text, pos, color, scale=0.65, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0),       thickness+3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness+1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,          thickness)


def draw_line(img, lm, a, b, color, w, h, thick=2):
    x1, y1 = int(lm[a].x*w), int(lm[a].y*h)
    x2, y2 = int(lm[b].x*w), int(lm[b].y*h)
    cv2.line(img, (x1,y1), (x2,y2), color, thick, cv2.LINE_AA)


EXERCISES = ["Dumbbell Curl", "Push-Up", "Squat"]


# ─────────────────────────────────────────────
#  VIDEO PROCESSOR
#  All metrics stored as instance vars so Streamlit can read them.
#  Nothing drawn on frame except skeleton + angle labels.
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

        # ── metrics exposed to Streamlit ──
        self.metrics  = {}   # dict of label → value displayed in dashboard
        self.warning  = ""   # current warning string ("" = no warning)

        # ── raw snapshot values (read by Save Workout button) ──
        self.snap_calories = 0.0
        self.snap_elapsed  = 0

        # ── Curl ──
        self.left_counter     = 0
        self.right_counter    = 0
        self.left_stage       = None
        self.right_stage      = None
        self.left_angle_buf   = []
        self.right_angle_buf  = []
        self.left_angle_hist  = []
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
        self.squat_back_warn_until = 0.0
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

        # skeleton
        lc = (0,0,255) if left_bad  else (0,255,100)
        rc = (0,0,255) if right_bad else (0,255,100)
        draw_line(img, lm, 11, 13, lc, w, h, 3)
        draw_line(img, lm, 13, 15, lc, w, h, 3)
        draw_line(img, lm, 12, 14, rc, w, h, 3)
        draw_line(img, lm, 14, 16, rc, w, h, 3)
        draw_line(img, lm, 11, 12, (200,200,200), w, h, 2)
        draw_line(img, lm, 23, 24, (200,200,200), w, h, 2)

        # rep counting
        if la > 160: self.left_stage = "Down"
        if la < 35 and self.left_stage == "Down" and not left_bad:
            self.left_stage = "Up"; self.left_counter += 1

        if ra > 160: self.right_stage = "Down"
        if ra < 35 and self.right_stage == "Down" and not right_bad:
            self.right_stage = "Up"; self.right_counter += 1

        # angle labels on video
        neon_text(img, f"{int(la)}", (int(lm[13].x*w)-18, int(lm[13].y*h)-10), (0,255,80),  0.5)
        neon_text(img, f"{int(ra)}", (int(lm[14].x*w)-18, int(lm[14].y*h)-10), (0,200,255), 0.5)

        # ── publish metrics & warning to Streamlit ──
        elapsed  = int(ts - self.start_time)
        total    = self.left_counter + self.right_counter
        calories = round(total * 0.5 * (USER_WEIGHT/65), 2)
        self.snap_elapsed  = elapsed
        self.snap_calories = calories
        self.metrics = {
            "LEFT REPS":  str(self.left_counter),
            "RIGHT REPS": str(self.right_counter),
            "TOTAL REPS": str(total),
            "TIME":       f"{elapsed}s",
            "CALORIES":   f"{calories} kcal",
        }
        bad_sides = []
        if left_bad:  bad_sides.append("LEFT")
        if right_bad: bad_sides.append("RIGHT")
        self.warning = f"⚠️ SLOW DOWN — {' & '.join(bad_sides)} ARM!  Rep not counted." if bad_sides else ""

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
        spine_color = (0,0,255) if back_bad else (0,255,100)

        draw_line(img, lm, 11, 13, (200,200,200), w, h, 2)
        draw_line(img, lm, 13, 15, (200,200,200), w, h, 2)
        draw_line(img, lm, 12, 14, (200,200,200), w, h, 2)
        draw_line(img, lm, 14, 16, (200,200,200), w, h, 2)
        draw_line(img, lm, 11, 12, (200,200,200), w, h, 2)
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
                      (0,0,255) if back_bad else (0,255,80), 0.5)

        elapsed  = int(ts - self.start_time)
        calories = round(self.pushup_counter * 0.7 * (USER_WEIGHT/65), 2)
        self.snap_elapsed  = elapsed
        self.snap_calories = calories
        self.metrics = {
            "PUSH-UPS": str(self.pushup_counter),
            "STAGE":    str(self.pushup_stage or "—"),
            "ELBOW":    f"{int(elbow_avg)}°",
            "TIME":     f"{elapsed}s",
            "CALORIES": f"{calories} kcal",
        }
        self.warning = f"⚠️ STRAIGHTEN YOUR BACK! ({int(spine_a)}° — keep {SPINE_MIN}–{SPINE_MAX}°)" if back_bad else ""

    # ─────────────────────────────────────────
    #  SQUAT
    # ─────────────────────────────────────────
    def process_squat(self, img, lm, h, w, ts):
        SPINE_LOW        = 100
        SPINE_HIGH       = 240
        KNEE_INWARD      = 0.22
        KNEE_BAD_THRESH  = 10
        SQUAT_DEEP_ANGLE = 100

        lh  = [lm[23].x, lm[23].y]; lk  = [lm[25].x, lm[25].y]; la_ = [lm[27].x, lm[27].y]
        rh  = [lm[24].x, lm[24].y]; rk  = [lm[26].x, lm[26].y]; ra_ = [lm[28].x, lm[28].y]
        lka      = smooth(calculate_angle(lh, lk, la_), self.squat_lk_buf)
        rka      = smooth(calculate_angle(rh, rk, ra_), self.squat_rk_buf)
        avg_knee = (lka + rka) / 2

        spine_a = calculate_angle([lm[11].x, lm[11].y],
                                  [lm[23].x, lm[23].y],
                                  [lm[25].x, lm[25].y])

        at_bottom = avg_knee < SQUAT_DEEP_ANGLE
        if at_bottom and (spine_a < SPINE_LOW or spine_a > SPINE_HIGH):
            self.squat_back_warn_until = ts + 1.5
        back_bad = ts < self.squat_back_warn_until

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

        sc  = (0,0,255) if back_bad else (0,255,100)
        lkc = (0,0,255) if (left_cave  and knee_bad) else (0,255,100)
        rkc = (0,0,255) if (right_cave and knee_bad) else (0,255,100)

        draw_line(img, lm, 11, 12, (200,200,200), w, h, 2)
        draw_line(img, lm, 23, 24, (200,200,200), w, h, 2)
        draw_line(img, lm, 11, 23, sc,  w, h, 3)
        draw_line(img, lm, 12, 24, sc,  w, h, 3)
        draw_line(img, lm, 23, 25, lkc, w, h, 3)
        draw_line(img, lm, 25, 27, lkc, w, h, 3)
        draw_line(img, lm, 24, 26, rkc, w, h, 3)
        draw_line(img, lm, 26, 28, rkc, w, h, 3)

        posture_ok = not back_bad and not knee_bad
        if avg_knee > 160: self.squat_stage = "Up"
        if avg_knee < 90 and self.squat_stage == "Up":
            self.squat_stage = "Down"
            if posture_ok:
                self.squat_counter += 1

        lkx, lky = int(lm[25].x*w), int(lm[25].y*h)
        neon_text(img, f"{int(lka)}", (lkx-18, lky-10),
                  (0,0,255) if (left_cave and knee_bad) else (0,255,80), 0.5)

        elapsed  = int(ts - self.start_time)
        calories = round(self.squat_counter * 0.6 * (USER_WEIGHT/65), 2)
        self.snap_elapsed  = elapsed
        self.snap_calories = calories
        self.metrics = {
            "SQUATS":   str(self.squat_counter),
            "STAGE":    str(self.squat_stage or "—"),
            "KNEE":     f"{int(lka)}°",
            "TIME":     f"{elapsed}s",
            "CALORIES": f"{calories} kcal",
        }
        warns = []
        if back_bad: warns.append(f"BACK TOO BENT! ({int(spine_a)}°)")
        if knee_bad: warns.append("KNEES CAVING IN!")
        self.warning = "⚠️ " + " | ".join(warns) if warns else ""

    # ─────────────────────────────────────────
    #  RECV  — only skeleton on frame, no dashboard/warning
    # ─────────────────────────────────────────
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
            for l in lm:
                cv2.circle(img, (int(l.x*w), int(l.y*h)), 3, (0,255,180), -1)
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

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}
h1 {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 3px;
    color: #00ffb3;
}

/* ── Dashboard metric cards ── */
.dash-grid {
    display: flex;
    gap: 12px;
    margin-bottom: 10px;
}
.dash-card {
    flex: 1;
    background: #13131e;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 14px 10px 10px;
    text-align: center;
}
.dash-card .label {
    font-size: 0.68rem;
    color: #666;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: 'Inter', sans-serif;
}
.dash-card .value {
    font-size: 1.55rem;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    line-height: 1;
}
.dash-card.green  .value { color: #00ff88; }
.dash-card.cyan   .value { color: #00cfff; }
.dash-card.white  .value { color: #ffffff; }
.dash-card.orange .value { color: #ffaa33; }

/* ── Warning box ── */
.warn-box {
    background: #1a0000;
    border: 1.5px solid #cc0000;
    border-radius: 8px;
    padding: 12px 18px;
    color: #ff4444;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    margin-top: 8px;
}
.warn-box.hidden { display: none; }

div[data-baseweb="select"] { background: #14141f; border-radius: 8px; }
.stMarkdown hr { border-color: #1e1e2e; }
.stButton>button {
    background: linear-gradient(135deg, #00ffb3, #00b3ff);
    color: #0a0a0f; font-weight: 700; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;
}
.stButton>button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

st.title("🏋️ AI GYM TRAINER PRO")
st.markdown("*Real-time pose detection · Posture correction · 3 exercises*")
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    exercise_choice = st.selectbox("Exercise", EXERCISES)
    st.markdown("---")
    st.markdown("### 📋 Posture Rules")
    if exercise_choice == "Dumbbell Curl":
        st.info("🔴 **Speed** — total elbow change > 250° in 2 s = too fast. Rep not counted.")
    elif exercise_choice == "Push-Up":
        st.info("✅ **Rep by elbow** — works front-facing.\n\n🔴 **Spine** — warned only when hips visible.")
    elif exercise_choice == "Squat":
        st.info("🔴 **Back** — only at deepest point (knee < 100°).\n\n🔴 **Knee cave** — checked throughout.")

# ── Dashboard placeholder (TOP — above video) ──
dash_placeholder = st.empty()

# ── Video ──
ctx = webrtc_streamer(
    key="pose",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    async_processing=True
)
if ctx.video_processor:
    ctx.video_processor.exercise = exercise_choice

# ── Warning placeholder (BELOW video) ──
warn_placeholder = st.empty()

# ── Live refresh loop ──
# Polls the processor state and updates Streamlit placeholders in real time.
CARD_COLORS = ["green", "cyan", "white", "orange", "cyan"]

if ctx.video_processor:
    while True:
        proc    = ctx.video_processor
        metrics = proc.metrics
        warning = proc.warning

        # ── render dashboard ──
        if metrics:
            keys   = list(metrics.keys())
            vals   = list(metrics.values())
            colors = CARD_COLORS[:len(keys)]
            cards  = "".join(
                f'<div class="dash-card {c}"><div class="label">{k}</div>'
                f'<div class="value">{v}</div></div>'
                for k, v, c in zip(keys, vals, colors)
            )
            dash_placeholder.markdown(
                f'<div class="dash-grid">{cards}</div>',
                unsafe_allow_html=True
            )

        # ── render warning ──
        if warning:
            warn_placeholder.markdown(
                f'<div class="warn-box">{warning}</div>',
                unsafe_allow_html=True
            )
        else:
            warn_placeholder.empty()

        time.sleep(0.1)   # refresh ~10×/s — smooth enough, low CPU

# ── Workout history ──
st.markdown("---")
st.subheader("📁 Workout History")

if "history" not in st.session_state:
    st.session_state.history = []

col_save, col_clear = st.columns([1, 1])

with col_save:
    if st.button("💾 Save Workout"):
        proc = ctx.video_processor if ctx.video_processor else None

        if proc:
            ex = proc.exercise

            # ── build rep summary string per exercise ──
            if ex == "Dumbbell Curl":
                total_reps = proc.left_counter + proc.right_counter
                rep_detail = f"L:{proc.left_counter}  R:{proc.right_counter}  Total:{total_reps}"
            elif ex == "Push-Up":
                total_reps = proc.pushup_counter
                rep_detail = f"{total_reps} push-ups"
            elif ex == "Squat":
                total_reps = proc.squat_counter
                rep_detail = f"{total_reps} squats"
            else:
                total_reps = 0
                rep_detail = "—"

            # ── format duration mm:ss ──
            secs     = proc.snap_elapsed
            duration = f"{secs // 60}m {secs % 60}s"

            st.session_state.history.append({
                "Saved At":   time.strftime("%Y-%m-%d  %H:%M:%S"),
                "Exercise":   ex,
                "Reps":       rep_detail,
                "Total Reps": total_reps,
                "Duration":   duration,
                "Calories":   f"{proc.snap_calories} kcal",
            })
            st.success(f"✅ Saved — {ex} · {rep_detail} · {duration} · {proc.snap_calories} kcal")
        else:
            st.warning("Start the camera first before saving.")

with col_clear:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.info("History cleared.")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    # ── Summary totals across all saved sessions ──
    st.markdown("#### 📊 Session Totals")
    total_saved_reps = df["Total Reps"].sum()
    total_sessions   = len(df)
    # parse calories back to float
    total_cal = df["Calories"].str.replace(" kcal","").astype(float).sum()

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Total Sessions", total_sessions)
    sc2.metric("Total Reps",     int(total_saved_reps))
    sc3.metric("Total Calories", f"{round(total_cal, 2)} kcal")