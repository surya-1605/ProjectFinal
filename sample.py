import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

try:
    import winsound
    def beep(freq, dur): winsound.Beep(freq, dur)
except ImportError:
    def beep(freq, dur): pass

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
#  CONFIG  —  change EXERCISE to switch mode
# ─────────────────────────────────────────────
USER_WEIGHT = 65
EXERCISE    = "squat"   # "curl" | "pushup" | "squat"

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def smooth_angle(new_angle, buf, window=5):
    buf.append(new_angle)
    if len(buf) > window:
        buf.pop(0)
    return sum(buf) / len(buf)


def neon_text(img, text, pos, color, scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0),       thickness+3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness+1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,          thickness)


def draw_line(img, lm, a, b, color, w, h, thick=2):
    x1,y1 = int(lm[a].x*w), int(lm[a].y*h)
    x2,y2 = int(lm[b].x*w), int(lm[b].y*h)
    cv2.line(img, (x1,y1), (x2,y2), color, thick, cv2.LINE_AA)


def draw_top_dashboard(canvas, metrics):
    """
    Solid dark strip at the VERY TOP of the canvas (above the video area).
    canvas height is extended by DASH_H pixels at top.
    metrics: list of (label, value, color)
    """
    w = canvas.shape[1]
    DASH_H = 70
    cv2.rectangle(canvas, (0,0), (w, DASH_H), (14,14,20), -1)
    cv2.line(canvas, (0, DASH_H), (w, DASH_H), (45,45,65), 1)

    col_w = w // len(metrics)
    for i, (label, value, color) in enumerate(metrics):
        cx = i * col_w + 14
        cv2.putText(canvas, label, (cx, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110,110,130), 1)
        neon_text(canvas, value, (cx, 56), color, 0.72, 2)


def draw_bottom_warning(canvas, msg, video_h):
    """
    Solid red strip at the VERY BOTTOM of the canvas (below the video area).
    """
    w      = canvas.shape[1]
    WARN_H = 46
    y1     = video_h  # starts right after video rows
    y2     = y1 + WARN_H
    cv2.rectangle(canvas, (0, y1), (w, y2), (10,0,0),   -1)
    cv2.rectangle(canvas, (0, y1), (w, y2), (0,0,180),  1)
    cv2.putText(canvas, f"!  {msg}", (12, y1+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80,80,255), 2)


# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
pose = vision.PoseLandmarker.create_from_options(options)

cap        = cv2.VideoCapture(0)
start_time = time.time()

DASH_H = 70   # pixels added above for dashboard
WARN_H = 46   # pixels added below for warning

# ── Curl state ──
left_counter = right_counter = 0
left_stage = right_stage = None
left_buf = []; right_buf = []
left_angle_hist = []; right_angle_hist = []
left_warn_until = right_warn_until = 0.0

# ── Push-Up state ──
pushup_counter = 0
pushup_stage = None
pushup_elbow_buf = []
pushup_back_bad_frames = 0
pushup_back_warn_until = 0.0

# ── Squat state ──
squat_counter = 0
squat_stage = None
squat_lk_buf = []; squat_rk_buf = []
squat_back_warn_until = 0.0
squat_knee_bad_frames = 0
squat_knee_warn_until = 0.0

calories = 0.0
elapsed_time = 0

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ts           = time.time()
    elapsed_time = int(ts - start_time)
    h, w, _      = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results   = pose.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # Collect dashboard/warning data — drawn onto canvas, NOT onto frame
    dashboard_metrics = []
    active_warning    = ""

    if results.pose_landmarks:
        lm = results.pose_landmarks[0]

        for landmark in lm:
            cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 4, (0,255,180), -1)

        # ═══════════════════════════════════════
        #  DUMBBELL CURL
        # ═══════════════════════════════════════
        if EXERCISE == "curl":
            SPEED_THRESH = 250; WINDOW_SECS = 2.0; WARN_HOLD = 1.5

            ls = [lm[11].x, lm[11].y]; le = [lm[13].x, lm[13].y]; lw_ = [lm[15].x, lm[15].y]
            rs = [lm[12].x, lm[12].y]; re = [lm[14].x, lm[14].y]; rw_ = [lm[16].x, lm[16].y]
            la = smooth_angle(calculate_angle(ls, le, lw_), left_buf)
            ra = smooth_angle(calculate_angle(rs, re, rw_), right_buf)

            left_angle_hist.append((ts, la));  left_angle_hist[:]  = [(t,a) for t,a in left_angle_hist  if ts-t <= WINDOW_SECS]
            right_angle_hist.append((ts, ra)); right_angle_hist[:] = [(t,a) for t,a in right_angle_hist if ts-t <= WINDOW_SECS]

            def total_change(hist):
                if len(hist) < 2: return 0.0
                return sum(abs(hist[i][1]-hist[i-1][1]) for i in range(1, len(hist)))

            if total_change(left_angle_hist)  > SPEED_THRESH: left_warn_until  = ts + WARN_HOLD
            if total_change(right_angle_hist) > SPEED_THRESH: right_warn_until = ts + WARN_HOLD

            left_bad  = ts < left_warn_until
            right_bad = ts < right_warn_until

            lc = (0,0,255) if left_bad  else (0,255,100)
            rc = (0,0,255) if right_bad else (0,255,100)
            draw_line(frame, lm, 11, 13, lc, w, h, 3)
            draw_line(frame, lm, 13, 15, lc, w, h, 3)
            draw_line(frame, lm, 12, 14, rc, w, h, 3)
            draw_line(frame, lm, 14, 16, rc, w, h, 3)
            draw_line(frame, lm, 11, 12, (200,200,200), w, h)
            draw_line(frame, lm, 23, 24, (200,200,200), w, h)

            if la > 160: left_stage = "Down"
            if la < 35 and left_stage == "Down" and not left_bad:
                left_stage = "Up"; left_counter += 1; beep(1000, 150)

            if ra > 160: right_stage = "Down"
            if ra < 35 and right_stage == "Down" and not right_bad:
                right_stage = "Up"; right_counter += 1; beep(800, 150)

            neon_text(frame, f"{int(la)}", (int(lm[13].x*w)-18, int(lm[13].y*h)-10), (0,255,80),  0.5)
            neon_text(frame, f"{int(ra)}", (int(lm[14].x*w)-18, int(lm[14].y*h)-10), (0,200,255), 0.5)

            total_reps = left_counter + right_counter
            calories   = round(total_reps * 0.5 * (USER_WEIGHT/65), 2)
            dashboard_metrics = [
                ("LEFT REPS",  str(left_counter),  (0,255,80)),
                ("RIGHT REPS", str(right_counter), (0,200,255)),
                ("TOTAL",      str(total_reps),    (255,255,255)),
                ("TIME",       f"{elapsed_time}s", (255,255,255)),
                ("CALORIES",   f"{calories}",      (255,160,40)),
            ]
            bad_sides = []
            if left_bad:  bad_sides.append("LEFT")
            if right_bad: bad_sides.append("RIGHT")
            if bad_sides:
                active_warning = f"SLOW DOWN — {' & '.join(bad_sides)} ARM!  Rep not counted."

        # ═══════════════════════════════════════
        #  PUSH-UP
        # ═══════════════════════════════════════
        elif EXERCISE == "pushup":
            SPINE_MIN = 150; SPINE_MAX = 210; BAD_FRAME_THRESH = 8

            ls = [lm[11].x, lm[11].y]; le = [lm[13].x, lm[13].y]; lw_ = [lm[15].x, lm[15].y]
            rs = [lm[12].x, lm[12].y]; re = [lm[14].x, lm[14].y]; rw_ = [lm[16].x, lm[16].y]
            elbow_avg = smooth_angle(
                (calculate_angle(ls, le, lw_) + calculate_angle(rs, re, rw_)) / 2,
                pushup_elbow_buf)

            hip_vis       = lm[23].visibility if hasattr(lm[23], 'visibility') else 1.0
            knee_vis      = lm[25].visibility if hasattr(lm[25], 'visibility') else 1.0
            spine_visible = (hip_vis > 0.4 and knee_vis > 0.4)
            spine_a       = 0.0

            if spine_visible:
                spine_a = calculate_angle([lm[11].x, lm[11].y],
                                          [lm[23].x, lm[23].y],
                                          [lm[25].x, lm[25].y])
                if not (SPINE_MIN <= spine_a <= SPINE_MAX):
                    pushup_back_bad_frames += 1
                else:
                    pushup_back_bad_frames = max(0, pushup_back_bad_frames - 2)
                if pushup_back_bad_frames >= BAD_FRAME_THRESH:
                    pushup_back_warn_until = ts + 1.2

            back_bad    = ts < pushup_back_warn_until
            spine_color = (0,0,255) if back_bad else (0,255,100)

            draw_line(frame, lm, 11, 13, (200,200,200), w, h)
            draw_line(frame, lm, 13, 15, (200,200,200), w, h)
            draw_line(frame, lm, 12, 14, (200,200,200), w, h)
            draw_line(frame, lm, 14, 16, (200,200,200), w, h)
            draw_line(frame, lm, 11, 12, (200,200,200), w, h)
            if spine_visible:
                draw_line(frame, lm, 11, 23, spine_color, w, h, 3)
                draw_line(frame, lm, 23, 25, spine_color, w, h, 3)

            if elbow_avg > 155: pushup_stage = "Up"
            if elbow_avg < 85 and pushup_stage == "Up":
                pushup_stage = "Down"
                if not back_bad:
                    pushup_counter += 1; beep(1000, 150)
                else:
                    beep(400, 300)

            neon_text(frame, f"E:{int(elbow_avg)}",
                      (int(lm[13].x*w)-30, int(lm[13].y*h)-12), (0,200,255), 0.5)
            if spine_visible:
                neon_text(frame, f"B:{int(spine_a)}",
                          (int(lm[23].x*w)-30, int(lm[23].y*h)-12),
                          (0,0,255) if back_bad else (0,255,80), 0.5)

            calories = round(pushup_counter * 0.7 * (USER_WEIGHT/65), 2)
            dashboard_metrics = [
                ("PUSH-UPS", str(pushup_counter),       (0,255,80)),
                ("STAGE",    str(pushup_stage or "-"),  (0,200,255)),
                ("ELBOW",    f"{int(elbow_avg)}",       (0,200,255)),
                ("TIME",     f"{elapsed_time}s",        (255,255,255)),
                ("CALORIES", f"{calories}",              (255,160,40)),
            ]
            if back_bad:
                active_warning = f"STRAIGHTEN YOUR BACK! ({int(spine_a)} — keep {SPINE_MIN}-{SPINE_MAX})"

        # ═══════════════════════════════════════
        #  SQUAT
        # ═══════════════════════════════════════
        elif EXERCISE == "squat":
            SPINE_LOW = 100; SPINE_HIGH = 240; KNEE_INWARD = 0.22
            KNEE_BAD_THRESH = 10; SQUAT_DEEP_ANGLE = 100

            lh = [lm[23].x, lm[23].y]; lk = [lm[25].x, lm[25].y]; la_ = [lm[27].x, lm[27].y]
            rh = [lm[24].x, lm[24].y]; rk = [lm[26].x, lm[26].y]; ra_ = [lm[28].x, lm[28].y]
            lka      = smooth_angle(calculate_angle(lh, lk, la_), squat_lk_buf)
            rka      = smooth_angle(calculate_angle(rh, rk, ra_), squat_rk_buf)
            avg_knee = (lka + rka) / 2

            spine_a = calculate_angle([lm[11].x, lm[11].y],
                                      [lm[23].x, lm[23].y],
                                      [lm[25].x, lm[25].y])

            at_bottom = avg_knee < SQUAT_DEEP_ANGLE
            if at_bottom and (spine_a < SPINE_LOW or spine_a > SPINE_HIGH):
                squat_back_warn_until = ts + 1.5
            back_bad = ts < squat_back_warn_until

            hip_width  = abs(lm[23].x - lm[24].x)
            left_cave  = (lm[23].x - lm[25].x) > KNEE_INWARD * hip_width
            right_cave = (lm[26].x - lm[24].x) > KNEE_INWARD * hip_width
            if left_cave or right_cave:
                squat_knee_bad_frames += 1
            else:
                squat_knee_bad_frames = max(0, squat_knee_bad_frames - 2)
            if squat_knee_bad_frames >= KNEE_BAD_THRESH:
                squat_knee_warn_until = ts + 1.5
            knee_bad = ts < squat_knee_warn_until

            sc  = (0,0,255) if back_bad else (0,255,100)
            lkc = (0,0,255) if (left_cave  and knee_bad) else (0,255,100)
            rkc = (0,0,255) if (right_cave and knee_bad) else (0,255,100)

            draw_line(frame, lm, 11, 12, (200,200,200), w, h)
            draw_line(frame, lm, 23, 24, (200,200,200), w, h)
            draw_line(frame, lm, 11, 23, sc,  w, h, 3)
            draw_line(frame, lm, 12, 24, sc,  w, h, 3)
            draw_line(frame, lm, 23, 25, lkc, w, h, 3)
            draw_line(frame, lm, 25, 27, lkc, w, h, 3)
            draw_line(frame, lm, 24, 26, rkc, w, h, 3)
            draw_line(frame, lm, 26, 28, rkc, w, h, 3)

            posture_ok = not back_bad and not knee_bad
            if avg_knee > 160: squat_stage = "Up"
            if avg_knee < 90 and squat_stage == "Up":
                squat_stage = "Down"
                if posture_ok:
                    squat_counter += 1; beep(1000, 150)
                else:
                    beep(400, 300)

            lkx, lky = int(lm[25].x*w), int(lm[25].y*h)
            neon_text(frame, f"{int(lka)}", (lkx-18, lky-10),
                      (0,0,255) if (left_cave and knee_bad) else (0,255,80), 0.5)

            calories = round(squat_counter * 0.6 * (USER_WEIGHT/65), 2)
            dashboard_metrics = [
                ("SQUATS",   str(squat_counter),       (0,255,80)),
                ("STAGE",    str(squat_stage or "-"),  (0,200,255)),
                ("BACK",     f"{int(spine_a)}",        (0,0,255) if back_bad else (0,255,80)),
                ("TIME",     f"{elapsed_time}s",       (255,255,255)),
                ("CALORIES", f"{calories}",             (255,160,40)),
            ]
            warns = []
            if back_bad: warns.append(f"BACK TOO BENT! ({int(spine_a)})")
            if knee_bad: warns.append("KNEES CAVING IN!")
            if warns:
                active_warning = " | ".join(warns)

    # ── Build final canvas: [DASH_H top] + [frame] + [WARN_H bottom] ──
    canvas_h = DASH_H + h + WARN_H
    canvas   = np.zeros((canvas_h, w, 3), dtype=np.uint8)

    # paste video in middle
    canvas[DASH_H : DASH_H + h, :] = frame

    # draw dashboard at top
    if dashboard_metrics:
        draw_top_dashboard(canvas, dashboard_metrics)

    # draw warning at bottom
    if active_warning:
        draw_bottom_warning(canvas, active_warning, DASH_H + h)

    # exercise label top-right corner of video area
    neon_text(canvas, EXERCISE.upper(), (w - 140, DASH_H + 28), (0,255,180), 0.6, 2)

    cv2.imshow("AI GYM TRAINER PRO", canvas)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
#  SAVE CSV
# ─────────────────────────────────────────────
if EXERCISE == "curl":
    total_reps  = left_counter + right_counter
    rep_detail  = f"L:{left_counter} R:{right_counter} Total:{total_reps}"
elif EXERCISE == "pushup":
    total_reps  = pushup_counter
    rep_detail  = f"{total_reps} push-ups"
else:
    total_reps  = squat_counter
    rep_detail  = f"{total_reps} squats"

duration_str = f"{elapsed_time // 60}m {elapsed_time % 60}s"

data = {
    "Saved At":   [time.strftime("%Y-%m-%d %H:%M:%S")],
    "Exercise":   [EXERCISE],
    "Reps":       [rep_detail],
    "Total Reps": [total_reps],
    "Duration":   [duration_str],
    "Calories":   [f"{calories} kcal"],
}
df = pd.DataFrame(data)
if not os.path.exists("workout_history.csv"):
    df.to_csv("workout_history.csv", index=False)
else:
    df.to_csv("workout_history.csv", mode='a', header=False, index=False)

print(f"\n✅ Workout saved → {EXERCISE} | {rep_detail} | {duration_str} | {calories} kcal")