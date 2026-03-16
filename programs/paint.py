"""Orly Paint Program — object-tracking paintbrush.

Flow:
1. Show a guide circle in the center of the projection
2. Capture a baseline of the empty scene (hides projection briefly)
3. Wait for user to place an object in the circle (change detection vs baseline)
4. Wait for hands to clear (frame-to-frame stability)
5. Init color tracking on the object; flash + beep confirmation; circle disappears
6. Object movement paints with its dominant color; brush size = object size
7. If object leaves view for 10s, show 5s countdown; if it doesn't return, clear + exit
"""

# --- Configuration ---
CIRCLE_RADIUS = 60           # normalised 0-1000
CIRCLE_COLOR = (0, 255, 255) # cyan
FLASH_COLOR = (0, 255, 0)    # green
FLASH_DURATION = 0.15        # seconds per flash
TONE_FREQ = 880              # Hz
TONE_DURATION = 0.1          # seconds
MISSING_TIMEOUT = 10.0       # seconds before countdown starts
COUNTDOWN_SECONDS = 5        # countdown before clearing
PAINT_TRAIL_MIN_DIST = 5     # min normalised distance between paint stamps

# --- Setup ---
canvas = table.create_canvas()
frame = table.get_frame()
if frame is None:
    table.notify("No camera frame available — cannot start paint program.")
    table.stop()

if not table.stopped:
    fh, fw = frame.shape[:2]

    # Region at center of frame for object detection
    region_size = int(min(fh, fw) * CIRCLE_RADIUS / 500)
    ry = fh // 2 - region_size // 2
    rx = fw // 2 - region_size // 2
    region = (ry, rx, region_size, region_size)

    # --- Step 1: Show guide circle ---
    canvas.circle(500, 500, CIRCLE_RADIUS, CIRCLE_COLOR, thickness=4)
    table.notify("Place an object in the circle on the table.")

    # --- Step 2: Capture baseline (blanks projection so camera sees raw scene) ---
    table.log("Capturing baseline...")
    baseline = table.capture_baseline(region, settle_time=0.8)

    # Save debug images
    cv2.imwrite("debug_baseline.png", baseline)
    full_frame = table.get_frame()
    if full_frame is not None:
        cv2.imwrite("debug_full_frame.png", full_frame)
        y, x, h, w = region
        cv2.rectangle(full_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite("debug_frame_with_region.png", full_frame)
    table.log(f"Baseline saved. Region: {region}, shape: {baseline.shape if baseline is not None else None}")

    # Re-show the guide circle (capture_baseline hid it temporarily)
    canvas.circle(500, 500, CIRCLE_RADIUS, CIRCLE_COLOR, thickness=4)

    if baseline is None:
        table.notify("Could not capture baseline — paint program ending.")
        canvas.clear()
        table.stop()

if not table.stopped:
    # --- Step 3: Wait for object (change detection vs baseline) ---
    table.log("Waiting for object placement...")
    detected = table.wait_for_object_in_region(
        region, timeout=30.0, check_interval=0.5, baseline=baseline
    )

    if detected:
        # Save what triggered detection
        detect_frame = table.get_frame()
        if detect_frame is not None:
            y, x, h, w = region
            roi = cv2.cvtColor(detect_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(roi, baseline)
            cv2.imwrite("debug_detect_frame.png", detect_frame)
            cv2.imwrite("debug_detect_roi.png", roi)
            cv2.imwrite("debug_detect_diff.png", diff)
            changed = np.count_nonzero(diff > 30)
            ratio = changed / max(1, diff.size)
            table.log(f"Detection triggered: change_ratio={ratio:.3f}, changed_pixels={changed}/{diff.size}")
    else:
        table.notify("No object detected — paint program ending.")
        canvas.clear()
        table.stop()

if not table.stopped:
    # --- Step 4: Wait for hands to clear ---
    table.log("Object detected, waiting for hands to clear...")
    hands_clear = table.wait_for_hands_clear(
        region, timeout=15.0, stable_time=1.0, check_interval=0.2
    )

    if not hands_clear:
        table.notify("Hands didn't clear — paint program ending.")
        canvas.clear()
        table.stop()

if not table.stopped:
    # --- Step 5: Init tracking + flash confirmation ---
    # init_color_tracking hides projection briefly to get a clean capture
    table.log("Initializing color tracking...")
    obj_color = table.init_color_tracking("paint_obj", region)
    if obj_color is None:
        obj_color = (0, 200, 200)  # fallback cyan

    table.log(f"Tracked color: {obj_color}")

    # Flash the circle twice + beep
    for _ in range(2):
        canvas.clear()
        canvas.circle(500, 500, CIRCLE_RADIUS, FLASH_COLOR, thickness=-1)
        table.play_tone(TONE_FREQ, TONE_DURATION)
        time.sleep(FLASH_DURATION)
        canvas.clear()
        time.sleep(FLASH_DURATION)

    # Circle disappears — ready to paint
    canvas.clear()
    table.notify("Tracking started! Move the object to paint.")

    # --- Step 6: Paint loop ---
    last_pos = None
    last_visible_time = time.time()
    countdown_active = False
    countdown_start = None
    paint_color = tuple(max(50, c) for c in obj_color)  # ensure visible on projector

    while not table.stopped:
        info = table.get_tracked("paint_obj")

        if info and info["visible"]:
            last_visible_time = time.time()

            # Cancel any active countdown
            if countdown_active:
                countdown_active = False
                countdown_start = None
                table.log("Object returned — countdown cancelled.")

            cy, cx = info["center"]  # normalised 0-1000

            # Brush size from object bbox
            obj_size = table.get_object_size("paint_obj")
            if obj_size:
                brush_radius = max(5, int((obj_size[0] + obj_size[1]) / 4))
            else:
                brush_radius = 15

            # Paint if moved enough
            if last_pos is None:
                canvas.stamp(cy, cx, brush_radius, paint_color)
                last_pos = (cy, cx)
            else:
                dy = cy - last_pos[0]
                dx = cx - last_pos[1]
                dist = math.sqrt(dy * dy + dx * dx)
                if dist >= PAINT_TRAIL_MIN_DIST:
                    # Interpolate stamps for smooth lines
                    steps = max(1, int(dist / PAINT_TRAIL_MIN_DIST))
                    for s in range(1, steps + 1):
                        t = s / steps
                        sy = last_pos[0] + dy * t
                        sx = last_pos[1] + dx * t
                        canvas.stamp(sy, sx, brush_radius, paint_color)
                    last_pos = (cy, cx)
        else:
            # --- Step 7: Object not visible — countdown logic ---
            elapsed_missing = time.time() - last_visible_time

            if elapsed_missing >= MISSING_TIMEOUT and not countdown_active:
                countdown_active = True
                countdown_start = time.time()
                table.notify("Object lost! Counting down to clear...")

            if countdown_active:
                elapsed_countdown = time.time() - countdown_start
                remaining = COUNTDOWN_SECONDS - elapsed_countdown

                if remaining <= 0:
                    canvas.clear()
                    table.notify("Object gone. Canvas cleared. Paint program ended.")
                    table.stop()
                    break
                else:
                    # Show countdown number at center
                    canvas.text(str(int(remaining) + 1), 480, 460,
                                (255, 255, 255), scale=3.0, thickness=4)

        time.sleep(0.05)  # ~20 FPS check rate
