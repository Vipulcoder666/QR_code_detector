import os
# IMPORTANT: Set FFMPEG options BEFORE importing cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import threading
import time
import zxingcpp
import csv
import sys
from datetime import datetime

RTSP_CANDIDATE_URLS = [
    # Phone Camera (IP Webcam Pro app)
    "http://192.168.1.54:8080/video",
    # CCTV IP Cameras
    "rtsp://admin:Smarden%4012@192.168.1.43:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif",
    "rtsp://admin:Smarden%4012@192.168.1.43:554/video/live?channel=1&subtype=1&unicast=true&proto=Onvif",
    "rtsp://admin:Smarden%4012@192.168.1.49:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif",
    "rtsp://admin:Smarden%4012@192.168.1.120:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif"
]

# --- Calibration Settings ---
KNOWN_WIDTH_MM = 50.0
FOCAL_LENGTH = 1200.0

# --- Display Settings ---
DISPLAY_WIDTH = 1280  # Change this to adjust the window size (e.g., 640, 800, 1280)


# =============================================================================
# RTSP Stream Reader (Background Thread – always grabs latest frame)
# =============================================================================
class RTSPStreamReader:
    def __init__(self, rtsp_urls):
        self.rtsp_urls = rtsp_urls if isinstance(rtsp_urls, list) else [rtsp_urls]
        self.active_url = self.rtsp_urls[0]
        self.frame = None
        self.ret = False
        self.stopped = False
        self.connected = False
        self.status_message = "Connecting to camera..."
        self.lock = threading.Lock()

    def start(self):
        self.stopped = False
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        url_idx = 0
        import urllib.request
        
        while not self.stopped:
            url = self.rtsp_urls[url_idx % len(self.rtsp_urls)]
            self.active_url = url
            self.status_message = f"Trying: {url.split('@')[-1]}"
            print(f"\n[STREAM] Connecting to: {url}")

            is_ip_webcam = "8080" in url
            self.source_type = "Phone Camera" if is_ip_webcam else "CCTV"

            if is_ip_webcam:
                # ZERO-LATENCY MODE FOR PHONE CAMERA (IP WEBCAM)
                # Bypasses OpenCV video buffer entirely by grabbing live snapshots
                snapshot_url = url.replace("/video", "/shot.jpg")
                print(f"[STREAM] >>> Connected to {self.source_type} (Zero Latency Mode) <<<")
                self.connected = True
                
                while not self.stopped:
                    try:
                        req = urllib.request.urlopen(snapshot_url, timeout=2.0)
                        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                        frame = cv2.imdecode(arr, -1)
                        if frame is not None:
                            with self.lock:
                                self.frame = frame
                                self.ret = True
                    except Exception as e:
                        print(f"[STREAM] Phone stream dropped: {e}")
                        self.connected = False
                        break
            else:
                # CCTV RTSP MODE
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG if url.startswith("rtsp") else cv2.CAP_ANY)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if not cap.isOpened():
                    print(f"[STREAM] Failed: {url}")
                    url_idx += 1
                    time.sleep(1.5)
                    continue

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[STREAM] >>> Connected to {self.source_type} ({w}x{h}) <<<")
                self.connected = True

                while not self.stopped:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("[STREAM] Dropped. Reconnecting...")
                        self.connected = False
                        break
                    
                    with self.lock:
                        self.frame = frame
                        self.ret = True

                cap.release()
                
            url_idx += 1
            time.sleep(1)

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.stopped = True


# =============================================================================
# QR Detection Worker (Runs in separate thread — does NOT block video)
# =============================================================================
class QRDetectionWorker:
    """
    Runs heavy multi-pass QR detection in a background thread.
    The main loop feeds it frames and reads back results without blocking.
    """
    def __init__(self):
        self.input_frame = None
        self.result_pts = None
        self.result_data = None
        self.result_pixel_width = 0.0
        self.result_mode = "none"
        self.result_time = 0
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stopped = False
        self.cv2_qr = cv2.QRCodeDetector()

    def start(self):
        self.stopped = False
        threading.Thread(target=self._worker, daemon=True).start()
        return self

    def submit_frame(self, frame):
        """Submit a new frame for detection (non-blocking, drops old frames)."""
        with self.lock:
            self.input_frame = frame.copy()
        self.new_frame_event.set()

    def get_result(self):
        """Get the latest detection result (non-blocking)."""
        with self.lock:
            return (self.result_pts, self.result_data,
                    self.result_pixel_width, self.result_mode, self.result_time)

    def _worker(self):
        while not self.stopped:
            self.new_frame_event.wait(timeout=0.5)
            self.new_frame_event.clear()

            with self.lock:
                frame = self.input_frame
                self.input_frame = None

            if frame is None:
                continue

            # Resize for processing if frame is very large (e.g. 4K/2K CCTV)
            h, w = frame.shape[:2]
            process_scale = 1.0
            if w > 1920:
                process_scale = 1920.0 / w
                proc_frame = cv2.resize(frame, (1920, int(h * process_scale)),
                                        interpolation=cv2.INTER_LINEAR)
            else:
                proc_frame = frame

            # Run multi-pass detection
            pts, data, pw, mode = self._detect(proc_frame)

            # Scale coordinates back to original frame size
            if pts is not None and process_scale != 1.0:
                pts = (pts.astype(np.float64) / process_scale).astype(np.int32)
                pw = pw / process_scale

            with self.lock:
                self.result_pts = pts
                self.result_data = data
                self.result_pixel_width = pw
                self.result_mode = mode
                self.result_time = time.time()

    def _detect(self, frame):
        """Multi-pass + tiled detection optimized for CCTV."""
        h, w = frame.shape[:2]

        # === STAGE 1: Full frame quick passes ===
        pts, data, pw, mode = self._scan_image(frame, "full")
        if pts is not None:
            return pts, data, pw, mode

        # === STAGE 2: Center 60% crop (most common QR position) ===
        cy1, cy2 = int(h * 0.2), int(h * 0.8)
        cx1, cx2 = int(w * 0.2), int(w * 0.8)
        center = frame[cy1:cy2, cx1:cx2]
        pts, data, pw, mode = self._scan_image(center, "center")
        if pts is not None:
            pts[:, 0] += cx1
            pts[:, 1] += cy1
            return pts, data, pw, mode

        # === STAGE 3: 2×2 quadrant scan with upscaling ===
        mid_h, mid_w = h // 2, w // 2
        pad_y, pad_x = int(mid_h * 0.15), int(mid_w * 0.15)
        quadrants = [
            (0,            max(0, mid_h + pad_y), 0,            max(0, mid_w + pad_x), "TL"),
            (0,            max(0, mid_h + pad_y), max(0, mid_w - pad_x), w,            "TR"),
            (max(0, mid_h - pad_y), h,            0,            max(0, mid_w + pad_x), "BL"),
            (max(0, mid_h - pad_y), h,            max(0, mid_w - pad_x), w,            "BR"),
        ]
        for y1, y2, x1, x2, label in quadrants:
            quad = frame[y1:y2, x1:x2]
            pts, data, pw, mode = self._scan_image(quad, f"quad_{label}", upscale=True)
            if pts is not None:
                pts[:, 0] += x1
                pts[:, 1] += y1
                return pts, data, pw, mode

        return None, None, 0.0, "none"

    def _scan_image(self, img, label, upscale=False):
        """
        Try multiple preprocessing variants on a single image region.
        Returns (pts, data, pixel_width, mode) or (None, None, 0, mode).
        """
        variants = []

        # 1. Raw
        variants.append((img, f"{label}_raw"))

        # Convert to gray once
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # 2. CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append((clahe_img, f"{label}_clahe"))

        # 3. Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        variants.append((sharp, f"{label}_sharp"))

        # 4. Adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 5)
        variants.append((thresh, f"{label}_thresh"))

        # 5. Upscaled versions (critical for CCTV — small QR codes)
        if upscale:
            h2, w2 = gray.shape[:2]
            up2 = cv2.resize(clahe_img, (w2 * 2, h2 * 2), interpolation=cv2.INTER_CUBIC)
            variants.append((up2, f"{label}_up2x_clahe"))

            up2_sharp = cv2.filter2D(up2, -1, kernel)
            variants.append((up2_sharp, f"{label}_up2x_sharp"))

        for variant_img, mode_name in variants:
            pts, data, pw = self._try_decode(variant_img, mode_name, upscale and "up2x" in mode_name)
            if pts is not None:
                return pts, data, pw, mode_name

        return None, None, 0.0, f"{label}_none"

    def _try_decode(self, img, mode_name, was_upscaled):
        """Try zxing-cpp then OpenCV QR detector. Returns (pts, data, pixel_width) or (None, None, 0)."""
        scale_back = 2.0 if was_upscaled else 1.0

        # zxing-cpp
        try:
            barcodes = zxingcpp.read_barcodes(img)
            for b in barcodes:
                pos = b.position
                if pos and b.text:
                    pts = np.array([
                        [pos.top_left.x, pos.top_left.y],
                        [pos.top_right.x, pos.top_right.y],
                        [pos.bottom_right.x, pos.bottom_right.y],
                        [pos.bottom_left.x, pos.bottom_left.y]
                    ], dtype=np.float64)
                    if was_upscaled:
                        pts /= scale_back
                    pts = pts.astype(np.int32)
                    tw = np.linalg.norm(pts[0].astype(float) - pts[1].astype(float))
                    bw = np.linalg.norm(pts[2].astype(float) - pts[3].astype(float))
                    pw = (tw + bw) / 2.0
                    return pts, b.text, pw
        except Exception:
            pass

        # OpenCV fallback
        try:
            data, points, _ = self.cv2_qr.detectAndDecode(img)
            if data and points is not None and len(points) > 0:
                pts = points[0].astype(np.float64)
                if was_upscaled:
                    pts /= scale_back
                pts = pts.astype(np.int32)
                tw = np.linalg.norm(pts[0].astype(float) - pts[1].astype(float))
                bw = np.linalg.norm(pts[2].astype(float) - pts[3].astype(float))
                pw = (tw + bw) / 2.0
                return pts, data, pw
        except Exception:
            pass

        return None, None, 0.0

    def stop(self):
        self.stopped = True


# =============================================================================
# Loading Screen
# =============================================================================
def create_loading_screen(status_text):
    s = np.zeros((720, 1280, 3), dtype=np.uint8)
    s[:] = (30, 30, 30)
    cv2.putText(s, "Connecting to CCTV Camera...", (380, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
    cv2.putText(s, status_text, (200, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(s, "Press 'q' to abort", (530, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
    return s


# =============================================================================
# Main Loop — Video stays smooth, detection runs in background
# =============================================================================
def main():
    global FOCAL_LENGTH

    urls = [sys.argv[1]] if len(sys.argv) > 1 else RTSP_CANDIDATE_URLS
    reader = RTSPStreamReader(urls).start()
    detector = QRDetectionWorker().start()

    win = "CCTV QR Scanner"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_WIDTH, int(DISPLAY_WIDTH * (9/16)))  # Maintain 16:9 aspect ratio

    # Digital zoom trackbar (1x – 4x)
    zoom_level = [1]
    cv2.createTrackbar("Zoom", win, 1, 4, lambda v: zoom_level.__setitem__(0, max(1, v)))

    print("=" * 60)
    print("CCTV QR Scanner – Threaded Detection (Smooth Video)")
    print("Controls: 'q' quit | 'c' calibrate | Zoom slider 1x-4x")
    print("=" * 60)

    CSV_FILE = "scanned_products.csv"
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Timestamp", "QR_Data", "Distance_meters"])

    last_printed_data = None
    last_print_time = 0
    fps_t = time.time()
    fps_cnt = 0
    fps = 0.0
    frame_submit_interval = 0.1   # Submit a frame for detection every 100ms
    last_submit_time = 0

    while True:
        try:
            ret, frame = reader.read()
            if not ret or frame is None:
                cv2.imshow(win, create_loading_screen(reader.status_message))
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
                continue

            now = time.time()

            # --- Digital zoom (center crop + resize back) ---
            z = zoom_level[0]
            if z > 1:
                fh, fw = frame.shape[:2]
                zh, zw = fh // z, fw // z
                zy1, zx1 = (fh - zh) // 2, (fw - zw) // 2
                frame = cv2.resize(frame[zy1:zy1+zh, zx1:zx1+zw],
                                   (fw, fh), interpolation=cv2.INTER_CUBIC)

            # Submit frame to detection worker periodically (not every frame)
            if now - last_submit_time >= frame_submit_interval:
                detector.submit_frame(frame)
                last_submit_time = now

            # Read latest detection result (non-blocking)
            pts, detected_data, pixel_width, match_mode, detect_time = detector.get_result()

            # FPS counter
            fps_cnt += 1
            if now - fps_t >= 1.0:
                fps = fps_cnt / (now - fps_t)
                fps_cnt = 0
                fps_t = now

            # --- HUD ---
            h, w = frame.shape[:2]
            is_det = (pts is not None and (now - detect_time < 1.0))

            # Top bar
            cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
            badge_col = (0, 200, 0) if is_det else (0, 140, 255)
            badge_txt = "QR DETECTED" if is_det else "SCANNING..."
            cv2.putText(frame, f"STATUS: {badge_txt}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_col, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}  ZOOM: {z}x", (w - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Guide rectangle
            gx1, gy1 = int(w * 0.15), int(h * 0.1)
            gx2, gy2 = int(w * 0.85), int(h * 0.9)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (60, 60, 60), 1)
            if not is_det:
                cv2.putText(frame, "Hold QR code in view | Use Zoom slider to magnify",
                            (gx1, gy2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

            if is_det and pixel_width > 0:
                # Clamp pts to frame bounds
                pts_draw = pts.copy()
                pts_draw[:, 0] = np.clip(pts_draw[:, 0], 0, w - 1)
                pts_draw[:, 1] = np.clip(pts_draw[:, 1], 0, h - 1)

                # Bounding polygon + corner dots
                cv2.polylines(frame, [pts_draw], True, (0, 255, 0), 3)
                for pt in pts_draw:
                    cv2.circle(frame, tuple(pt), 6, (0, 0, 255), -1)

                dist_mm = (KNOWN_WIDTH_MM * FOCAL_LENGTH) / pixel_width
                dist_m = dist_mm / 1000.0

                # Info card
                my = max(55, int(min(pts_draw[:, 1])))
                mx = max(5, int(min(pts_draw[:, 0])))
                cv2.rectangle(frame, (mx - 5, my - 78), (mx + 420, my - 3), (0, 0, 0), -1)
                cv2.putText(frame, f"DATA: {detected_data[:32]}", (mx, my - 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(frame, f"DEPTH: {dist_m:.2f}m ({dist_mm:.0f}mm) | {int(pixel_width)}px",
                            (mx, my - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"MODE: {match_mode}", (mx, my - 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                # Console + CSV
                if detected_data != last_printed_data or (now - last_print_time > 1.0):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\nData: {detected_data}")
                    print(f"Distance: {dist_m:.2f} meters")
                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow([
                            timestamp, detected_data, f"{dist_m:.2f}"
                        ])
                    last_printed_data = detected_data
                    last_print_time = now

            # Display (resize if frame width is larger than DISPLAY_WIDTH for smooth rendering)
            if w > DISPLAY_WIDTH:
                dh = int(h * (float(DISPLAY_WIDTH) / w))
                display = cv2.resize(frame, (DISPLAY_WIDTH, dh))
            else:
                display = frame

            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and is_det:
                try:
                    print("\n--- Calibration Mode ---")
                    d = input("Enter actual distance from camera to QR code (cm): ")
                    FOCAL_LENGTH = (pixel_width * float(d) * 10.0) / KNOWN_WIDTH_MM
                    print(f"Calibrated! Focal Length = {FOCAL_LENGTH:.2f}\n")
                except Exception as e:
                    print("Calibration skipped:", e)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.05)

    detector.stop()
    reader.stop()
    cv2.destroyAllWindows()
    os._exit(0)  # Clean exit — suppresses nanobind leak warnings


if __name__ == "__main__":
    main()
