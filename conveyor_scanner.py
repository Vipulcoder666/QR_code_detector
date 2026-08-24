"""
Conveyor Belt QR Scanner — Real-Time Pipeline
===============================================
Purpose-built for scanning QR codes on products moving on a conveyor belt.

ARCHITECTURE:
  Camera → Frame Grab → Detection Zone Crop → Fast QR Locator
    → ROI Crop → Fast Decode → Product Tracker → CSV Log (unique only)

KEY DESIGN DECISIONS:
  - NO multi-frame fusion (moving QR shifts between frames, averaging blurs)
  - NO perspective correction (camera is mounted above, square-on)
  - NO ThreadPoolExecutor (single-thread is faster for 1-2 small ROIs)
  - NO heavy preprocessing variants (only raw + sharpen, 2 attempts max)
  - Locator finds QR position first, decoder only runs on the small crop
  - Product tracker prevents duplicate logging of the same box

USAGE:
  python conveyor_scanner.py                                    # laptop webcam
  python conveyor_scanner.py 0                                  # webcam index 0
  python conveyor_scanner.py "http://192.168.1.54:8080/video"   # phone camera
  python conveyor_scanner.py "rtsp://user:pass@ip:554/stream"   # CCTV/IP camera

CONTROLS:
  q     = quit
  z/x   = shrink/expand detection zone width
  a/s   = shrink/expand detection zone height
"""

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import threading
import time
import csv
import sys
import winsound
from datetime import datetime
from collections import deque

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False
    print("[WARN] zxing-cpp not installed. pip install zxing-cpp")

try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAS_ZBAR = True
except Exception:
    HAS_ZBAR = False


# =============================================================================
# CONFIG
# =============================================================================

DISPLAY_WIDTH = 1280
CSV_FILE = "scanned_products.csv"

# Detection zone: fraction of the frame to scan (center crop).
# Narrower = faster processing. Wider = catches codes at edges.
DETECT_ZONE_X = 0.20  # left/right margin (0.20 = scan center 60% width)
DETECT_ZONE_Y = 0.15  # top/bottom margin (0.15 = scan center 70% height)

# Product tracker settings
DEDUP_TIME_WINDOW = 5.0     # seconds to remember a scanned product
DEDUP_POSITION_THRESHOLD = 0.30  # fraction of QR width; if center moved less, same product

# Locator downscale target (pixels wide) — lower = faster locator
LOCATOR_WIDTH = 640


# =============================================================================
# 1. STREAM READER (Camera Thread)
# =============================================================================

class ConveyorStreamReader:
    """Threaded camera reader. Supports webcam, phone (MJPEG), and CCTV (RTSP)."""

    def __init__(self, source):
        self.source = source
        self.frame = None
        self.ret = False
        self.stopped = False
        self.stream_res = (0, 0)
        self.source_type = "?"
        self.lock = threading.Lock()

    def start(self):
        self.stopped = False
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return self

    def _capture_loop(self):
        src = self.source
        is_webcam = isinstance(src, int) or (isinstance(src, str) and src.isdigit())
        is_phone = isinstance(src, str) and "8080" in src

        if is_webcam:
            self.source_type = "Webcam"
            cap_src = int(src) if isinstance(src, str) else src
        elif is_phone:
            self.source_type = "Phone"
            cap_src = src
        else:
            self.source_type = "CCTV"
            cap_src = src

        while not self.stopped:
            if self.source_type == "CCTV":
                cap = cv2.VideoCapture(cap_src, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(cap_src)

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                print(f"[STREAM] Failed to open {self.source_type} camera, retrying...")
                time.sleep(2)
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.stream_res = (w, h)
            print(f"[STREAM] >>> {self.source_type} Connected {w}x{h} <<<")

            while not self.stopped:
                if self.source_type == "CCTV":
                    # Flush FFMPEG buffer: grab 3 frames, only decode the last
                    for _ in range(3):
                        if not cap.grab():
                            break
                    ok, frame = cap.retrieve()
                else:
                    ok, frame = cap.read()

                if not ok or frame is None:
                    print(f"[STREAM] {self.source_type} dropped, reconnecting...")
                    break

                with self.lock:
                    self.frame = frame
                    self.ret = True
                    self.stream_res = (frame.shape[1], frame.shape[0])

            cap.release()
            time.sleep(1)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True


# =============================================================================
# 2. FAST QR LOCATOR (Finder Pattern Detector — <5ms)
# =============================================================================

class FastQRLocator:
    """
    Finds QR code bounding boxes using contour nesting (finder pattern geometry).
    Does NOT decode — just locates where the QR codes are in the frame.
    Runs in <5ms by downscaling to 640px width.
    """

    def locate(self, gray):
        """
        Returns list of (x, y, w, h) bounding boxes in original image coordinates.
        Sorted by area descending. Max 3 candidates.
        """
        h, w = gray.shape[:2]
        scale = 1.0

        # Downscale for speed
        if w > LOCATOR_WIDTH:
            scale = w / LOCATOR_WIDTH
            small = cv2.resize(gray, (LOCATOR_WIDTH, int(h / scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = gray

        # Otsu threshold — fast, works well on high-contrast QR codes
        _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []

        hierarchy = hierarchy[0]
        patterns = []

        for i in range(len(contours)):
            # QR finder pattern = nested contour depth >= 2
            child1 = hierarchy[i][2]
            if child1 == -1:
                continue
            child2 = hierarchy[child1][2]
            if child2 == -1:
                continue

            x, y, cw, ch = cv2.boundingRect(contours[i])
            if cw < 6 or ch < 6:
                continue

            aspect = cw / ch
            if not (0.7 < aspect < 1.4):
                continue

            cx = (x + cw / 2.0) * scale
            cy = (y + ch / 2.0) * scale
            size = max(cw, ch) * scale
            patterns.append((cx, cy, size))

        if not patterns:
            return []

        # Group nearby patterns into QR code regions
        return self._group_patterns(patterns, w, h)

    def _group_patterns(self, patterns, img_w, img_h):
        """Group spatially close finder patterns into QR bounding boxes."""
        used = [False] * len(patterns)
        regions = []

        for i in range(len(patterns)):
            if used[i]:
                continue
            group = [patterns[i]]
            used[i] = True
            cx1, cy1, s1 = patterns[i]

            for j in range(i + 1, len(patterns)):
                if used[j]:
                    continue
                cx2, cy2, s2 = patterns[j]
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < max(s1, s2) * 12.0:
                    group.append(patterns[j])
                    used[j] = True

            # Bounding box of the group with padding
            min_x = min(p[0] - p[2] / 2 for p in group)
            max_x = max(p[0] + p[2] / 2 for p in group)
            min_y = min(p[1] - p[2] / 2 for p in group)
            max_y = max(p[1] + p[2] / 2 for p in group)

            gw = max_x - min_x
            gh = max_y - min_y

            # Padding: 40% for groups, 3x size for single patterns
            if len(group) > 1:
                pad = 0.4
            else:
                pad = 2.0

            px = int(gw * pad)
            py = int(gh * pad)

            rx = int(max(0, min_x - px))
            ry = int(max(0, min_y - py))
            rw = int(min(img_w - rx, gw + 2 * px))
            rh = int(min(img_h - ry, gh + 2 * py))

            if rw > 20 and rh > 20:
                regions.append((rx, ry, rw, rh))

        # Sort by area descending, return top 3
        regions.sort(key=lambda r: -(r[2] * r[3]))
        return regions[:3]


# =============================================================================
# 3. FAST DECODER (Lightweight — <15ms per ROI)
# =============================================================================

class FastDecoder:
    """
    Lightweight QR decoder. Only 2 attempts per ROI:
      1. Raw grayscale
      2. Sharpened + upscaled 2x (only if raw fails)
    """

    def __init__(self):
        self.cv2_qr = cv2.QRCodeDetector()

    def decode(self, gray_roi):
        """
        Decode a grayscale ROI image.
        Returns (text, pts_in_roi) or (None, None).
        """
        # Attempt 1: Raw
        text, pts = self._try_decode(gray_roi)
        if text:
            return text, pts

        # Attempt 2: Sharpen + upscale 2x (only for small/blurry ROIs)
        h, w = gray_roi.shape[:2]
        if h * w < 4_000_000:  # don't upscale huge regions
            up = cv2.resize(gray_roi, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            up_sharp = cv2.filter2D(up, -1, kernel)

            text, pts = self._try_decode(up_sharp)
            if text:
                # Scale points back to original ROI coordinates
                if pts is not None:
                    pts = (pts.astype(np.float64) / 2.0).astype(np.int32)
                return text, pts

        return None, None

    def _try_decode(self, img):
        """Run decoder cascade on a single image."""

        # 1. zxing-cpp (fastest and most robust)
        if HAS_ZXING:
            try:
                results = zxingcpp.read_barcodes(img, try_rotate=True, try_invert=True)
                for b in results:
                    if b.text and b.position:
                        p = b.position
                        pts = np.array([
                            [p.top_left.x, p.top_left.y],
                            [p.top_right.x, p.top_right.y],
                            [p.bottom_right.x, p.bottom_right.y],
                            [p.bottom_left.x, p.bottom_left.y],
                        ], dtype=np.int32)
                        return b.text, pts
            except Exception:
                pass

        # 2. pyzbar
        if HAS_ZBAR:
            try:
                for r in zbar_decode(img):
                    if r.data:
                        pts = np.array([[p.x, p.y] for p in r.polygon], dtype=np.int32)
                        if len(pts) == 4:
                            return r.data.decode("utf-8", errors="replace"), pts
            except Exception:
                pass

        # 3. OpenCV fallback
        try:
            text, points, _ = self.cv2_qr.detectAndDecode(img)
            if text and points is not None and len(points) > 0:
                return text, points[0].astype(np.int32)
        except Exception:
            pass

        return None, None


# =============================================================================
# 4. PRODUCT TRACKER (Deduplication Engine)
# =============================================================================

class ProductTracker:
    """
    Tracks scanned products to prevent duplicate logging.

    A product is considered "the same" if:
      - QR data matches a recent scan
      - Center position hasn't jumped far (still the same box moving)
      - Within the dedup time window

    A product is considered "new" if:
      - QR data is different from all recent scans
      - OR the center position has jumped significantly (new box with same label)
    """

    def __init__(self):
        self.recent_scans = []  # list of {data, cx, cy, qr_w, time}
        self.total_unique = 0

    def is_new_product(self, qr_data, center_x, center_y, qr_width):
        """Returns True if this is a new product that should be logged."""
        now = time.time()

        # Expire old scans
        self.recent_scans = [
            s for s in self.recent_scans
            if now - s["time"] < DEDUP_TIME_WINDOW
        ]

        # Check against recent scans
        for scan in self.recent_scans:
            if scan["data"] != qr_data:
                continue

            # Same data — check if it's the same physical product still moving
            if qr_width > 0:
                dx = abs(center_x - scan["cx"])
                threshold = qr_width * DEDUP_POSITION_THRESHOLD
                if dx < threshold:
                    # Same data, hasn't moved far -> same product, update position
                    scan["cx"] = center_x
                    scan["cy"] = center_y
                    scan["time"] = now
                    return False

            # Same data but position jumped -> could be new product with same label
            # Only consider it new if significant horizontal displacement
            frame_jump = abs(center_x - scan["cx"])
            if qr_width > 0 and frame_jump < qr_width * 2.0:
                # Not far enough — still same product
                scan["cx"] = center_x
                scan["cy"] = center_y
                scan["time"] = now
                return False

        # New product!
        self.recent_scans.append({
            "data": qr_data,
            "cx": center_x,
            "cy": center_y,
            "qr_w": qr_width,
            "time": now,
        })
        self.total_unique += 1
        return True


# =============================================================================
# 5. CONVEYOR SCANNER (Threaded Detection + Display)
# =============================================================================

class ConveyorScanner:
    """
    Two-thread architecture:
      - DISPLAY thread (main): reads frames, draws HUD, shows video at full FPS
      - DETECTION thread (background): locates + decodes QR, never blocks display
    """

    def __init__(self, source):
        self.reader = ConveyorStreamReader(source)
        self.locator = FastQRLocator()
        self.decoder = FastDecoder()
        self.tracker = ProductTracker()

        # Detection zone margins (fraction of frame)
        self.zone_x = DETECT_ZONE_X
        self.zone_y = DETECT_ZONE_Y

        # Shared state between detection thread and display thread
        self._det_lock = threading.Lock()
        self._det_frame = None          # latest frame for detection thread
        self._det_result = {            # latest detection result
            "detected": False,
            "text": None,
            "pts": None,
            "pw": 0.0,
        }

        # Display state (only touched by main thread)
        self.last_decode_text = ""
        self.last_decode_time = 0.0
        self.flash_until = 0.0
        self.fps = 0.0
        self._stopped = False

    def _detection_loop(self):
        """Background thread: continuously locate + decode QR codes."""
        while not self._stopped:
            # Grab the latest frame
            with self._det_lock:
                frame = self._det_frame
                zone_x = self.zone_x
                zone_y = self.zone_y

            if frame is None:
                time.sleep(0.005)
                continue

            frame_h, frame_w = frame.shape[:2]

            # Calculate detection zone
            zx1 = int(frame_w * zone_x)
            zx2 = int(frame_w * (1.0 - zone_x))
            zy1 = int(frame_h * zone_y)
            zy2 = int(frame_h * (1.0 - zone_y))

            zone = frame[zy1:zy2, zx1:zx2]
            zone_gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

            # Fast QR Locator
            candidates = self.locator.locate(zone_gray)

            detected = False
            text = None
            pts_full = None
            pw = 0.0

            for (rx, ry, rw, rh) in candidates:
                roi = zone_gray[ry:ry + rh, rx:rx + rw]
                if roi.size == 0:
                    continue

                t, pts_in_roi = self.decoder.decode(roi)
                if not t:
                    continue

                # Map points to full frame
                if pts_in_roi is not None:
                    pf = pts_in_roi.copy()
                    pf[:, 0] += rx + zx1
                    pf[:, 1] += ry + zy1
                else:
                    pf = None

                # QR pixel width
                p = 0.0
                if pf is not None and len(pf) >= 4:
                    a = np.linalg.norm(pf[0].astype(float) - pf[1].astype(float))
                    b = np.linalg.norm(pf[2].astype(float) - pf[3].astype(float))
                    p = (a + b) / 2.0

                center_x = rx + rw / 2.0 + zx1
                center_y = ry + rh / 2.0 + zy1

                is_new = self.tracker.is_new_product(t, center_x, center_y, p)

                if is_new:
                    self._log_product(t, p)
                    now = time.time()
                    self.flash_until = now + 0.8
                    self.last_decode_text = t
                    self.last_decode_time = now
                    print(f"\n[NEW SCAN #{self.tracker.total_unique}] {t}")
                    print(f"  QR Width: {p:.0f}px")
                    try:
                        winsound.Beep(1200, 100)
                    except Exception:
                        pass

                detected = True
                text = t
                pts_full = pf
                pw = p
                self.last_decode_text = t
                self.last_decode_time = time.time()
                break

            # Publish result
            with self._det_lock:
                self._det_result = {
                    "detected": detected,
                    "text": text,
                    "pts": pts_full,
                    "pw": pw,
                }
                # Clear the frame so we don't re-process the same one
                self._det_frame = None

    def run(self):
        self.reader.start()

        # Start detection thread
        det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        det_thread.start()

        win = "Conveyor QR Scanner"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, DISPLAY_WIDTH, int(DISPLAY_WIDTH * 9 / 16))

        # Ensure CSV exists
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Timestamp", "QR_Data", "PixelWidth", "Status"])

        print("\n" + "=" * 60)
        print("CONVEYOR QR SCANNER - Real-Time Pipeline")
        print("  q = quit")
        print("  z/x = shrink/expand detection zone width")
        print("  a/s = shrink/expand detection zone height")
        print("=" * 60 + "\n")

        fps_counter = 0
        fps_timer = time.time()

        while True:
            ok, raw = self.reader.read()
            if not ok or raw is None:
                blank = np.full((720, 1280, 3), 30, np.uint8)
                cv2.putText(blank, "Waiting for camera...", (60, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.imshow(win, blank)
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                continue

            now = time.time()
            frame_h, frame_w = raw.shape[:2]

            # Feed frame to detection thread (non-blocking)
            with self._det_lock:
                self._det_frame = raw.copy()
                det = self._det_result.copy()

            # Detection zone coordinates (for HUD drawing only)
            zx1 = int(frame_w * self.zone_x)
            zx2 = int(frame_w * (1.0 - self.zone_x))
            zy1 = int(frame_h * self.zone_y)
            zy2 = int(frame_h * (1.0 - self.zone_y))

            # --- FPS ---
            fps_counter += 1
            if now - fps_timer >= 1.0:
                self.fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # --- Draw HUD (using latest detection result) ---
            display = raw.copy()
            self._draw_hud(display, frame_w, frame_h,
                           zx1, zy1, zx2, zy2,
                           det["detected"], det["text"],
                           det["pts"], det["pw"], now)

            # Resize for display
            if frame_w > DISPLAY_WIDTH:
                display = cv2.resize(display, (DISPLAY_WIDTH,
                                               int(frame_h * DISPLAY_WIDTH / frame_w)),
                                     interpolation=cv2.INTER_AREA)

            cv2.imshow(win, display)

            # --- Keyboard controls ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z"):
                self.zone_x = min(0.45, self.zone_x + 0.02)
            elif key == ord("x"):
                self.zone_x = max(0.0, self.zone_x - 0.02)
            elif key == ord("a"):
                self.zone_y = min(0.45, self.zone_y + 0.02)
            elif key == ord("s"):
                self.zone_y = max(0.0, self.zone_y - 0.02)

        self._stopped = True
        self.reader.stop()
        cv2.destroyAllWindows()
        os._exit(0)

    def _log_product(self, text, pixel_width):
        """Append a unique product to CSV."""
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                text,
                int(pixel_width),
                "SCANNED"
            ])

    def _draw_hud(self, frame, fw, fh, zx1, zy1, zx2, zy2,
                  detected, text, pts, pw, now):
        """Draw the scanner HUD overlay."""

        # --- Detection zone rectangle ---
        is_flash = now < self.flash_until
        zone_color = (0, 255, 0) if is_flash else (0, 180, 0)
        zone_thickness = 3 if is_flash else 1
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), zone_color, zone_thickness)

        # Zone label
        cv2.putText(frame, "DETECTION ZONE", (zx1 + 5, zy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_color, 1)

        # --- Top HUD bar ---
        cv2.rectangle(frame, (0, 0), (fw, 70), (20, 20, 20), -1)

        # Status
        if detected:
            cv2.putText(frame, "QR DETECTED", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SCANNING...", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (fw - 150, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Total unique scans
        cv2.putText(frame, f"SCANNED: {self.tracker.total_unique}", (fw - 300, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        # Stream resolution
        sr = self.reader.stream_res
        cv2.putText(frame, f"RES: {sr[0]}x{sr[1]} | {self.reader.source_type}",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Last scanned data
        if self.last_decode_text and (now - self.last_decode_time < 5.0):
            cv2.putText(frame, f"LAST: {self.last_decode_text[:50]}",
                        (15, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # --- QR code overlay ---
        if detected and pts is not None and len(pts) >= 4:
            # Green polygon around QR
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            for p in pts:
                cv2.circle(frame, tuple(p), 5, (0, 0, 255), -1)

            # Data label above QR
            label_y = max(90, int(pts[:, 1].min()) - 10)
            label_x = max(5, int(pts[:, 0].min()))
            cv2.rectangle(frame, (label_x - 3, label_y - 22),
                         (label_x + 400, label_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, f"{text[:40]}", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # --- NEW SCAN flash banner ---
        if is_flash:
            cv2.rectangle(frame, (0, fh // 2 - 30), (fw, fh // 2 + 30), (0, 180, 0), -1)
            cv2.putText(frame, f"NEW PRODUCT SCANNED: {self.last_decode_text[:40]}",
                        (20, fh // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # Check if it's a webcam index
        if source.isdigit():
            source = int(source)
    else:
        source = 0  # default to laptop webcam

    scanner = ConveyorScanner(source)
    scanner.run()


if __name__ == "__main__":
    main()
