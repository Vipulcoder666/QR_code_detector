

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import threading
import time
import csv
import sys
from collections import deque
from datetime import datetime

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

# IMPORTANT: subtype=0 is the MAIN stream (full resolution).
# subtype=1 is the SUB stream (often 704x480) - do not use it for scanning.
RTSP_CANDIDATE_URLS = [
    "http://192.168.1.54:8080/video",
    "rtsp://admin:Smarden%4012@192.168.1.43:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif",
    "rtsp://admin:Smarden%4012@192.168.1.49:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif",
    "rtsp://admin:Smarden%4012@192.168.1.120:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif",
]

WECHAT_MODEL_DIR = "wechat_models"
DISPLAY_WIDTH = 1280  # Use 640 for half screen (phone), 1280 for full (CCTV)
FRAME_BUFFER_SIZE = 12       # how many recent frames to keep for best-of retry
FOCUS_SHARP_THRESHOLD = 200  # Laplacian variance above this = sharp
FOCUS_SOFT_THRESHOLD = 50    # below this = badly blurred


# =============================================================================
# Focus measurement - the key diagnostic
# =============================================================================

def focus_score(gray):
    """
    Laplacian variance. High = sharp edges present. Low = blurred.
    This is how you tell 'out of focus' apart from 'too few pixels'.
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# =============================================================================
# Stream Reader
# =============================================================================

class StreamReader:
    def __init__(self, urls):
        self.urls = urls if isinstance(urls, list) else [urls]
        self.frame = None
        self.ret = False
        self.stopped = False
        self.status_message = "Connecting..."
        self.stream_res = (0, 0)
        self.source_type = "?"
        self.lock = threading.Lock()

    def start(self):
        self.stopped = False
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        import urllib.request
        url_idx = 0

        while not self.stopped:
            url = self.urls[url_idx % len(self.urls)]
            self.status_message = f"Trying: {url.split('@')[-1][:50]}"
            print(f"\n[STREAM] Connecting: {url}")

            # Detect camera type
            is_webcam = url.isdigit()
            is_phone = not is_webcam and "8080" in url

            if is_webcam:
                self.source_type = "Webcam"
            elif is_phone:
                self.source_type = "Phone"
            else:
                self.source_type = "CCTV"

            if is_webcam:
                # Local USB/laptop webcam
                cap = cv2.VideoCapture(int(url))
                if not cap.isOpened():
                    print(f"[STREAM] Failed to open webcam {url}")
                    url_idx += 1
                    time.sleep(1.5)
                    continue

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.stream_res = (w, h)
                print(f"[STREAM] >>> Webcam Connected {w}x{h} <<<")

                while not self.stopped:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print("[STREAM] Webcam dropped, reconnecting...")
                        break
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                        self.stream_res = (frame.shape[1], frame.shape[0])
                cap.release()
            elif is_phone:
                # Use OpenCV default backend for HTTP MJPEG (not FFMPEG, not snapshot)
                cap = cv2.VideoCapture(url)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if not cap.isOpened():
                    print(f"[STREAM] Failed to open phone camera")
                    url_idx += 1
                    time.sleep(1.5)
                    continue

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.stream_res = (w, h)
                print(f"[STREAM] >>> Phone Camera Connected {w}x{h} <<<")

                while not self.stopped:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print("[STREAM] Phone stream dropped, reconnecting...")
                        break
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                        self.stream_res = (frame.shape[1], frame.shape[0])
                cap.release()
            else:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if not cap.isOpened():
                    print(f"[STREAM] Failed to open")
                    url_idx += 1
                    time.sleep(1.5)
                    continue

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.stream_res = (w, h)
                print(f"[STREAM] >>> Connected {w}x{h} <<<")

                if w < 1280:
                    print("=" * 60)
                    print(f"  WARNING: Stream is only {w}x{h}")
                    print("  You are probably on the SUB-STREAM.")
                    print("  Use a subtype=0 URL for full resolution.")
                    print("=" * 60)

                while not self.stopped:
                    # Aggressively grab frames to drain the FFMPEG buffer without decoding them
                    # This prevents the 4K stream from lagging
                    for _ in range(3):
                        cap.grab()
                    
                    # Only retrieve (decode) the most recent frame
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        print("[STREAM] Dropped, reconnecting...")
                        break
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                        self.stream_res = (frame.shape[1], frame.shape[0])
                cap.release()

            url_idx += 1
            time.sleep(1)

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.stopped = True


# =============================================================================
# Decoder cascade
# =============================================================================

class Decoder:
    """WeChat CNN (with super-resolution) -> zxing-cpp -> pyzbar -> OpenCV."""

    def __init__(self):
        self.cv2_qr = cv2.QRCodeDetector()
        self.wechat = None
        self._init_wechat()

    def _init_wechat(self):
        d = WECHAT_MODEL_DIR
        files = [
            os.path.join(d, "detect.prototxt"),
            os.path.join(d, "detect.caffemodel"),
            os.path.join(d, "sr.prototxt"),
            os.path.join(d, "sr.caffemodel"),
        ]
        if all(os.path.exists(f) for f in files):
            try:
                self.wechat = cv2.wechat_qrcode_WeChatQRCode(*files)
                print("[DECODER] WeChat CNN detector loaded (best option)")
                return
            except Exception as e:
                print(f"[DECODER] WeChat init failed: {e}")
        print(f"[DECODER] WeChat models not found in ./{d}/")
        print("[DECODER] Download from github.com/WeChatCV/opencv_3rdparty")
        print("[DECODER] This is the single biggest accuracy gain available.")

    def decode(self, img):
        """Returns (text, pts) or (None, None). pts is 4x2 int array."""

        # 1. WeChat - CNN detection + built-in super-resolution
        if self.wechat is not None:
            try:
                texts, points = self.wechat.detectAndDecode(img)
                if texts and len(texts) > 0 and texts[0]:
                    pts = np.array(points[0], dtype=np.int32) if len(points) else None
                    return texts[0], pts
            except Exception:
                pass

        # 2. zxing-cpp (configured directly via arguments for older bindings)
        if HAS_ZXING:
            try:
                for b in zxingcpp.read_barcodes(img, try_rotate=True, try_invert=True):
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

        # 3. pyzbar
        if HAS_ZBAR:
            try:
                for r in zbar_decode(img):
                    if r.data:
                        pts = np.array([[p.x, p.y] for p in r.polygon], dtype=np.int32)
                        if len(pts) == 4:
                            return r.data.decode("utf-8", errors="replace"), pts
            except Exception:
                pass

        # 4. OpenCV fallback
        try:
            text, points, _ = self.cv2_qr.detectAndDecode(img)
            if text and points is not None and len(points) > 0:
                return text, points[0].astype(np.int32)
        except Exception:
            pass

        return None, None


# =============================================================================
# Detection worker - runs at NATIVE resolution
# =============================================================================

class DetectionWorker:
    def __init__(self):
        self.decoder = Decoder()
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.input_frame = None
        self.result = (None, None, 0.0, "none", 0.0, 0.0)  # pts,data,pw,mode,time,focus
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.stopped = False
        self.last_roi = None  # Stores last successfully scanned QR bounding box (x, y, w, h)
        self.last_roi_time = 0.0

    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()
        return self

    def submit(self, frame):
        """Submit RAW frame. Never pass a digitally-zoomed frame here."""
        with self.lock:
            self.input_frame = frame
        self.event.set()

    def get_result(self):
        with self.lock:
            return self.result

    def _worker(self):
        while not self.stopped:
            self.event.wait(timeout=0.5)
            self.event.clear()

            with self.lock:
                frame = self.input_frame
                self.input_frame = None
            if frame is None:
                continue

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fscore = focus_score(gray_full)

            # Process the current live frame immediately
            pts, data, pw, mode = self._detect(gray_full)

            with self.lock:
                if pts is not None:
                    self.result = (pts, data, pw, mode, time.time(), fscore)
                    # Track this region for subsequent frames
                    min_x, min_y = pts.min(axis=0)
                    max_x, max_y = pts.max(axis=0)
                    self.last_roi = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
                    self.last_roi_time = time.time()
                else:
                    old = self.result
                    self.result = (old[0], old[1], old[2], old[3], old[4], fscore)
                    # Clear tracked ROI if it's older than 1.5 seconds
                    if time.time() - self.last_roi_time > 1.5:
                        self.last_roi = None

    def _detect(self, gray):
        """Smart ROI detection like Google Lens. Fast, multi-scale, and zero-lag."""
        h, w = gray.shape[:2]

        # STAGE 0: Temporal ROI Tracking (Prioritize scanning where the QR code just was)
        if self.last_roi is not None:
            rx, ry, rw, rh = self.last_roi
            # Expand region by 35% to allow for camera movement/motion blur
            pad_x = int(rw * 0.35)
            pad_y = int(rh * 0.35)
            tx = max(0, rx - pad_x)
            ty = max(0, ry - pad_y)
            tw = min(w - tx, rw + 2 * pad_x)
            th = min(h - ty, rh + 2 * pad_y)
            
            if tw > 20 and th > 20:
                roi = gray[ty:ty+th, tx:tx+tw]
                r = self._scan(roi, "tracked_roi", enhance=True, upscale=True)
                if r[0] is not None:
                    pts = r[0]
                    pts[:, 0] += tx
                    pts[:, 1] += ty
                    return pts, r[1], r[2], r[3]

        # STAGE 1: Full-frame raw scan (Fast path for close-up QR codes)
        r = self._scan(gray, "full")
        if r[0] is not None:
            return r

        # STAGE 2: Smart ROI Candidate detection using contour hierarchy
        candidates = self._find_qr_candidates(gray)
        # Sort candidates by area in descending order and limit to top 2 to keep speed high
        candidates = sorted(candidates, key=lambda c: -(c[2] * c[3]))[:2]
        for idx, (rx, ry, rw, rh) in enumerate(candidates):
            roi = gray[ry:ry+rh, rx:rx+rw]
            # Run deep scan on the specific region
            r = self._scan(roi, f"candidate_{idx}", enhance=True, upscale=True)
            if r[0] is not None:
                pts = r[0]
                pts[:, 0] += rx
                pts[:, 1] += ry
                return pts, r[1], r[2], r[3]

        # STAGE 3: Center guide box fallback (Where users naturally hold items)
        # Crop center 45% of the frame
        cy1, cy2 = int(h * 0.275), int(h * 0.725)
        cx1, cx2 = int(w * 0.275), int(w * 0.725)
        center_roi = gray[cy1:cy2, cx1:cx2]
        r = self._scan(center_roi, "center_fallback", enhance=True, upscale=True)
        if r[0] is not None:
            pts = r[0]
            pts[:, 0] += cx1
            pts[:, 1] += cy1
            return pts, r[1], r[2], r[3]

        # STAGE 4: Fast 2x2 quadrant fallback if candidates weren't detected
        mid_h, mid_w = h // 2, w // 2
        pad_y, pad_x = int(mid_h * 0.15), int(mid_w * 0.15)
        quadrants = [
            (0, max(0, mid_h + pad_y), 0, max(0, mid_w + pad_x), "TL"),
            (0, max(0, mid_h + pad_y), max(0, mid_w - pad_x), w, "TR"),
            (max(0, mid_h - pad_y), h, 0, max(0, mid_w + pad_x), "BL"),
            (max(0, mid_h - pad_y), h, max(0, mid_w - pad_x), w, "BR"),
        ]
        for y1, y2, x1, x2, label in quadrants:
            quad = gray[y1:y2, x1:x2]
            r = self._scan(quad, f"quad_{label}", enhance=True, upscale=True)
            if r[0] is not None:
                pts = r[0]
                pts[:, 0] += x1
                pts[:, 1] += y1
                return pts, r[1], r[2], r[3]

        return None, None, 0.0, "none"

    def _find_qr_candidates(self, gray):
        """Locates finder patterns using dual thresholding (adaptive + Otsu) and groups them into ROIs."""
        h, w = gray.shape[:2]
        scale = 1.0
        
        # Downscale to 1080p height for high-speed contour processing
        if w > 1920:
            scale = w / 1920.0
            gray_small = cv2.resize(gray, (1920, int(h / scale)), interpolation=cv2.INTER_LINEAR)
        else:
            gray_small = gray

        # Try both Adaptive (uneven light) and Otsu (high contrast close-up) thresholding
        thresh_adapt = cv2.adaptiveThreshold(
            gray_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 5
        )
        _, thresh_otsu = cv2.threshold(gray_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        patterns = []
        for thresh in [thresh_adapt, thresh_otsu]:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue

            hierarchy = hierarchy[0]
            for i in range(len(contours)):
                # Check for nesting depth of 2 (outer box -> inner space -> center box)
                child1 = hierarchy[i][2]
                if child1 != -1:
                    child2 = hierarchy[child1][2]
                    if child2 != -1:
                        x, y, cw, ch = cv2.boundingRect(contours[i])
                        aspect = cw / ch
                        if 0.75 < aspect < 1.25 and cw > 6 and ch > 6:
                            cx, cy = x + cw / 2.0, y + ch / 2.0
                            # Prevent adding identical patterns from the two threshold passes
                            if not any(np.sqrt((cx - px)**2 + (cy - py)**2) < 5 for px, py, ps in patterns):
                                patterns.append((cx, cy, max(cw, ch)))

        if not patterns:
            return []

        # Group patterns that are spatially close (within same QR code structure)
        regions = []
        used = [False] * len(patterns)

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
                max_s = max(s1, s2)
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                # If they are within 12x their size, they belong to the same QR code
                if dist < max_s * 12.0:
                    group.append(patterns[j])
                    used[j] = True

            # Bounding box of the group
            min_x = min(p[0] - p[2]/2.0 for p in group)
            max_x = max(p[0] + p[2]/2.0 for p in group)
            min_y = min(p[1] - p[2]/2.0 for p in group)
            max_y = max(p[1] + p[2]/2.0 for p in group)

            # Map coordinates back to the full resolution
            min_x *= scale
            max_x *= scale
            min_y *= scale
            max_y *= scale

            gw = max_x - min_x
            gh = max_y - min_y

            # Add padding: if only 1 pattern found, pad 3x its size; else pad 40% of group box
            pad_x = int(gw * 0.4) if len(group) > 1 else int(max(s1 * scale, 30.0) * 3.0)
            pad_y = int(gh * 0.4) if len(group) > 1 else int(max(s1 * scale, 30.0) * 3.0)

            rx = int(max(0, min_x - pad_x))
            ry = int(max(0, min_y - pad_y))
            rw = int(min(w - rx, gw + 2 * pad_x))
            rh = int(min(h - ry, gh + 2 * pad_y))

            if rw > 20 and rh > 20:
                regions.append((rx, ry, rw, rh))

        return regions

    def _scan(self, gray, label, enhance=False, upscale=False):
        """Ultra-fast scanning: Decodes raw grayscale and a high-speed upscaled-sharpened variant."""
        variants = [(gray, f"{label}_raw")]

        if upscale:
            hh, ww = gray.shape[:2]
            if hh * ww < 4_000_000:  # don't upscale huge regions
                # Fast cubic upscale is 2x faster than Lanczos
                up = cv2.resize(gray, (ww * 2, hh * 2), interpolation=cv2.INTER_CUBIC)
                
                # Apply a sharpening filter specifically to combat motion blur
                k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                up_sharp = cv2.filter2D(up, -1, k)
                
                variants.append((up_sharp, f"{label}_up2x_sharp"))

        for img, mode in variants:
            was_up = "up2x" in mode
            text, pts = self.decoder.decode(img)
            if text and pts is not None and len(pts) >= 4:
                pts = pts.astype(np.float64)
                if was_up:
                    pts /= 2.0
                pts = pts.astype(np.int32)
                tw_ = np.linalg.norm(pts[0].astype(float) - pts[1].astype(float))
                bw_ = np.linalg.norm(pts[2].astype(float) - pts[3].astype(float))
                return pts, text, (tw_ + bw_) / 2.0, mode

        return None, None, 0.0, f"{label}_none"

    def stop(self):
        self.stopped = True


# =============================================================================
# Main
# =============================================================================

def main():
    urls = [sys.argv[1]] if len(sys.argv) > 1 else RTSP_CANDIDATE_URLS
    reader = StreamReader(urls).start()
    detector = DetectionWorker().start()

    win = "CCTV QR Scanner - Diagnostic"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_WIDTH, int(DISPLAY_WIDTH * 9 / 16))

    display_zoom = [1]
    cv2.createTrackbar("DisplayZoom", win, 1, 4,
                       lambda v: display_zoom.__setitem__(0, max(1, v)))

    print("\n" + "=" * 60)
    print("DIAGNOSTIC BUILD")
    print("Watch the FOCUS number:")
    print(f"  < {FOCUS_SOFT_THRESHOLD}   BLURRED - move BACK, you are too close")
    print(f"  > {FOCUS_SHARP_THRESHOLD}  SHARP   - if still failing, it's resolution")
    print("Note: DisplayZoom only affects the window, NOT detection.")
    print("Press 'q' to quit")
    print("=" * 60 + "\n")

    CSV_FILE = "scanned_products.csv"
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "QR_Data", "PixelWidth", "Focus"])

    last_data = None
    last_log = 0.0
    fps_t, fps_n, fps = time.time(), 0, 0.0

    while True:
        ok, raw = reader.read()
        if not ok or raw is None:
            blank = np.full((720, 1280, 3), 30, np.uint8)
            cv2.putText(blank, reader.status_message, (60, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.imshow(win, blank)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
            continue

        now = time.time()

        # CRITICAL: raw frame goes to detector, every frame, no zoom applied
        detector.submit(raw)

        pts, data, pw, mode, det_t, fscore = detector.get_result()

        fps_n += 1
        if now - fps_t >= 1.0:
            fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

        # Display copy only
        frame = raw.copy()
        z = display_zoom[0]
        if z > 1:
            fh, fw = frame.shape[:2]
            zh, zw = fh // z, fw // z
            frame = cv2.resize(frame[(fh - zh) // 2:(fh - zh) // 2 + zh,
                                     (fw - zw) // 2:(fw - zw) // 2 + zw],
                               (fw, fh), interpolation=cv2.INTER_LANCZOS4)

        h, w = frame.shape[:2]
        detected = pts is not None and (now - det_t < 1.0)

        # HUD
        cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)

        col = (0, 220, 0) if detected else (0, 140, 255)
        cv2.putText(frame, "QR DETECTED" if detected else "SCANNING...",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        # FOCUS - the key diagnostic
        if fscore < FOCUS_SOFT_THRESHOLD:
            fcol, ftxt = (0, 0, 255), "BLURRED - MOVE BACK"
        elif fscore < FOCUS_SHARP_THRESHOLD:
            fcol, ftxt = (0, 200, 255), "SOFT"
        else:
            fcol, ftxt = (0, 255, 0), "SHARP"
        cv2.putText(frame, f"FOCUS: {fscore:.0f}  {ftxt}", (15, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcol, 2)

        sr = reader.stream_res
        rcol = (0, 0, 255) if sr[0] < 1280 else (0, 255, 255)
        cv2.putText(frame, f"RES: {sr[0]}x{sr[1]}", (15, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rcol, 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 160, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if detected and pw > 0:
            # Rescale detection points if display is zoomed
            pd = pts.copy()
            if z > 1:
                fh, fw = raw.shape[:2]
                zh, zw = fh // z, fw // z
                oy, ox = (fh - zh) // 2, (fw - zw) // 2
                pd = ((pd - [ox, oy]) * z).astype(np.int32)
            pd[:, 0] = np.clip(pd[:, 0], 0, w - 1)
            pd[:, 1] = np.clip(pd[:, 1], 0, h - 1)

            cv2.polylines(frame, [pd], True, (0, 255, 0), 3)
            for p in pd:
                cv2.circle(frame, tuple(p), 6, (0, 0, 255), -1)

            my = max(140, int(pd[:, 1].min()))
            mx = max(5, int(pd[:, 0].min()))
            cv2.rectangle(frame, (mx - 5, my - 80), (mx + 460, my - 5), (0, 0, 0), -1)
            cv2.putText(frame, f"DATA: {str(data)[:34]}", (mx, my - 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"QR WIDTH: {int(pw)} px", (mx, my - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # px/module for common QR versions
            est = "  ".join(f"V{v}:{pw / m:.1f}"
                            for v, m in [(1, 21), (2, 25), (4, 33)])
            cv2.putText(frame, f"px/module  {est}", (mx, my - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            if data != last_data or now - last_log > 1.0:
                print(f"\nData: {data}")
                print(f"Distance: {int(pw)}px  focus={fscore:.0f}  mode={mode}")
                with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        data, int(pw), int(fscore)])
                last_data, last_log = data, now

        if w > DISPLAY_WIDTH:
            frame = cv2.resize(frame, (DISPLAY_WIDTH,
                                       int(h * DISPLAY_WIDTH / w)), interpolation=cv2.INTER_AREA)
        cv2.imshow(win, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.stop()
    reader.stop()
    cv2.destroyAllWindows()
    os._exit(0)


if __name__ == "__main__":
    main()
