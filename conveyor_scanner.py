"""
Conveyor Belt QR Scanner - Low Latency Rebuild
===============================================
Fixes the 3-4 second lag and the freezing.

WHAT WAS WRONG IN THE OLD VERSION
---------------------------------
1. for _ in range(8): cap.grab()
   grab() BLOCKS on RTSP when the buffer is empty. This did not drain a
   backlog - it waited for 8 NEW frames. At 25fps that is 320ms per loop,
   so capture ran at ~3fps while the stream arrived at 25fps. The FFmpeg
   buffer then grew without bound. THIS was the 3-4 second lag.

2. Resize + cvtColor + crop inside the capture loop.
   A 4K resize is 15-25ms. Budget at 25fps is 40ms. Any hiccup becomes
   PERMANENT latency because the decoder falls behind and never recovers.

3. winsound.Beep() is synchronous - blocked the detection thread 100ms.

4. read_display() did .copy() while holding the lock, stalling capture.

5. prev_zone is zone_gray - identity check on a numpy ref. Fragile.

THE CORRECT ARCHITECTURE
------------------------
  CAPTURE thread : cap.read() ONLY. Nothing else. Ever.
                   This is the only way the decoder keeps up and the
                   buffer never accumulates.
  WORKER thread  : takes the LATEST frame, drops anything it missed.
                   resize -> display frame, crop -> locate -> decode.
                   Dropping frames here costs nothing.
  MAIN thread    : imshow + HUD. Never touches a raw frame.

Locks are held only for a pointer swap - never for a copy or a resize.

USAGE
-----
  python conveyor_scanner.py
  python conveyor_scanner.py 0
  python conveyor_scanner.py "rtsp://user:pass@192.168.1.43:554/..."
  python conveyor_scanner.py "rtsp://..." --gst      # GStreamer, lowest latency
  python conveyor_scanner.py "rtsp://..." --udp      # UDP transport

CONTROLS
--------
  q     quit
  z/x   detection zone width
  a/s   detection zone height
  r     force reconnect
"""

import os
import sys
import time
import csv
import threading
from collections import deque
from datetime import datetime

# ---------------------------------------------------------------------------
# FFmpeg options MUST be set before importing cv2.
# nobuffer + low_delay stop FFmpeg from queuing frames internally.
# analyzeduration/probesize keep connection setup fast.
# ---------------------------------------------------------------------------
_TRANSPORT = "udp" if "--udp" in sys.argv else "tcp"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    f"rtsp_transport;{_TRANSPORT}"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|analyzeduration;0"
    "|probesize;32"
    "|reorder_queue_size;0"
)

import cv2
import numpy as np

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False
    print("[WARN] pip install zxing-cpp")

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

DETECT_ZONE_X = 0.20
DETECT_ZONE_Y = 0.15

DEDUP_TIME_WINDOW = 5.0

LOCATOR_WIDTH = 640

# Throttle detection so it cannot starve capture on a weak CPU.
# Your box dwells 3-4s, so 15 detections/sec is far more than enough.
MAX_DETECT_FPS = 15

# If no frame arrives for this long, force a reconnect.
STALL_TIMEOUT = 3.0


# =============================================================================
# NON-BLOCKING BEEP
# =============================================================================

def beep_async():
    """Fire and forget. The old code blocked the detection thread 100ms."""
    def _b():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1200, 90)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass
    threading.Thread(target=_b, daemon=True).start()


# =============================================================================
# 1. CAPTURE - does nothing but read
# =============================================================================

class CaptureThread:
    """
    The single most important rule in this file:
    this loop calls cap.read() and NOTHING else.

    Any work added here falls behind the stream and turns into permanent
    latency. All processing happens downstream where frames can be dropped
    for free.
    """

    def __init__(self, source, use_gstreamer=False):
        self.source = source
        self.use_gst = use_gstreamer

        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()

        self.stopped = False
        self.reconnect_flag = False
        self.res = (0, 0)
        self.kind = "?"
        self.last_frame_mono = 0.0
        self.capture_fps = 0.0

    # -- opening -----------------------------------------------------------

    def _gst_pipeline(self, url):
        """
        GStreamer is genuinely lower latency than FFmpeg for RTSP.
        latency=0 + drop-on-latency + appsink max-buffers=1 means the
        pipeline never holds more than one frame.
        """
        return (
            f'rtspsrc location="{url}" latency=0 drop-on-latency=true '
            f'protocols={_TRANSPORT} ! '
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def _open(self):
        src = self.source
        is_idx = isinstance(src, int) or (isinstance(src, str) and str(src).isdigit())

        if is_idx:
            self.kind = "Webcam"
            cap = cv2.VideoCapture(int(src))
            # UVC cameras: ask for MJPEG, it is far lighter than raw YUY2
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        elif isinstance(src, str) and "8080" in src:
            self.kind = "Phone"
            cap = cv2.VideoCapture(src)
        else:
            self.kind = "RTSP"
            if self.use_gst:
                cap = cv2.VideoCapture(self._gst_pipeline(src), cv2.CAP_GSTREAMER)
                if not cap.isOpened():
                    print("[STREAM] GStreamer failed, falling back to FFmpeg")
                    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        # Works on some backends, harmless on others.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return cap

    # -- the loop ----------------------------------------------------------

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self.stopped:
            cap = self._open()
            if not cap or not cap.isOpened():
                print(f"[STREAM] cannot open {self.kind}, retry in 2s")
                time.sleep(2)
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.res = (w, h)
            print(f"[STREAM] >>> {self.kind} connected {w}x{h} "
                  f"({'GStreamer' if self.use_gst else 'FFmpeg/' + _TRANSPORT}) <<<")

            if 0 < w < 1280:
                print("  ! Low resolution - you may be on the RTSP SUB-stream.")
                print("  ! Use a subtype=0 / main-stream URL.")

            fails = 0
            n, t0 = 0, time.monotonic()

            while not self.stopped and not self.reconnect_flag:
                # ---- THE ONLY THING THIS THREAD DOES ----
                ok, frame = cap.read()

                if not ok or frame is None:
                    fails += 1
                    if fails > 8:
                        print("[STREAM] too many read failures, reconnecting")
                        break
                    continue
                fails = 0

                # cap.read() already returns a fresh array - no copy needed.
                # Lock is held for one pointer assignment only.
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                self.last_frame_mono = time.monotonic()

                n += 1
                if n >= 30:
                    dt = time.monotonic() - t0
                    if dt > 0:
                        self.capture_fps = n / dt
                    n, t0 = 0, time.monotonic()

            cap.release()
            self.reconnect_flag = False
            time.sleep(0.3)

    # -- consumer API ------------------------------------------------------

    def latest(self, since_seq):
        """
        Return (frame, seq) if a NEWER frame exists, else (None, since_seq).
        No copy: the capture thread never mutates a published frame, it only
        replaces the reference.
        """
        with self._lock:
            if self._seq == since_seq or self._frame is None:
                return None, since_seq
            return self._frame, self._seq

    def lag(self):
        if self.last_frame_mono == 0:
            return 999.0
        return time.monotonic() - self.last_frame_mono

    def reconnect(self):
        self.reconnect_flag = True

    def stop(self):
        self.stopped = True


# =============================================================================
# 2. FAST QR LOCATOR
# =============================================================================

class FastQRLocator:
    """Finds QR bounding boxes by finder-pattern contour nesting. ~3-5ms."""

    def locate(self, gray):
        h, w = gray.shape[:2]
        scale = 1.0

        if w > LOCATOR_WIDTH:
            scale = w / float(LOCATOR_WIDTH)
            small = cv2.resize(gray, (LOCATOR_WIDTH, int(h / scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = gray

        _, th = cv2.threshold(small, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cnts, hier = cv2.findContours(th, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return []
        hier = hier[0]

        pats = []
        for i in range(len(cnts)):
            c1 = hier[i][2]
            if c1 == -1:
                continue
            if hier[c1][2] == -1:
                continue

            x, y, cw, ch = cv2.boundingRect(cnts[i])
            if cw < 6 or ch < 6:
                continue
            if not (0.7 < cw / float(ch) < 1.4):
                continue

            pats.append(((x + cw / 2.0) * scale,
                         (y + ch / 2.0) * scale,
                         max(cw, ch) * scale))

        return self._group(pats, w, h) if pats else []

    def _group(self, pats, iw, ih):
        used = [False] * len(pats)
        out = []

        for i in range(len(pats)):
            if used[i]:
                continue
            used[i] = True
            grp = [pats[i]]
            cx1, cy1, s1 = pats[i]

            for j in range(i + 1, len(pats)):
                if used[j]:
                    continue
                cx2, cy2, s2 = pats[j]
                if np.hypot(cx1 - cx2, cy1 - cy2) < max(s1, s2) * 12.0:
                    grp.append(pats[j])
                    used[j] = True

            x0 = min(p[0] - p[2] / 2 for p in grp)
            x1 = max(p[0] + p[2] / 2 for p in grp)
            y0 = min(p[1] - p[2] / 2 for p in grp)
            y1 = max(p[1] + p[2] / 2 for p in grp)

            gw, gh = x1 - x0, y1 - y0
            pad = 0.4 if len(grp) > 1 else 2.0
            px, py = int(gw * pad), int(gh * pad)

            rx = int(max(0, x0 - px))
            ry = int(max(0, y0 - py))
            rw = int(min(iw - rx, gw + 2 * px))
            rh = int(min(ih - ry, gh + 2 * py))

            if rw > 20 and rh > 20:
                out.append((rx, ry, rw, rh))

        out.sort(key=lambda r: -(r[2] * r[3]))
        return out[:3]


# =============================================================================
# 3. FAST DECODER
# =============================================================================

class FastDecoder:
    """Two attempts per ROI. Raw, then sharpened 2x upscale."""

    def __init__(self):
        self.cv_qr = cv2.QRCodeDetector()
        self._k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

    def decode(self, roi):
        t, p = self._try(roi)
        if t:
            return t, p

        h, w = roi.shape[:2]
        if h * w < 1_500_000:
            up = cv2.resize(roi, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            up = cv2.filter2D(up, -1, self._k)
            t, p = self._try(up)
            if t:
                if p is not None:
                    p = (p.astype(np.float64) / 2.0).astype(np.int32)
                return t, p
        return None, None

    def _try(self, img):
        if HAS_ZXING:
            try:
                for b in zxingcpp.read_barcodes(img, try_rotate=True,
                                                try_invert=True):
                    if b.text and b.position:
                        q = b.position
                        return b.text, np.array([
                            [q.top_left.x, q.top_left.y],
                            [q.top_right.x, q.top_right.y],
                            [q.bottom_right.x, q.bottom_right.y],
                            [q.bottom_left.x, q.bottom_left.y]], np.int32)
            except Exception:
                pass

        if HAS_ZBAR:
            try:
                for r in zbar_decode(img):
                    if r.data:
                        p = np.array([[pt.x, pt.y] for pt in r.polygon], np.int32)
                        return r.data.decode("utf-8", "replace"), \
                               (p if len(p) == 4 else None)
            except Exception:
                pass

        try:
            t, p, _ = self.cv_qr.detectAndDecode(img)
            if t and p is not None and len(p):
                return t, p[0].astype(np.int32)
        except Exception:
            pass

        return None, None


# =============================================================================
# 4. PRODUCT TRACKER
# =============================================================================

class ProductTracker:
    """
    Simplified from the old version, which had contradictory position logic
    (two branches that both returned False for the same condition).

    On a conveyor with 2-3s gaps between boxes, time-based dedup is the
    correct and sufficient rule: same code inside the window = same box.
    """

    def __init__(self, window=DEDUP_TIME_WINDOW):
        self.window = window
        self.seen = {}          # text -> last seen monotonic
        self.total_unique = 0

    def is_new(self, text):
        now = time.monotonic()

        # expire
        for k in [k for k, v in self.seen.items() if now - v > self.window]:
            del self.seen[k]

        if text in self.seen:
            self.seen[text] = now
            return False

        self.seen[text] = now
        self.total_unique += 1
        return True


# =============================================================================
# 5. SCANNER
# =============================================================================

class ConveyorScanner:
    def __init__(self, source, use_gst=False):
        self.zone_x = DETECT_ZONE_X
        self.zone_y = DETECT_ZONE_Y

        self.cap = CaptureThread(source, use_gst)
        self.locator = FastQRLocator()
        self.decoder = FastDecoder()
        self.tracker = ProductTracker()

        self._lock = threading.Lock()
        self._display = None
        self._disp_seq = 0
        self._zone_rect = (0, 0, 0, 0)
        self._det = {"on": False, "text": None, "pts": None, "pw": 0.0}

        self.last_text = ""
        self.last_text_t = 0.0
        self.flash_until = 0.0
        self.worker_fps = 0.0
        self.detect_ms = 0.0
        self.dropped = 0
        self.stopped = False

    # -- worker: everything except read() and imshow ------------------------

    def _worker(self):
        seq = -1
        n, t0 = 0, time.monotonic()
        next_detect = 0.0
        min_gap = 1.0 / MAX_DETECT_FPS

        while not self.stopped:
            frame, new_seq = self.cap.latest(seq)
            if frame is None:
                time.sleep(0.002)
                continue

            # Count how many frames we skipped. Dropping here is FREE -
            # it does not add latency, unlike falling behind in capture.
            if seq >= 0:
                self.dropped += max(0, new_seq - seq - 1)
            seq = new_seq

            rh, rw = frame.shape[:2]

            # resize for display
            if rw > DISPLAY_WIDTH:
                s = DISPLAY_WIDTH / float(rw)
                disp = cv2.resize(frame, (DISPLAY_WIDTH, int(rh * s)),
                                  interpolation=cv2.INTER_LINEAR)
            else:
                disp = frame.copy()

            dh, dw = disp.shape[:2]
            zx1 = int(dw * self.zone_x)
            zx2 = int(dw * (1.0 - self.zone_x))
            zy1 = int(dh * self.zone_y)
            zy2 = int(dh * (1.0 - self.zone_y))

            # publish display frame immediately so the window stays smooth
            with self._lock:
                self._display = disp
                self._disp_seq += 1
                self._zone_rect = (zx1, zy1, zx2, zy2)

            # throttled detection
            now = time.monotonic()
            if now >= next_detect:
                next_detect = now + min_gap
                zone = cv2.cvtColor(disp[zy1:zy2, zx1:zx2], cv2.COLOR_BGR2GRAY)
                t_start = time.monotonic()
                self._detect(zone, zx1, zy1)
                self.detect_ms = (time.monotonic() - t_start) * 1000.0

            n += 1
            if n >= 30:
                dt = time.monotonic() - t0
                if dt > 0:
                    self.worker_fps = n / dt
                n, t0 = 0, time.monotonic()

    def _detect(self, zone, zx1, zy1):
        for (rx, ry, rw, rh) in self.locator.locate(zone):
            roi = zone[ry:ry + rh, rx:rx + rw]
            if roi.size == 0:
                continue

            text, pts = self.decoder.decode(roi)
            if not text:
                continue

            pf = None
            pw = 0.0
            if pts is not None:
                pf = pts.copy()
                pf[:, 0] += rx + zx1
                pf[:, 1] += ry + zy1
                if len(pf) >= 4:
                    a = np.linalg.norm(pf[0].astype(float) - pf[1].astype(float))
                    b = np.linalg.norm(pf[2].astype(float) - pf[3].astype(float))
                    pw = (a + b) / 2.0

            if self.tracker.is_new(text):
                self._log(text, pw)
                self.flash_until = time.monotonic() + 0.8
                print(f"\n[SCAN #{self.tracker.total_unique}] {text}   "
                      f"({pw:.0f}px)")
                beep_async()                       # non-blocking now

            self.last_text = text
            self.last_text_t = time.monotonic()

            with self._lock:
                self._det = {"on": True, "text": text, "pts": pf, "pw": pw}
            return

        with self._lock:
            self._det = {"on": False, "text": None, "pts": None, "pw": 0.0}

    # -- main loop: imshow only --------------------------------------------

    def run(self):
        self.cap.start()
        threading.Thread(target=self._worker, daemon=True).start()

        win = "Conveyor QR Scanner"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, DISPLAY_WIDTH, int(DISPLAY_WIDTH * 9 / 16))

        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["Timestamp", "QR_Data", "PixelWidth", "Status"])

        print("\n" + "=" * 60)
        print("CONVEYOR QR SCANNER - low latency build")
        print("  q quit | z/x zone width | a/s zone height | r reconnect")
        print("=" * 60 + "\n")

        shown = -1
        ui_n, ui_t = 0, time.monotonic()
        ui_fps = 0.0

        while True:
            with self._lock:
                if self._disp_seq != shown and self._display is not None:
                    frame = self._display.copy()
                    shown = self._disp_seq
                    zr = self._zone_rect
                    det = dict(self._det)
                else:
                    frame = None

            if frame is None:
                # No new frame. Do NOT spin - that starves the worker.
                if self.cap.lag() > STALL_TIMEOUT:
                    blank = np.full((720, 1280, 3), 25, np.uint8)
                    cv2.putText(blank, "STREAM STALLED - reconnecting",
                                (300, 340), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 120, 255), 2)
                    cv2.imshow(win, blank)
                    self.cap.reconnect()
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            ui_n += 1
            if time.monotonic() - ui_t >= 1.0:
                ui_fps = ui_n / (time.monotonic() - ui_t)
                ui_n, ui_t = 0, time.monotonic()

            self._hud(frame, zr, det, ui_fps)
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("z"):
                self.zone_x = min(0.45, self.zone_x + 0.02)
            elif k == ord("x"):
                self.zone_x = max(0.0, self.zone_x - 0.02)
            elif k == ord("a"):
                self.zone_y = min(0.45, self.zone_y + 0.02)
            elif k == ord("s"):
                self.zone_y = max(0.0, self.zone_y - 0.02)
            elif k == ord("r"):
                print("[STREAM] manual reconnect")
                self.cap.reconnect()

        self.stopped = True
        self.cap.stop()
        cv2.destroyAllWindows()
        os._exit(0)

    def _log(self, text, pw):
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                text, int(pw), "SCANNED"])

    def _hud(self, f, zr, det, ui_fps):
        fh, fw = f.shape[:2]
        now = time.monotonic()
        flash = now < self.flash_until
        zx1, zy1, zx2, zy2 = zr

        col = (0, 255, 0) if flash else (0, 170, 0)
        cv2.rectangle(f, (zx1, zy1), (zx2, zy2), col, 3 if flash else 1)
        cv2.putText(f, "DETECTION ZONE", (zx1 + 5, max(12, zy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        cv2.rectangle(f, (0, 0), (fw, 92), (18, 18, 18), -1)

        if det["on"]:
            cv2.putText(f, "QR DETECTED", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(f, "SCANNING...", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        # LAG is the number to watch. Should stay under ~0.15s.
        lag = self.cap.lag()
        lcol = (0, 255, 0) if lag < 0.2 else ((0, 200, 255) if lag < 0.6
                                             else (0, 0, 255))
        cv2.putText(f, f"LAG {lag*1000:5.0f}ms", (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lcol, 2)

        cv2.putText(f, f"cap {self.cap.capture_fps:4.1f}  "
                       f"work {self.worker_fps:4.1f}  ui {ui_fps:4.1f}",
                    (170, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

        cv2.putText(f, f"detect {self.detect_ms:4.0f}ms   "
                       f"skipped {self.dropped}",
                    (14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (170, 170, 170), 1)

        sr = self.cap.res
        cv2.putText(f, f"{sr[0]}x{sr[1]} {self.cap.kind}", (fw - 250, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(f, f"SCANNED {self.tracker.total_unique}", (fw - 250, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        if self.last_text and now - self.last_text_t < 5.0:
            cv2.putText(f, f"LAST: {self.last_text[:52]}", (14, fh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        pts = det["pts"]
        if det["on"] and pts is not None and len(pts) >= 4:
            p = pts.copy()
            p[:, 0] = np.clip(p[:, 0], 0, fw - 1)
            p[:, 1] = np.clip(p[:, 1], 0, fh - 1)
            cv2.polylines(f, [p], True, (0, 255, 0), 3)
            for q in p:
                cv2.circle(f, tuple(q), 5, (0, 0, 255), -1)

            ly = max(112, int(p[:, 1].min()) - 10)
            lx = max(5, int(p[:, 0].min()))
            cv2.rectangle(f, (lx - 3, ly - 22), (lx + 400, ly + 5),
                          (0, 0, 0), -1)
            cv2.putText(f, det["text"][:40], (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if flash:
            cv2.rectangle(f, (0, fh // 2 - 28), (fw, fh // 2 + 28),
                          (0, 170, 0), -1)
            cv2.putText(f, f"NEW: {self.last_text[:40]}",
                        (20, fh // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)


# =============================================================================
# ENTRY
# =============================================================================

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_gst = "--gst" in sys.argv

    src = args[0] if args else 0
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    ConveyorScanner(src, use_gst).run()


if __name__ == "__main__":
    main()