import argparse
import time
from collections import deque, Counter
from threading import Thread, Lock
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


def percentile(values, p):
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float32)
    return float(np.percentile(arr, p))


class CameraReader:
    def __init__(self, cam_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Kamera açılamadı. macOS izinlerini kontrol et.")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        self.lock = Lock()
        self.frame = None
        self.ok = True
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.ok:
            ret, frm = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frm

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_camera_fps(self, default=30.0):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # bazı mac/webcam kombinasyonlarında 0.0 döner
        if fps is None or fps <= 1.0 or fps > 240:
            return float(default)
        return float(fps)

    def release(self):
        self.ok = False
        time.sleep(0.05)
        self.cap.release()


def draw_panel(img, text, x=12, y=28):
    for i, line in enumerate(text.splitlines()):
        yy = y + i * 26
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def extract_label_conf(res0):
    # classification
    if getattr(res0, "probs", None) is not None and res0.probs is not None:
        top1 = int(res0.probs.top1)
        top1conf = float(res0.probs.top1conf)
        name = res0.names.get(top1, str(top1))
        return name, top1conf

    # detection
    if getattr(res0, "boxes", None) is not None and res0.boxes is not None and len(res0.boxes) > 0:
        confs = res0.boxes.conf.detach().cpu().numpy()
        idx = int(np.argmax(confs))
        cls_id = int(res0.boxes.cls[idx].item())
        name = res0.names.get(cls_id, str(cls_id))
        confv = float(confs[idx])
        return name, confv

    return "none", 0.0


class SensorFailureGate:
    def __init__(self, dark_mean_thr=25.0, low_contrast_std_thr=8.0, freeze_diff_thr=1.2):
        self.dark_mean_thr = dark_mean_thr
        self.low_contrast_std_thr = low_contrast_std_thr
        self.freeze_diff_thr = freeze_diff_thr
        self.prev_gray = None

    def score(self, frame_bgr) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))

        freeze = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            mad = float(np.mean(diff))
            if mad < self.freeze_diff_thr:
                freeze = 1.0
        self.prev_gray = gray

        dark = 1.0 if mean < self.dark_mean_thr else 0.0
        low_contrast = 1.0 if std < self.low_contrast_std_thr else 0.0

        score = 0.45 * dark + 0.45 * low_contrast + 0.10 * freeze
        return float(np.clip(score, 0.0, 1.0))


class ClassAwareSmoother:
    def __init__(self,
                 window=7,
                 unknown_label="UNKNOWN",
                 per_class_conf=None,
                 switch_conf_by_target=None,
                 min_votes_by_class=None,
                 hold_frames_by_class=None):
        self.window = window
        self.unknown_label = unknown_label

        self.per_class_conf = per_class_conf or {}
        self.switch_conf_by_target = switch_conf_by_target or {}
        self.min_votes_by_class = min_votes_by_class or {}
        self.hold_frames_by_class = hold_frames_by_class or {}

        self.hist = deque(maxlen=window)  # (label, conf)
        self.stable_label = unknown_label
        self.stable_conf = 0.0
        self.hold_counter = 0

    def thr(self, label, default=0.40):
        return float(self.per_class_conf.get(label, default))

    def switch_thr(self, target, default=0.55):
        return float(self.switch_conf_by_target.get(target, default))

    def min_votes(self, label, default=3):
        return int(self.min_votes_by_class.get(label, default))

    def hold_frames(self, label, default=0):
        return int(self.hold_frames_by_class.get(label, default))

    def update(self, label, conf):
        self.hist.append((label, conf))

        if self.hold_counter > 0:
            self.hold_counter -= 1
            return self.stable_label, self.stable_conf

        filtered = [(l, c) for (l, c) in self.hist if c >= self.thr(l)]
        if not filtered:
            self.stable_label = self.unknown_label
            self.stable_conf = 0.0
            return self.stable_label, self.stable_conf

        labels = [l for (l, c) in filtered]
        counts = Counter(labels)
        winner, votes = counts.most_common(1)[0]

        winner_confs = [c for (l, c) in filtered if l == winner]
        avg_conf = float(np.mean(winner_confs)) if winner_confs else 0.0

        if votes < self.min_votes(winner):
            self.stable_label = self.unknown_label
            self.stable_conf = avg_conf
            return self.stable_label, self.stable_conf

        if winner == self.stable_label:
            self.stable_conf = avg_conf
            return self.stable_label, self.stable_conf

        if avg_conf >= max(self.switch_thr(winner), self.thr(winner)):
            self.stable_label = winner
            self.stable_conf = avg_conf
            self.hold_counter = self.hold_frames(winner)
            return self.stable_label, self.stable_conf

        return self.stable_label, self.stable_conf


# ✅ Script'in bulunduğu klasöre göre (camera-live) sabit base
BASE_DIR = Path(__file__).resolve().parent


def _build_output_path(out_arg: str) -> str:
    """
    out_arg:
      - "runs/recordings" gibi relative => camera-live/runs/recordings'e gider
      - absolute path verirsen olduğu gibi kullanır
      - dosya verirsen direkt onu yazar
    """
    p = Path(out_arg)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()

    video_exts = [".mp4", ".avi", ".mov", ".mkv"]
    if p.suffix.lower() in video_exts:
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    p.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(p / f"record_{ts}.mp4")


def _init_writer(path: str, w: int, h: int, fps: float):
    # mac'te genelde sorunsuz: mp4v + .mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        # fallback: avi + MJPG
        alt_path = str(Path(path).with_suffix(".avi"))
        fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(alt_path, fourcc2, float(fps), (int(w), int(h)))
        if not writer.isOpened():
            raise RuntimeError("VideoWriter açılamadı. Codec sorunu olabilir.")
        return writer, alt_path
    return writer, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--device", default="mps", type=str)
    ap.add_argument("--imgsz", default=224, type=int)
    ap.add_argument("--conf", default=0.25, type=float)
    ap.add_argument("--iou", default=0.45, type=float)
    ap.add_argument("--max_det", default=1, type=int)
    ap.add_argument("--overlay_every", default=5, type=int)
    ap.add_argument("--hist", default=200, type=int)
    ap.add_argument("--cam", default=0, type=int)
    ap.add_argument("--width", default=640, type=int)
    ap.add_argument("--height", default=480, type=int)
    ap.add_argument("--window", default=7, type=int)

    # ✅ Video kayıt
    ap.add_argument("--record", action="store_true", help="Overlay'li görüntüyü videoya kaydet.")
    ap.add_argument("--out", default="runs/recordings", type=str,
                    help="Kayıt klasörü veya dosya yolu. (default: camera-live/runs/recordings)")
    ap.add_argument("--out_fps", default=0.0, type=float,
                    help="0 bırakılırsa kameradan okunur (fallback 30).")

    args = ap.parse_args()

    model = YOLO(args.model)
    cam = CameraReader(args.cam, width=args.width, height=args.height)

    # === CLASS BOOST AYARLARI ===
    per_class_conf = {
        "Normal": 0.35,
        "Human Occlusion": 0.40,
        "Object Occlusion": 0.45,
        "Weather Occlusion": 0.45,
        "Sensor Failure": 0.55,
    }

    switch_conf_by_target = {
        "Normal": 0.45,
        "Human Occlusion": 0.50,
        "Object Occlusion": 0.55,
        "Weather Occlusion": 0.60,
        "Sensor Failure": 0.70,
    }

    min_votes_by_class = {
        "Normal": 4,
        "Human Occlusion": 3,
        "Object Occlusion": 3,
        "Weather Occlusion": 3,
        "Sensor Failure": 3,
    }

    hold_frames_by_class = {
        "Human Occlusion": 2,
        "Object Occlusion": 2,
        "Weather Occlusion": 2,
        "Sensor Failure": 3,
    }

    smoother = ClassAwareSmoother(
        window=args.window,
        per_class_conf=per_class_conf,
        switch_conf_by_target=switch_conf_by_target,
        min_votes_by_class=min_votes_by_class,
        hold_frames_by_class=hold_frames_by_class,
    )

    gate = SensorFailureGate()

    infer_hist = deque(maxlen=args.hist)
    total_hist = deque(maxlen=args.hist)

    fps = 0.0
    last_fps_t = time.perf_counter()
    ema_alpha = 0.12
    frame_idx = 0
    cached_text = "starting..."

    writer = None
    out_path = None

    try:
        while True:
            t0 = time.perf_counter()

            frame = cam.read_latest()
            if frame is None:
                continue

            # Inference
            t_infer0 = time.perf_counter()
            res = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                half=True,
                verbose=False
            )
            infer_ms = (time.perf_counter() - t_infer0) * 1000.0
            infer_hist.append(infer_ms)

            raw_label, raw_conf = extract_label_conf(res[0])

            sf_score = gate.score(frame)

            boosted_label = raw_label
            boosted_conf = raw_conf
            if sf_score >= 0.85:
                boosted_label = "Sensor Failure"
                boosted_conf = max(boosted_conf, 0.90)
            elif sf_score >= 0.70:
                if boosted_label == "Sensor Failure":
                    boosted_conf = max(boosted_conf, 0.80)
                else:
                    boosted_label = "Sensor Failure"
                    boosted_conf = max(boosted_conf, 0.70)

            stable_label, stable_conf = smoother.update(boosted_label, boosted_conf)

            total_ms = (time.perf_counter() - t0) * 1000.0
            total_hist.append(total_ms)

            # FPS
            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (now - last_fps_t))
            last_fps_t = now
            fps = inst_fps if fps == 0.0 else (fps * (1 - ema_alpha) + inst_fps * ema_alpha)

            if frame_idx % args.overlay_every == 0:
                infer_p95 = percentile(list(infer_hist), 95)
                infer_p99 = percentile(list(infer_hist), 99)
                total_p95 = percentile(list(total_hist), 95)
                total_p99 = percentile(list(total_hist), 99)

                cached_text = (
                    f"FPS: {fps:.1f}\n"
                    f"Infer p95:{infer_p95:.1f} p99:{infer_p99:.1f} | Total p95:{total_p95:.1f} p99:{total_p99:.1f}\n"
                    f"RAW: {raw_label} ({raw_conf*100:.1f}%)\n"
                    f"BOOST: {boosted_label} ({boosted_conf*100:.1f}%)  SF_score:{sf_score:.2f}\n"
                    f"STABLE: {stable_label} ({stable_conf*100:.1f}%)  window={args.window}"
                )

                print(
                    f"[Perf] FPS~{fps:.1f} | infer p95={infer_p95:.1f} p99={infer_p99:.1f} | "
                    f"total p95={total_p95:.1f} p99={total_p99:.1f} | imgsz={args.imgsz}"
                )

            vis = frame.copy()
            draw_panel(vis, cached_text)
            cv2.imshow("YOLO Boosted (MPS) - class-aware", vis)

            # ✅ Video kaydı
            if args.record:
                if writer is None:
                    out_path = _build_output_path(args.out)
                    h, w = vis.shape[:2]

                    rec_fps = args.out_fps
                    if rec_fps <= 0.0:
                        rec_fps = cam.get_camera_fps(default=30.0)

                    writer, out_path = _init_writer(out_path, w, h, rec_fps)
                    print(f"[REC] Recording started: {out_path} (fps={rec_fps})")

                writer.write(vis)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if writer is not None:
            writer.release()
            print(f"[REC] Recording saved: {out_path}")
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
