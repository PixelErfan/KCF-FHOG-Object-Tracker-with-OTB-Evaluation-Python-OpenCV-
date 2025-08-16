"""
Usage:
# 1) Webcam (select ROI with mouse); save predictions; (optional) evaluate vs GT
python Main.py 1 --camera 0 --save preds.txt --gt /path/to/groundtruth_rect.txt

# 2) Video + bbox (use first line of GT for init; also save/eval)
python Main.py 2 --video /path/to/video.mp4 --gt /path/to/groundtruth_rect.txt --save preds.txt
# or explicit bbox
python Main.py 2 --video /path/to/video.mp4 --bbox 143 125 30 54 --save preds.txt

# 3) Frames + bbox (GT is tab-separated x\ty\tw\th)
python Main.py 3 --frames ./ClifBar/img --gt ./ClifBar/groundtruth_rect.txt --save ./ClifBar/preds.txt
"""
import argparse
import os
import sys
from time import time
import numpy as np
import cv2
import KCF_Tracker
import evaluation_metric as ev

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
inteval = 1
duration = 0.01

import sys
import numpy as np

def maybe_save_predictions(path, preds):
    if not path:
        return
    if len(preds) == 0:
        print("[WARN] No predictions to save; file not written.", flush=True)
        return
    with open(path, "w", encoding="utf-8") as f:
        for (x, y, w, h) in preds:
            f.write(f"{float(x)}\t{float(y)}\t{float(w)}\t{float(h)}\n")
    print(f"[INFO] Saved predictions to: {path}", flush=True)

def maybe_eval_and_print(gt_path, preds):
    if not gt_path:
        print("[INFO] No --gt provided → skipping evaluation.", flush=True)
        return
    if len(preds) == 0:
        print("[WARN] No predictions collected (tracker never initialized?) → skipping evaluation.", flush=True)
        return

    try:
        gt = load_gt_rects(gt_path)                # Nx4 array
        res = np.array(preds, dtype=float)         # Mx4 array
        L = min(len(gt), len(res))
        if L == 0:
            print("[WARN] GT/pred length is zero → skipping evaluation.", flush=True)
            return
        gt = gt[:L, :]
        res = res[:L, :]

        precision_curve, success_curve, cle, auc, dp20 = ev.compile_results(gt, res)

        print("\n=== Evaluation (OTB-style) ===", flush=True)
        print(f"Frames evaluated:          {L}", flush=True)
        print(f"DP@20 (Precision @ 20px):  {dp20:.4f}", flush=True)
        print(f"Average CLE (px):          {cle:.4f}", flush=True)
        print(f"AUC (Success):             {auc:.4f}", flush=True)
        print(f"Precision curve (first 5): {np.round(precision_curve[:5], 4)} ...", flush=True)
        print(f"Success curve (first 5):   {np.round(success_curve[:5], 4)} ...", flush=True)
        print("=============================\n", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", flush=True)

def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and selectingObject:
        cx, cy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if abs(x - ix) > 10 and abs(y - iy) > 10:
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if w > 0:
            ix, iy = int(x - w / 2), int(y - h / 2)
            initTracking = True

def _parse_four_numbers(line):
    line = line.strip()
    if not line:
        return None
    line = line.replace(",", " ")
    parts = line.split()  # splits on spaces OR tabs
    vals = list(map(float, parts[:4]))
    return int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])

def load_bbox_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            bbox = _parse_four_numbers(line)
            if bbox is not None:
                return bbox
    raise ValueError("No valid bbox line found in file: %s" % path)

def load_gt_rects(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_four_numbers(line)
            if parsed is not None:
                arr.append(parsed)
    if not arr:
        raise ValueError("GT file is empty or invalid: %s" % path)
    return np.array(arr, dtype=float)

def iter_frames_from_dir(frames_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(exts)]
    def keyfn(s):
        import re
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    files.sort(key=keyfn)
    for name in files:
        frame = cv2.imread(os.path.join(frames_dir, name))
        if frame is None:
            continue
        yield frame, name

def ensure_init_bbox(args):
    if args.bbox:
        bx, by, bw, bh = args.bbox
        return int(bx), int(by), int(bw), int(bh)
    if args.bbox_file:
        return load_bbox_from_file(args.bbox_file)
    if args.gt:
        gt = load_gt_rects(args.gt)
        x, y, w, h = gt[0, :4]
        return int(x), int(y), int(w), int(h)
    return None

def draw_and_fps(frame, bbox, t0, t1):
    global duration
    bx, by, bw, bh = map(int, bbox)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
    duration = 0.8 * duration + 0.2 * (t1 - t0)
    cv2.putText(frame, f"FPS: {str(1/duration)[:4].strip('.')}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def maybe_save_predictions(path, preds):
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        for (x, y, w, h) in preds:
            f.write(f"{float(x)}\t{float(y)}\t{float(w)}\t{float(h)}\n")
    print(f"[INFO] Saved predictions to: {path}")

def maybe_eval_and_print(gt_path, preds):
    if not gt_path:
        return
    gt = load_gt_rects(gt_path)
    res = np.array(preds, dtype=float)
    L = min(len(gt), len(res))
    gt = gt[:L, :]
    res = res[:L, :]

    precision_curve, success_curve, cle, auc, dp20 = ev.compile_results(gt, res)

    print("\n=== Evaluation (OTB-style) ===")
    print(f"Frames evaluated: {L}")
    print(f"DP@20 (Precision @ 20px): {dp20:.4f}")
    print(f"Average CLE (px):         {cle:.4f}")
    print(f"AUC (Success):            {auc:.4f}")
    # quick summaries
    print(f"Precision curve (first 5): {np.round(precision_curve[:5], 4)} ...")
    print(f"Success curve (first 5):   {np.round(success_curve[:5], 4)} ...")
    print("=============================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KCF demo: 1=webcam, 2=video+bbox, 3=frames+bbox")
    parser.add_argument("mode", type=int, choices=[1, 2, 3], help="1 webcam | 2 video+bbox | 3 frames+bbox")
    parser.add_argument("--camera", type=int, default=0, help="camera index for mode 1")
    parser.add_argument("--video", type=str, help="video file path for mode 2")
    parser.add_argument("--frames", type=str, help="frames folder for mode 3")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="initial bbox (modes 2/3)")
    parser.add_argument("--bbox-file", type=str, help="text file with x y w h (commas/spaces/tabs)")
    parser.add_argument("--gt", type=str, help="groundtruth_rect.txt (tabs supported: x\\ty\\tw\\th)")
    parser.add_argument("--save", type=str, help="path to save predicted bboxes (txt, tab-separated)")
    args = parser.parse_args()

    tracker = KCF_Tracker.KCFTracker(True, True, True)
    cv2.namedWindow("tracking")

    # ---- Mode 1: webcam (mouse ROI) ----
    if args.mode == 1:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Cannot open camera", file=sys.stderr)
            sys.exit(1)
        cv2.setMouseCallback("tracking", draw_boundingbox)
        preds = []
        inited = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if selectingObject:
                cv2.rectangle(frame, (int(ix), int(iy)), (int(cx), int(cy)), (0, 255, 255), 1)
            elif initTracking:
                cv2.rectangle(frame, (int(ix), int(iy)), (int(ix + w), int(iy + h)), (0, 255, 255), 2)
                tracker.init([float(ix), float(iy), float(w), float(h)], frame)
                preds.append([ix, iy, w, h])  # first frame prediction = init
                initTracking = False
                inited = True
                onTracking = True
            elif onTracking and inited:
                t0 = time()
                bbox = tracker.update(frame)
                t1 = time()
                preds.append(list(map(float, bbox)))
                draw_and_fps(frame, bbox, t0, t1)

            cv2.imshow("tracking", frame)
            c = cv2.waitKey(inteval) & 0xFF
            if c in (27, ord("q")):
                break

        cap.release()
        cv2.destroyAllWindows()
        maybe_save_predictions(args.save, preds)
        maybe_eval_and_print(args.gt, preds)

    # ---- Mode 2: video + bbox ----
    elif args.mode == 2:
        if not args.video:
            print("Mode 2 requires --video PATH and a bbox (--bbox/--bbox-file or --gt).", file=sys.stderr)
            sys.exit(2)
        bbox0 = ensure_init_bbox(args)
        if bbox0 is None:
            print("Provide bbox for mode 2 via --bbox X Y W H, --bbox-file file.txt, or --gt groundtruth_rect.txt", file=sys.stderr)
            sys.exit(2)

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("Cannot open video:", args.video, file=sys.stderr)
            sys.exit(1)

        preds = []
        inited = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if not inited:
                ix, iy, w, h = bbox0
                cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
                tracker.init([float(ix), float(iy), float(w), float(h)], frame)
                preds.append([float(ix), float(iy), float(w), float(h)])
                inited = True
                onTracking = True
            else:
                t0 = time()
                bbox = tracker.update(frame)
                t1 = time()
                preds.append(list(map(float, bbox)))
                draw_and_fps(frame, bbox, t0, t1)

            cv2.imshow("tracking", frame)
            c = cv2.waitKey(30) & 0xFF
            if c in (27, ord("q")):
                break

        cap.release()
        cv2.destroyAllWindows()
        maybe_save_predictions(args.save, preds)
        maybe_eval_and_print(args.gt, preds)

    # ---- Mode 3: frames folder + bbox ----
    elif args.mode == 3:
        if not args.frames:
            print("Mode 3 requires --frames FOLDER and a bbox (--bbox/--bbox-file or --gt).", file=sys.stderr)
            sys.exit(3)
        if not os.path.isdir(args.frames):
            print("Frames folder not found:", args.frames, file=sys.stderr)
            sys.exit(3)
        bbox0 = ensure_init_bbox(args)
        if bbox0 is None:
            print("Provide bbox for mode 3 via --bbox X Y W H, --bbox-file file.txt, or --gt groundtruth_rect.txt", file=sys.stderr)
            sys.exit(3)

        preds = []
        inited = False

        for frame, name in iter_frames_from_dir(args.frames):
            if not inited:
                ix, iy, w, h = bbox0
                cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
                tracker.init([float(ix), float(iy), float(w), float(h)], frame)
                preds.append([float(ix), float(iy), float(w), float(h)])
                inited = True
                onTracking = True
            else:
                t0 = time()
                bbox = tracker.update(frame)
                t1 = time()
                preds.append(list(map(float, bbox)))
                draw_and_fps(frame, bbox, t0, t1)

            cv2.imshow("tracking", frame)
            c = cv2.waitKey(30) & 0xFF
            if c in (27, ord("q")):
                break

        cv2.destroyAllWindows()
        maybe_save_predictions(args.save, preds)
        maybe_eval_and_print(args.gt, preds)