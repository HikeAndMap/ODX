# [PATCH7] Vehicles and people kept out of the orthophoto.
#
# On a busy road a vehicle stands on any given spot in many of the photos, often in most of them (a jeepney stop, a queue,
# two flights on different days), each time a different one. The photometric outlier removal of texturing then keeps the
# vehicles and drops the road, and the orthophoto is pieced together from halves of different vehicles. Here every
# undistorted photo goes through two YOLOX detectors (COCO: person, bicycle, car, motorcycle, bus, truck; the medium and the
# large model each find vehicles the other misses from above, so both are used); the boxes, with a margin, go to texrecon
# (--masked_boxes), which paints a face from a photo with a box on it only when no photo without one sees that face (a
# vehicle parked during all flights stays, there is nothing better).
#
# YOLOX: Megvii, Apache-2.0, https://github.com/Megvii-BaseDetection/YOLOX (yolox_m.onnx and yolox_l.onnx of release
# 0.1.1rc0, COCO weights).

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

MODEL_VERSION = "0.1.1rc0"
MODEL_NAMESPACE = "yolox"
MODELS = [  # file, URL, SHA-256 (checked when the node image is built, hikeandmap/fetch_models.py)
    ("yolox_m.onnx", "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx",
     "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f"),
    ("yolox_l.onnx", "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx",
     "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"),
]

CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
INPUT = 640                 # the models' input size (square)
DETECT_SIZE = 2560          # the longest side the photo is scaled to before it is cut into tiles of INPUT
OVERLAP = 192               # overlap between tiles, so that a vehicle cut by a tile border is whole in the next tile
SCORE_MIN = 0.15            # low on purpose: a false box only keeps that one photo off that spot
NMS_IOU = 0.45
MAX_BOX = 0.25              # boxes longer than this fraction of the photo's longest side are not vehicles
MARGIN = 0.15               # each box grows by this fraction of its longest side on every side (shadows, edges)
STRIDES = (8, 16, 32)
WORKERS = 1                 # photos at a time (measured on 16 cores: 1, 2, 4 or 8 at a time all take about 7 s per photo)

# First line of the boxes file: boxes found with other models or settings are found again
HEADER = ("# [PATCH7] boxes v1: %s (YOLOX %s), size %d, tiles %d/%d, score %.2f, nms %.2f, max %.2f, margin %.2f; "
          "image_name x0 y0 x1 y1 class score (pixels of the undistorted image, margin included)") % (
    "+".join(m[0] for m in MODELS), MODEL_VERSION, DETECT_SIZE, INPUT, OVERLAP, SCORE_MIN, NMS_IOU, MAX_BOX, MARGIN)


def tile_origins(length, size=INPUT, overlap=OVERLAP):
    """Start positions of tiles of `size` covering `length` with at least `overlap`."""
    if length <= size:
        return [0]
    n = int(np.ceil((length - overlap) / float(size - overlap)))
    step = (length - size) / float(n - 1)
    return [int(round(i * step)) for i in range(n)]


def _grids():
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        strides.append(np.full((g.shape[0], 1), s))
    return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32)


GRIDS, GRID_STRIDES = _grids()


def decode(raw):
    """YOLOX raw output (N, 85) -> boxes x0 y0 x1 y1 (tile pixels), scores, classes, for our classes only."""
    out = raw.copy()
    out[:, :2] = (out[:, :2] + GRIDS) * GRID_STRIDES
    out[:, 2:4] = np.exp(out[:, 2:4]) * GRID_STRIDES
    cls_scores = out[:, 5:] * out[:, 4:5]
    cls = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(cls)), cls]
    keep = (score >= SCORE_MIN) & np.isin(cls, list(CLASSES.keys()))
    out, cls, score = out[keep], cls[keep], score[keep]
    boxes = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 0] + out[:, 2] / 2, out[:, 1] + out[:, 3] / 2], 1)
    return boxes, score, cls


def nms(boxes, scores):
    """Indices kept by non-maximum suppression (all classes together: a jeepney found as bus and as truck is one box)."""
    if len(boxes) == 0:
        return []
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], SCORE_MIN, NMS_IOU)
    return [int(i) for i in np.array(idx).reshape(-1)]


def raw_boxes(session, img):
    """All boxes of one model on the scaled image, in scaled-image pixels."""
    sh, sw = img.shape[:2]
    name = session.get_inputs()[0].name
    all_boxes, all_scores, all_cls = [], [], []
    for y0 in tile_origins(sh):
        for x0 in tile_origins(sw):
            tile = img[y0:y0 + INPUT, x0:x0 + INPUT]
            padded = np.full((INPUT, INPUT, 3), 114, dtype=np.uint8)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None].astype(np.float32))
            raw = session.run(None, {name: blob})[0][0]
            boxes, scores, cls = decode(raw)
            if len(boxes):
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                all_boxes.append(boxes); all_scores.append(scores); all_cls.append(cls)
    return all_boxes, all_scores, all_cls


def drop_false_large(result, longest):
    """Leave out boxes too long to be a vehicle, and boxes that hold the centres of two or more smaller boxes (a false box
    over a stretch of road with several vehicles on it)."""
    kept = [b for b in result if max(b[2] - b[0], b[3] - b[1]) <= MAX_BOX * longest]
    out = []
    for b in kept:
        area = (b[2] - b[0]) * (b[3] - b[1])
        inside = 0
        for o in kept:
            if o is b or (o[2] - o[0]) * (o[3] - o[1]) >= area:
                continue
            cx, cy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                inside += 1
        if inside < 2:
            out.append(b)
    return out


def detect(sessions, image_bgr):
    """Boxes (x0, y0, x1, y1, score, class) in pixels of the image, from all models together, margin not yet added."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, DETECT_SIZE / float(max(w, h)))
    img = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else image_bgr
    all_boxes, all_scores, all_cls = [], [], []
    for session in sessions:
        b, s, c = raw_boxes(session, img)
        all_boxes += b; all_scores += s; all_cls += c
    if not all_boxes:
        return []
    boxes = np.concatenate(all_boxes); scores = np.concatenate(all_scores); cls = np.concatenate(all_cls)
    result = []
    for i in nms(boxes, scores):
        b = boxes[i] / scale
        result.append((max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3]), float(scores[i]), int(cls[i])))
    return drop_false_large(result, max(w, h))


def with_margin(box, width, height, margin=MARGIN):
    x0, y0, x1, y1 = box[:4]
    m = margin * max(x1 - x0, y1 - y0)
    return max(0.0, x0 - m), max(0.0, y0 - m), min(float(width), x1 + m), min(float(height), y1 + m)


def nvm_images(nvm_file):
    """(name as texrecon matches it, path) of every view of an NVM file."""
    with open(nvm_file) as f:
        lines = [l.strip() for l in f.read().split("\n")]
    if len(lines) < 3 or lines[0] != "NVM_V3":
        raise Exception("%s does not seem to be a valid NVM file" % nvm_file)
    base = os.path.dirname(os.path.abspath(nvm_file))
    result = []
    for l in lines[3:3 + int(lines[2])]:
        p = l.split()
        if not p:
            continue
        path = p[0] if os.path.isabs(p[0]) else os.path.join(base, p[0])
        result.append((os.path.basename(p[0]), path))
    return result


def is_current(boxes_file, nvm_file):
    """True when the boxes file was written for these undistorted photos with the current models and settings."""
    if not (os.path.isfile(boxes_file) and os.path.isfile(nvm_file)):
        return False
    if os.path.getmtime(boxes_file) < os.path.getmtime(nvm_file):
        return False
    with open(boxes_file) as f:
        return f.readline().rstrip("\n") == HEADER


def make_session(model_file, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model_file, sess_options=options, providers=['CPUExecutionProvider'])


def write_boxes(output_file, nvm_file, model_files, max_concurrency, log, workers=None):
    """Detect in every view of the NVM file and write one line per box: image_name x0 y0 x1 y1 class score (pixels,
    margin included). Returns (photos, photos with boxes, boxes)."""
    images = nvm_images(nvm_file)
    if workers is None:
        workers = WORKERS
    workers = max(1, min(workers, max_concurrency))
    threads = max(1, max_concurrency // workers)
    sessions = [make_session(m, threads) for m in model_files]
    lines = [None] * len(images)
    done = [0]
    lock = threading.Lock()
    started = time.time()
    counts = {}

    def run(i):
        name, path = images[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            log.WARNING("[PATCH7] Cannot read %s, no vehicles kept out of it" % path)
            lines[i] = []
        else:
            h, w = img.shape[:2]
            out = []
            for b in detect(sessions, img):
                x0, y0, x1, y1 = with_margin(b, w, h)
                out.append("%s %.1f %.1f %.1f %.1f %s %.3f" % (name, x0, y0, x1, y1, CLASSES[b[5]], b[4]))
            lines[i] = out
        with lock:
            for l in lines[i]:
                c = l.split()[5]
                counts[c] = counts.get(c, 0) + 1
            done[0] += 1
            if done[0] % 200 == 0 or done[0] == len(images):
                elapsed = time.time() - started
                log.INFO("[PATCH7] Vehicles and people: %s of %s photos (%.0f min, about %.0f min left)" % (
                    done[0], len(images), elapsed / 60.0, elapsed / done[0] * (len(images) - done[0]) / 60.0))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(len(images))))

    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        f.write(HEADER + "\n")
        for l in lines:
            for x in l:
                f.write(x + "\n")
    os.replace(tmp, output_file)
    with_boxes = sum(1 for l in lines if l)
    total = sum(len(l) for l in lines)
    log.INFO("[PATCH7] %s boxes in %s of %s photos (%s), took %.0f min" % (
        total, with_boxes, len(images), ", ".join("%s %s" % (counts[c], c) for c in sorted(counts)) or "none", (time.time() - started) / 60.0))
    return len(images), with_boxes, total
