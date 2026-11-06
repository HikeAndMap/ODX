# [PATCH8] Overhead wires kept out of the orthophoto.
#
# Power and telephone wires are too thin to be reconstructed, so every photo paints its wires onto the surface below them,
# each photo somewhere else (a wire 7 m above the road, seen from 54 m by photos 20 m apart, lands 3 m apart). Where the
# orthophoto changes from one photo to the next, wires break off, jump or come twice. Under any spot, though, a wire is
# there in only some of the photos.
#
# The survey is cut into tiles. Each tile is projected from every photo that sees it whole within MAX_ANGLE of straight
# down onto the 2.5D mesh (the surface texturing paints on). On the surface, things that lie on it (markings, kerbs,
# roofs, GCPs, shadows) land on the same cells in every photo; a wire lands somewhere else in each. Per photo: the
# brightness evened out against the others, the difference with the median of up to REF_VIEWS photos spread as far apart
# as possible (with a tolerance of TOL cells for misalignment), and of that the pieces at least MINLEN long, widened by
# GROW cells. The result is a mask per photo and tile, in mesh coordinates, which texrecon reads (--masked_ground): a face
# is not painted from a photo whose mask has a wire under it, as long as a photo without one sees it.

import os
import math
import time
import struct
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

TILE_CELLS = 1280           # cells per tile side
CELL = 0.015                # metres per cell
TILE = TILE_CELLS * CELL    # 19.2 m
MASK_FACTOR = 4             # stored cells: 4 x 4 cells (6 cm), a wire anywhere in them marks them
MASK_CELLS = TILE_CELLS // MASK_FACTOR
MAX_ANGLE = 25.0            # degrees from straight down
REF_VIEWS = 16
MIN_VIEWS = 5
TOL = 2                     # cells of misalignment tolerated
THR = 16.0                  # grey levels
MINLEN = 0.3                # metres
GROW = 2                    # cells
CACHE = 48                  # photos kept decoded per worker
MAGIC = b'HAMWIRE1'

HEADER = ("# [PATCH8] wire masks v1: tile %d x %.3f m, mask %d, max %.0f deg, refs %d, min %d, tol %d, thr %.0f, "
          "minlen %.2f, grow %d") % (TILE_CELLS, CELL, MASK_FACTOR, MAX_ANGLE, REF_VIEWS, MIN_VIEWS, TOL, THR, MINLEN, GROW)


def quaternion_to_rotation(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def read_nvm(nvm_file):
    """Views of an NVM file: list of dict(name, path, f, R (world to camera), C)."""
    with open(nvm_file) as f:
        lines = [l.strip() for l in f.read().split("\n")]
    if len(lines) < 3 or lines[0] != "NVM_V3":
        raise Exception("%s does not seem to be a valid NVM file" % nvm_file)
    base = os.path.dirname(os.path.abspath(nvm_file))
    views = []
    for l in lines[3:3 + int(lines[2])]:
        p = l.split()
        if len(p) < 9:
            continue
        views.append({'name': os.path.basename(p[0]), 'path': p[0] if os.path.isabs(p[0]) else os.path.join(base, p[0]),
                      'f': float(p[1]), 'R': quaternion_to_rotation(*map(float, p[2:6])),
                      'C': np.array([float(p[6]), float(p[7]), float(p[8])])})
    return views


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def project(view, X, w, h):
    """Pixel coordinates (pixel centres at integers) of points X (..., 3) in an undistorted view of size w x h."""
    xc = (X - view['C']) @ view['R'].T
    z = xc[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = view['f'] * xc[..., 0] / z + w / 2.0 - 0.5
        v = view['f'] * xc[..., 1] / z + h / 2.0 - 0.5
    return u, v, z


def off_nadir(view, point):
    d = point - view['C']
    return math.degrees(math.acos(max(-1.0, min(1.0, -d[2] / max(1e-9, np.linalg.norm(d))))))


def spread(order, centres, n):
    """Up to n indices of `order` (most straight down first), each next one as far as possible from those chosen."""
    chosen = [order[0]]
    rest = list(order[1:])
    while rest and len(chosen) < n:
        d = [min(np.linalg.norm(centres[r][:2] - centres[c][:2]) for c in chosen) for r in rest]
        chosen.append(rest.pop(int(np.argmax(d))))
    return chosen


def tile_heights(vertices, x0, y1):
    """Heights of the cells of the tile whose north-west corner is (x0, y1), from the mesh vertices around it (linear
    interpolation, nearest vertex outside their hull). None when the mesh has no vertices there."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    m = 1.0
    sel = (vertices[:, 0] >= x0 - m) & (vertices[:, 0] <= x0 + TILE + m) & (vertices[:, 1] >= y1 - TILE - m) & (vertices[:, 1] <= y1 + m)
    v = vertices[sel]
    if len(v) < 10:
        return None
    n = MASK_CELLS
    xs = x0 + (np.arange(n) + 0.5) * (TILE / n)
    ys = y1 - (np.arange(n) + 0.5) * (TILE / n)
    gx, gy = np.meshgrid(xs, ys)
    z = LinearNDInterpolator(v[:, :2], v[:, 2])(gx, gy)
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = NearestNDInterpolator(v[:, :2], v[:, 2])(gx[bad], gy[bad])
    return cv2.resize(z.astype(np.float32), (TILE_CELLS, TILE_CELLS), interpolation=cv2.INTER_LINEAR)


def wire_mask(L, M):
    """Wire cells of one photo's tile L, given the median M of the reference photos (both evened out)."""
    k = np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)
    lo, hi = cv2.erode(M, k), cv2.dilate(M, k)
    cand = (np.maximum(lo - L, L - hi) > THR).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if num <= 1:
        return np.zeros_like(cand)
    keep = np.hypot(stats[:, 2], stats[:, 3]) * CELL >= MINLEN
    keep[0] = False
    wire = keep[lab].astype(np.uint8)
    if GROW:
        wire = cv2.dilate(wire, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1)))
    return wire


def even_out(Ls, ref):
    """Gain and offset per photo so that its large-scale brightness matches ref."""
    small = cv2.resize(ref, (100, 100), interpolation=cv2.INTER_AREA).ravel()
    out = []
    for L in Ls:
        a, b = np.polyfit(cv2.resize(L, (100, 100), interpolation=cv2.INTER_AREA).ravel(), small, 1)
        out.append((a * L + b).astype(np.float32))
    return out


class _Worker:
    def __init__(self, views, sizes, vertices):
        self.views, self.sizes, self.vertices = views, sizes, vertices
        self.centres = np.array([v['C'] for v in views])
        self.cache = OrderedDict()

    def image(self, i, reduce):
        key = (i, reduce)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        flag = cv2.IMREAD_REDUCED_GRAYSCALE_2 if reduce == 2 else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(self.views[i]['path'], flag)
        self.cache[key] = img
        if len(self.cache) > CACHE:
            self.cache.popitem(last=False)
        return img

    def tile(self, tx, ty, x_min, y_min):
        x0 = x_min + tx * TILE
        y1 = y_min + (ty + 1) * TILE
        H = tile_heights(self.vertices, x0, y1)
        if H is None:
            return []
        xs = x0 + (np.arange(TILE_CELLS) + 0.5) * CELL
        ys = y1 - (np.arange(TILE_CELLS) + 0.5) * CELL
        gx, gy = np.meshgrid(xs, ys)
        G = np.stack([gx, gy, H], -1)
        centre = G[TILE_CELLS // 2, TILE_CELLS // 2]
        corners = G[[0, 0, -1, -1, TILE_CELLS // 2], [0, -1, 0, -1, TILE_CELLS // 2]]

        seen = []
        d = centre - self.centres
        near = np.nonzero(np.hypot(d[:, 0], d[:, 1]) <= np.abs(d[:, 2]) * math.tan(math.radians(MAX_ANGLE)) + TILE)[0]
        for i in near:
            view = self.views[i]
            ang = off_nadir(view, centre)
            if ang > MAX_ANGLE:
                continue
            w, h = self.sizes[i]
            u, v, z = project(view, corners, w, h)
            if np.all(z > 0) and np.all((u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)):
                seen.append((ang, i))
        if len(seen) < MIN_VIEWS:
            return []
        seen.sort()
        order = [i for _, i in seen]

        Ls = []
        for i in order:
            view = self.views[i]
            w, h = self.sizes[i]
            gsd = np.linalg.norm(centre - view['C']) / view['f']
            reduce = 2 if 2 * gsd <= CELL * 1.05 else 1
            u, v, z = project(view, G, w, h)
            img = self.image(i, reduce)
            Ls.append(cv2.remap(img, (u / reduce).astype(np.float32), (v / reduce).astype(np.float32),
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32))
        refs = spread(list(range(len(order))), self.centres[order], REF_VIEWS)
        raw = np.median(np.stack([Ls[r] for r in refs]), 0).astype(np.float32)
        Ls = even_out(Ls, raw)
        M = np.median(np.stack([Ls[r] for r in refs]), 0).astype(np.float32)

        out = []
        for j, i in enumerate(order):
            wire = wire_mask(Ls[j], M)
            small = wire.reshape(MASK_CELLS, MASK_FACTOR, MASK_CELLS, MASK_FACTOR).max(axis=(1, 3))
            out.append((i, np.packbits(small.astype(np.uint8).ravel()).tobytes(), float(wire.mean())))
        return out


_worker = None


def _init(views, sizes, vertices):
    global _worker
    _worker = _Worker(views, sizes, vertices)


def _run(job):
    tiles, x_min, y_min = job
    result = []
    for tx, ty in tiles:
        result.append((tx, ty, _worker.tile(tx, ty, x_min, y_min)))
    return result


def read_mesh_vertices(mesh_file):
    import trimesh
    mesh = trimesh.load(mesh_file, process=False)
    return np.asarray(mesh.vertices, dtype=np.float64)


def is_current(masks_file, inputs):
    """True when the masks file was written after all inputs with the current settings."""
    side = masks_file + ".txt"
    if not (os.path.isfile(masks_file) and os.path.isfile(side)):
        return False
    t = os.path.getmtime(masks_file)
    if any(not os.path.isfile(f) or os.path.getmtime(f) > t for f in inputs):
        return False
    with open(side) as f:
        return f.readline().rstrip("\n") == HEADER


def write_masks(output_file, nvm_file, mesh_file, max_concurrency, log):
    """Find the wires of every view on every tile of the mesh and write the masks file. Returns a summary dict."""
    started = time.time()
    views = read_nvm(nvm_file)
    sizes = [image_size(v['path']) for v in views]
    vertices = read_mesh_vertices(mesh_file)
    x_min, y_min = float(vertices[:, 0].min()), float(vertices[:, 1].min())
    nx = int(math.ceil((vertices[:, 0].max() - x_min) / TILE))
    ny = int(math.ceil((vertices[:, 1].max() - y_min) / TILE))
    tiles = [(tx, ty) for ty in range(ny) for tx in range(nx)]
    workers = max(1, min(max_concurrency, 8))
    # contiguous strips of tiles per job, so that each worker reuses the photos it has decoded
    per_job = max(1, int(math.ceil(len(tiles) / float(workers * 4))))
    jobs = [(tiles[i:i + per_job], x_min, y_min) for i in range(0, len(tiles), per_job)]
    log.INFO("[PATCH8] Looking for overhead wires: %s views, %s tiles of %.1f m, %s workers" % (len(views), len(tiles), TILE, workers))

    entries = []
    done = 0
    shares = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(views, sizes, vertices)) as pool:
        for result in pool.map(_run, jobs):
            for tx, ty, masks in result:
                for i, bits, share in masks:
                    entries.append((tx, ty, i, bits))
                    shares.append(share)
            done += 1
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                elapsed = time.time() - started
                log.INFO("[PATCH8] Overhead wires: %s of %s parts (%.0f min, about %.0f min left)" % (
                    done, len(jobs), elapsed / 60.0, elapsed / done * (len(jobs) - done) / 60.0))

    tmp = output_file + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<dddd", x_min, y_min, TILE, TILE / MASK_CELLS))
        f.write(struct.pack("<iii", nx, ny, MASK_CELLS))
        f.write(struct.pack("<i", len(views)))
        for v in views:
            name = v['name'].encode('utf-8')
            f.write(struct.pack("<H", len(name)))
            f.write(name)
        f.write(struct.pack("<q", len(entries)))
        for tx, ty, i, bits in entries:
            f.write(struct.pack("<iii", tx, ty, i))
            f.write(bits)
    os.replace(tmp, output_file)
    with open(output_file + ".txt", "w") as f:
        f.write(HEADER + "\n")

    analysed = len(set((e[0], e[1]) for e in entries))
    summary = {'views': len(views), 'tiles': len(tiles), 'analysed': analysed, 'masks': len(entries),
               'wire_share': float(np.mean(shares)) if shares else 0.0, 'minutes': (time.time() - started) / 60.0}
    log.INFO("[PATCH8] Overhead wires: %s masks on %s of %s tiles, wires on %.1f %% of the photo area, took %.0f min" % (
        summary['masks'], analysed, len(tiles), 100 * summary['wire_share'], summary['minutes']))
    return summary
