# [PATCH12] The ground of the orthophoto painted pixel by pixel from the photos without a wire there.
#
# Texturing paints every triangle of the 2.5D mesh from one photo. PATCH8 keeps photos with a wire over a triangle out
# and cuts triangles smaller, but a triangle is at least 12 cm, neighbouring triangles come from different photos and
# flights, and the seams are evened out: pieces of wire stay, wire shadows break off where the flight changes, and the
# road looks patched. Yet nearly every spot of the ground is clean in many photos (on a road in the NL roof test, every
# spot in at least 5, mostly 20 to 30).
#
# So after the orthophoto is made, its ground is painted again, per cell of 1.5 cm: the median colour of the photos that
# see the cell within 25 deg of straight down, whose wire mask (wire_masks.py) checked the cell and found no wire, that
# have no vehicle box there (PATCH7) and that belong to the survey PATCH10 paints the spot from (reserve photos left out).
# Each photo's colour is first scaled per channel to the survey's median (one exposure factor per photo; a straight-line
# fit as the wire search uses for grey followed the wires and flattened the texture), and the result is brought to the orthophoto's own brightness over LOCAL_MEAN, so that it never stands out. Wires are
# gone wherever enough photos are clean; a wire shadow, which lies on the road, comes out whole, from most photos.
#
# Only the ground: cells the wire search checked after roofs and roof edges were left out (wire_ground.py), and not
# within WALL_GAP of a roof or roof edge, where the photos are projected onto the slope the mesh makes of a wall and
# their colours mix; the painting fades in over FEATHER. A cell with fewer than MIN_CLEAN clean photos keeps the
# orthophoto. Only colours change, never a height. Tested offline on the NL roof test road: the median of the clean
# photos removed the wires and the bundle along the kerb, and the per-pixel choice of one photo instead gave colour
# fringes, so the median is used.

import os
import math
import time
import struct
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

from modules import wire_masks as wm

WALL_GAP = 0.4              # metres from a roof or roof edge where the orthophoto stays as texturing made it ...
FEATHER = 0.15              # ... and over which the painting fades in
MIN_CLEAN = 3               # clean photos a cell needs
LOCAL_MEAN = 0.3            # metres: the painting keeps the orthophoto's brightness averaged over this
SURVEY_GOOD = 10.0          # [PATCH10] degrees: the survey rule of texrecon (survey_good_angle, survey_angle_margin)
SURVEY_MARGIN = 5.0
CACHE = 16                  # photos kept decoded per worker (colour)
GAIN = (0.7, 1.4)           # the exposure factor of a photo against the others stays within this


def read_masks(path):
    """(header dict, structured array of the mask records) of a HAMWIRE2 masks file."""
    with open(path, 'rb') as f:
        if f.read(8) != wm.MAGIC:
            raise Exception("%s is not a v2 wire masks file" % path)
        x_min, y_min, tile, cell = struct.unpack('<dddd', f.read(32))
        nx, ny, n = struct.unpack('<iii', f.read(12))
        (num_names,) = struct.unpack('<i', f.read(4))
        names = []
        for _ in range(num_names):
            (ln,) = struct.unpack('<H', f.read(2))
            names.append(f.read(ln).decode('utf-8'))
        (count,) = struct.unpack('<q', f.read(8))
        offset = f.tell()
    mb = (n * n + 7) // 8
    dt = np.dtype([('tx', '<i4'), ('ty', '<i4'), ('i', '<i4'), ('w', 'u1', (mb,)), ('c', 'u1', (mb,))])
    rec = np.memmap(path, dtype=dt, mode='r', offset=offset, shape=(count,))
    return {'x_min': x_min, 'y_min': y_min, 'tile': tile, 'cell': cell, 'nx': nx, 'ny': ny, 'n': n, 'names': names}, rec


def checked_any(head, rec):
    """Cells (mask grid of the whole survey, row 0 = north) that at least one photo checked."""
    n, nx, ny = head['n'], head['nx'], head['ny']
    out = np.zeros((ny * n, nx * n), bool)
    for k in range(len(rec)):
        tx, ty = int(rec['tx'][k]), int(rec['ty'][k])
        c = np.unpackbits(rec['c'][k])[:n * n].reshape(n, n) > 0
        r0 = (ny - 1 - ty) * n
        out[r0:r0 + n, tx * n:tx * n + n] |= c
    return out


def read_view_surveys(path, views):
    """[PATCH10] (rank, reserve) of every view from texrecon's view surveys file; all 0 without one."""
    ranks = np.zeros(len(views), int)
    reserve = np.zeros(len(views), bool)
    if not path or not os.path.isfile(path):
        return ranks, reserve
    entries = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and not line.startswith('#'):
                entries[p[0]] = (int(p[1]), int(p[2]) != 0)
    for k, v in enumerate(views):
        name = v['name']
        e = entries.get(name)
        cut = len(name)
        while e is None and cut > 0:
            cut = name.rfind('.', 0, cut)
            if cut <= 0:
                break
            e = entries.get(name[:cut])
        if e is not None:
            ranks[k], reserve[k] = e
    return ranks, reserve


def read_boxes(path, views):
    """[PATCH7] boxes (x0, y0, x1, y1 in pixels of the undistorted photo) of every view; none without a file."""
    boxes = [[] for _ in views]
    if not path or not os.path.isfile(path):
        return boxes
    index = {v['name']: k for k, v in enumerate(views)}
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 5 and p[0] in index:
                boxes[index[p[0]]].append(tuple(float(x) for x in p[1:5]))
    return boxes


class _Painter:
    def __init__(self, views, sizes, vertices, ranks, reserve, boxes, masks_path, max_angle):
        self.views, self.sizes, self.vertices = views, sizes, vertices
        self.ranks, self.reserve, self.boxes = ranks, reserve, boxes
        self.centres = np.array([v['C'] for v in views])
        self.head, self.rec = read_masks(masks_path)
        self.max_angle = max_angle
        self.cache = OrderedDict()

    def image(self, i, reduce):
        key = (i, reduce)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        img = cv2.imread(self.views[i]['path'], cv2.IMREAD_REDUCED_COLOR_2 if reduce == 2 else cv2.IMREAD_COLOR)
        self.cache[key] = img
        if len(self.cache) > CACHE:
            self.cache.popitem(last=False)
        return img

    def tile(self, tx, ty, records):
        """The painted colour (TILE_CELLS square, BGR uint8) and the cells that have one, or None."""
        n, m = wm.TILE_CELLS, wm.MASK_CELLS
        x0 = self.head['x_min'] + tx * wm.TILE
        y1 = self.head['y_min'] + (ty + 1) * wm.TILE
        H = wm.tile_heights(self.vertices, x0, y1)
        if H is None:
            return None
        xs = x0 + (np.arange(n) + 0.5) * wm.CELL
        ys = y1 - (np.arange(n) + 0.5) * wm.CELL
        gx, gy = np.meshgrid(xs, ys)
        G = np.stack([gx, gy, H], -1)
        Gs = G[wm.MASK_FACTOR // 2::wm.MASK_FACTOR, wm.MASK_FACTOR // 2::wm.MASK_FACTOR]
        centre = G[n // 2, n // 2]
        cos_max = math.cos(math.radians(wm.MAX_ANGLE))

        cols, valids, cleans, angles, order = [], [], [], [], []
        for k in records:
            i = int(self.rec['i'][k])
            view = self.views[i]
            w, h = self.sizes[i]
            checked = np.unpackbits(self.rec['c'][k])[:m * m].reshape(m, m) > 0
            if not checked.any():
                continue
            wire = np.unpackbits(self.rec['w'][k])[:m * m].reshape(m, m) > 0
            gsd = np.linalg.norm(centre - view['C']) / view['f']
            reduce = 2 if 2 * gsd <= wm.CELL * 1.05 else 1
            u, v, z = wm.project(view, G, w, h)
            dz = view['C'][2] - G[..., 2]
            dist = np.sqrt((G[..., 0] - view['C'][0]) ** 2 + (G[..., 1] - view['C'][1]) ** 2 + dz ** 2)
            valid = (z > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1) & (dz >= cos_max * dist)
            img = self.image(i, reduce)
            if img is None:
                continue
            P = cv2.remap(img, (u / reduce).astype(np.float32), (v / reduce).astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
            valid &= P.max(axis=2) >= 1      # the black fill of the undistorted photo
            clean = cv2.resize((checked & ~wire).astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0
            clean &= valid
            for b in self.boxes[i]:           # [PATCH7] a vehicle or person on the cell in this photo
                clean &= ~((u >= b[0]) & (u <= b[2]) & (v >= b[1]) & (v <= b[3]))
            ds = Gs - view['C']
            ang = np.degrees(np.arccos(np.clip(-ds[..., 2] / np.linalg.norm(ds, axis=-1), -1, 1))).astype(np.float32)
            vs = valid[wm.MASK_FACTOR // 2::wm.MASK_FACTOR, wm.MASK_FACTOR // 2::wm.MASK_FACTOR]
            cols.append(P); valids.append(valid); cleans.append(clean)
            angles.append(np.where(vs, ang, np.inf).astype(np.float32)); order.append(i)
        if len(order) < MIN_CLEAN:
            return None

        # [PATCH10] the survey of every cell (6 cm), texrecon's rule: the newest whose straightest photo is within the
        # larger of SURVEY_GOOD and the straightest photo of any survey + SURVEY_MARGIN (at most max_angle); else the
        # survey of the straightest photo. Reserve photos do not count.
        rk = self.ranks[order]
        rs = self.reserve[order]
        surveys = sorted(set(int(r) for r in rk))
        best = {r: np.min([a for a, rr, res in zip(angles, rk, rs) if rr == r and not res] or [np.full((m, m), np.inf, np.float32)], 0)
                for r in surveys}
        overall = np.min([best[r] for r in surveys], 0)
        limit = np.minimum(self.max_angle, np.maximum(SURVEY_GOOD, overall + SURVEY_MARGIN))
        chosen = np.full((m, m), -1, int)
        for r in reversed(surveys):          # oldest first, so that the newest within the limit wins
            chosen = np.where(best[r] <= limit, r, chosen)
        if len(surveys) > 1:
            arg = np.argmin(np.stack([best[r] for r in surveys]), 0)
            chosen = np.where(chosen < 0, np.array(surveys)[arg], chosen)
        else:
            chosen[:] = surveys[0]
        chosen = cv2.resize(chosen.astype(np.int32), (n, n), interpolation=cv2.INTER_NEAREST)

        # colours evened out per survey and channel, then the median of the clean photos of the cell's survey
        out = np.full((n, n, 3), np.nan, np.float32)
        count = np.zeros((n, n), np.int32)
        for r in surveys:
            group = [j for j in range(len(order)) if rk[j] == r and not rs[j]]
            if len(group) < MIN_CLEAN:
                continue
            refs = wm.spread(list(range(len(group))), self.centres[[order[j] for j in group]], wm.REF_VIEWS)
            even = []
            for c in range(3):
                ch = [np.where(valids[j], cols[j][..., c].astype(np.float32), np.nan) for j in group]
                ref = wm.median(np.stack([ch[q] for q in refs]))
                even.append([np.clip(np.nan_to_num(e), 0, 255).astype(np.uint8) for e in exposure(ch, ref, [valids[j] for j in group])])
                del ch
            use = [cleans[j] & (chosen == r) for j in group]
            cnt = np.sum(use, 0)
            for r0 in range(0, n, 128):
                sel = cnt[r0:r0 + 128] >= MIN_CLEAN
                if not sel.any():
                    continue
                for c in range(3):
                    st = np.stack([np.where(u_[r0:r0 + 128], e[r0:r0 + 128], np.nan).astype(np.float32) for u_, e in zip(use, even[c])])
                    med = wm.median(st)
                    blk = out[r0:r0 + 128, :, c]
                    blk[sel] = med[sel]
            count = np.where(chosen == r, cnt, count)
        have = np.isfinite(out[..., 0]) & (count >= MIN_CLEAN)
        if not have.any():
            return None
        col = np.clip(np.nan_to_num(out), 0, 255).astype(np.uint8)
        return col, have


_painter = None


def _init(*a):
    global _painter
    _painter = _Painter(*a)


def _run(job):
    tx, ty, records = job
    try:
        return tx, ty, _painter.tile(tx, ty, records), None
    except Exception as e:
        return tx, ty, None, "%s: %s" % (type(e).__name__, e)


def exposure(Ls, ref, valids):
    """Every photo times one factor (its exposure against ref, the median of the reference photos): the ratio of the
    medians over the cells both have, within GAIN. Not wire_masks.even_out's straight-line fit: over a road with wires the
    fit follows the wires, which only the photo has, and flattens the photo's own texture (a gain of 0.3 in a test)."""
    out = []
    rk = np.isfinite(ref)
    for L, v in zip(Ls, valids):
        both = v & rk
        a = 1.0
        if both.sum() > 1000:
            med_l = float(np.median(L[both]))
            if med_l > 1.0:
                a = float(np.clip(np.median(ref[both]) / med_l, GAIN[0], GAIN[1]))
        out.append((L * a).astype(np.float32))
    return out


def lowpass(a, mask, sigma):
    num = cv2.GaussianBlur(np.where(mask[..., None], a, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-3)


def paint(orthophoto, nvm_file, mesh_file, masks_file, ground_file, boxes_file, surveys_file, east_offset, north_offset,
          max_angle, max_concurrency, log):
    """Paint the ground of the orthophoto (GeoTIFF, changed in place) from the clean photos. Returns a summary dict, or
    None when it could not be done (the orthophoto is then left as it is)."""
    import rasterio
    from rasterio.windows import from_bounds, Window
    started = time.time()
    head, rec = read_masks(ground_file)
    head0, rec0 = read_masks(masks_file)
    n, m, cell = wm.TILE_CELLS, head['n'], head['cell']
    if m != wm.MASK_CELLS or abs(head['tile'] - wm.TILE) > 1e-6:
        log.WARNING("[PATCH12] The wire masks were made with other tiles, the orthophoto is left as it is")
        return None

    # roofs and roof edges: checked before wire_ground.py left them out, not after; the painting keeps WALL_GAP from them
    ground = checked_any(head, rec)
    wall = checked_any(head0, rec0) & ~ground
    dist = cv2.distanceTransform((~wall).astype(np.uint8), cv2.DIST_L2, 5) * cell
    ramp = np.clip((dist - WALL_GAP) / FEATHER, 0, 1).astype(np.float32) * ground

    views = wm.read_nvm(nvm_file)
    sizes = [wm.image_size(v['path']) for v in views]
    vertices = wm.read_mesh_vertices(mesh_file)
    ranks, reserve = read_view_surveys(surveys_file, views)
    boxes = read_boxes(boxes_file, views)

    by_tile = {}
    for k in range(len(rec)):
        by_tile.setdefault((int(rec['tx'][k]), int(rec['ty'][k])), []).append(k)
    jobs = [(tx, ty, ks) for (tx, ty), ks in sorted(by_tile.items(), key=lambda t: (t[0][1], t[0][0]))
            if ramp[(head['ny'] - 1 - ty) * m:(head['ny'] - ty) * m, tx * m:(tx + 1) * m].any()]
    workers = max(1, min(max_concurrency, 8))
    log.INFO("[PATCH12] Painting the ground pixel by pixel from the photos without a wire: %d tiles, %d workers" % (len(jobs), workers))

    painted = considered = 0
    failed = []
    done = 0
    with rasterio.open(orthophoto, 'r+') as dst:
        if dst.count < 3 or dst.dtypes[0] != 'uint8':
            log.WARNING("[PATCH12] The orthophoto is not 8-bit RGB, left as it is")
            return None
        res = dst.res[0]
        pad = LOCAL_MEAN * 2
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(views, sizes, vertices, ranks, reserve, boxes, ground_file, max_angle)) as pool:
            for tx, ty, result, error in pool.map(_run, jobs, chunksize=1):
                done += 1
                if error is not None:
                    failed.append((tx, ty, error))
                elif result is not None:
                    col, have = result
                    x0 = head['x_min'] + tx * wm.TILE
                    y1 = head['y_min'] + (ty + 1) * wm.TILE
                    r0 = (head['ny'] - 1 - ty) * m
                    rp = cv2.resize(ramp[r0:r0 + m, tx * m:(tx + 1) * m], (n, n), interpolation=cv2.INTER_LINEAR)
                    weight = cv2.GaussianBlur(have.astype(np.float32), (0, 0), 2.0) * have * rp
                    win = from_bounds(x0 + east_offset - pad, y1 - wm.TILE + north_offset - pad, x0 + wm.TILE + east_offset + pad,
                                      y1 + north_offset + pad, dst.transform).round_offsets().round_lengths()
                    win = win.intersection(Window(0, 0, dst.width, dst.height))
                    if win.width > 0 and win.height > 0:
                        O = np.transpose(dst.read([1, 2, 3], window=win), (1, 2, 0)).astype(np.float32)
                        alpha = dst.read(4, window=win) > 0 if dst.count >= 4 else np.ones(O.shape[:2], bool)
                        t = dst.window_transform(win)
                        cx = t.c + (np.arange(int(win.width)) + 0.5) * t.a
                        cy = t.f + (np.arange(int(win.height)) + 0.5) * t.e
                        mx, my = np.meshgrid(((cx - east_offset - x0) / wm.CELL - 0.5).astype(np.float32),
                                             ((y1 - (cy - north_offset)) / wm.CELL - 0.5).astype(np.float32))
                        # every orthophoto pixel belongs to one tile; colours are not sampled past its edge
                        inside = (mx >= -0.5) & (mx < n - 0.5) & (my >= -0.5) & (my < n - 0.5)
                        C = cv2.remap(col[..., ::-1].astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        Wt = cv2.remap(weight, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        hv = (cv2.remap(have.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0) & alpha
                        Wt = Wt * (hv & inside)
                        if (Wt > 0).any():
                            sig = LOCAL_MEAN / res
                            Ca = C + lowpass(O, hv, sig) - lowpass(C, hv, sig)
                            new = O * (1 - Wt[..., None]) + Ca * Wt[..., None]
                            dst.write(np.transpose(np.clip(new + 0.5, 0, 255).astype(np.uint8), (2, 0, 1)), [1, 2, 3], window=win)
                            painted += int((Wt > 0.5).sum())
                            considered += int(alpha.sum())
                if done % max(1, len(jobs) // 10) == 0 or done == len(jobs):
                    elapsed = time.time() - started
                    log.INFO("[PATCH12] Ground pixel by pixel: %d of %d tiles (%.0f min, about %.0f min left)" % (
                        done, len(jobs), elapsed / 60.0, elapsed / done * (len(jobs) - done) / 60.0))
    summary = {'tiles': len(jobs), 'painted': painted / max(1, considered), 'minutes': (time.time() - started) / 60.0,
               'failed': len(failed)}
    log.INFO("[PATCH12] Ground painted pixel by pixel on %d tiles: %.0f %% of the orthophoto there from the median of the "
             "photos without a wire (more than %.2f m from roofs and roof edges), took %.0f min" % (
                 len(jobs), 100 * summary['painted'], WALL_GAP, summary['minutes']))
    if failed:
        log.WARNING("[PATCH12] %d tile(s) not painted after an error, left as texturing made them (first: tile %d,%d, %s)" % (
            len(failed), failed[0][0], failed[0][1], failed[0][2]))
    return summary
