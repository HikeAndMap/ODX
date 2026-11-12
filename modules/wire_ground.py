# [PATCH8] Overhead wires kept out of the orthophoto on the ground only, never on roofs or roof edges.
#
# The wire masks (wire_masks.py) mark far more than wires on roofs: corrugation, gutters and the edges themselves differ
# from photo to photo. On New Lucban roof edges were marked in 58 % of the views on average, roofs in 29 %. Where most
# views are marked, texrecon is left with a few slanted views from the other side, and those paint the roof edge onto the
# road beside it.
#
# So the masks are cleared wherever the 2.5D mesh is a roof or a roof edge, before texrecon reads them:
#   - a roof: the mesh more than ROOF_HEIGHT above the ground, from a ground grid of GROUND_CELL made with PDAL's simple
#     morphological filter (the --smrf-* settings odm_georeferencing classifies with later) on every GROUND_DECIMATION-th
#     point of the filtered point cloud, its gaps (under buildings) filled from around them;
#   - a roof edge: the mesh rising more than STEP_HEIGHT within STEP_DIST up to a roof, and EDGE_MARGIN around that.
# Those cells are written as not checked in every photo, so texrecon neither drops views nor cuts faces there: a face on
# a roof or an edge is textured as without PATCH8 (a wire over a roof stays as the photos show it).
#
# The wire search itself (wire_masks.bin) is not touched and stays reusable; this writes wire_masks_ground.bin from it.

import os
import math
import time
import struct
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

from modules import wire_masks

ROOF_HEIGHT = 1.5           # metres above the ground
STEP_HEIGHT = 0.5           # metres the mesh rises ...
STEP_DIST = 0.3             # ... within this many metres: a roof edge
EDGE_MARGIN = 0.5           # metres round a roof edge also left out
GROUND_CELL = 1.0           # metres
GROUND_DECIMATION = 10      # every n-th point of the point cloud for the ground
GROUND_FILL = 300.0         # metres searched round a gap in the ground grid
NODATA = -9999.0

HEADER = ("# [PATCH8] ground only v2: roof %.1f m above the ground, edge %.2f m within %.2f m plus %.2f m, ground grid "
          "%.1f m from every %d. point") % (ROOF_HEIGHT, STEP_HEIGHT, STEP_DIST, EDGE_MARGIN, GROUND_CELL, GROUND_DECIMATION)


def smrf_header(scalar, slope, threshold, window):
    return "# smrf scalar %s slope %s threshold %s window %s" % (scalar, slope, threshold, window)


def is_current(output_file, inputs, smrf):
    """True when the cleared masks were written after all inputs with the current settings."""
    side = output_file + ".txt"
    if not (os.path.isfile(output_file) and os.path.isfile(side)):
        return False
    t = os.path.getmtime(output_file)
    if any(not os.path.isfile(f) or os.path.getmtime(f) > t for f in inputs):
        return False
    with open(side) as f:
        return f.readline().rstrip("\n") == HEADER and f.readline().rstrip("\n") == smrf_header(*smrf)


def ground_grid(point_cloud, out_tif, smrf, log):
    """The ground grid (heights, west edge, north edge, cell) from the point cloud; None when it cannot be made."""
    from modules.dem import pdal
    import rasterio
    from rasterio.fill import fillnodata
    scalar, slope, threshold, window = smrf
    pipeline = {'pipeline': [
        {'type': 'readers.ply', 'filename': os.path.abspath(point_cloud)},
        {'type': 'filters.decimation', 'step': GROUND_DECIMATION},
        {'type': 'filters.smrf', 'scalar': scalar, 'slope': slope, 'threshold': threshold, 'window': window},
        {'type': 'filters.range', 'limits': 'Classification[2:2]'},
        {'type': 'writers.gdal', 'filename': out_tif, 'resolution': GROUND_CELL, 'output_type': 'min',
         'radius': GROUND_CELL * math.sqrt(2) / 2.0, 'data_type': 'float32', 'nodata': NODATA, 'gdaldriver': 'GTiff'},
    ]}
    if os.path.isfile(out_tif):
        os.remove(out_tif)
    pdal.run_pipeline(pipeline)
    if not os.path.isfile(out_tif):
        return None
    with rasterio.open(out_tif) as src:
        z = src.read(1).astype(np.float32)
        t = src.transform
    valid = (z != NODATA) & np.isfinite(z)
    if valid.sum() < 4:
        return None
    filled = fillnodata(z, mask=valid.astype(np.uint8), max_search_distance=GROUND_FILL / GROUND_CELL, smoothing_iterations=0)
    filled[~np.isfinite(filled) | (filled == NODATA)] = float(np.median(z[valid]))
    return filled.astype(np.float32), float(t.c), float(t.f), float(t.a), float(valid.mean())


def heights(vertices, x0, y1, size, n):
    """Mesh heights of n x n cells over the square of side `size` whose north-west corner is (x0, y1) (as
    wire_masks.tile_heights); None when the mesh has no vertices there."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    m = 1.0
    sel = (vertices[:, 0] >= x0 - m) & (vertices[:, 0] <= x0 + size + m) & (vertices[:, 1] >= y1 - size - m) & (vertices[:, 1] <= y1 + m)
    v = vertices[sel]
    if len(v) < 10:
        return None
    xs = x0 + (np.arange(n) + 0.5) * (size / n)
    ys = y1 - (np.arange(n) + 0.5) * (size / n)
    gx, gy = np.meshgrid(xs, ys)
    z = LinearNDInterpolator(v[:, :2], v[:, 2])(gx, gy)
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = NearestNDInterpolator(v[:, :2], v[:, 2])(gx[bad], gy[bad])
    return z.astype(np.float32), gx.astype(np.float32), gy.astype(np.float32)


def raised_cells(vertices, ground, x0, y1, tile, n):
    """Cells (n x n, row 0 north) of the tile at (x0, y1) that are a roof or a roof edge, as a boolean array; None when
    the mesh has no vertices there. ground: (heights, west, north, cell)."""
    cell = tile / n
    pad = int(math.ceil((STEP_DIST + EDGE_MARGIN) / cell)) + 2
    got = heights(vertices, x0 - pad * cell, y1 + pad * cell, tile + 2 * pad * cell, n + 2 * pad)
    if got is None:
        return None
    H, gx, gy = got
    G, gw, gn, gc = ground[:4]
    mx = ((gx - gw) / gc - 0.5).astype(np.float32)
    my = ((gn - gy) / gc - 0.5).astype(np.float32)
    ground_h = cv2.remap(G, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    roof = (H - ground_h) > ROOF_HEIGHT
    k = max(1, int(round(STEP_DIST / cell)))
    kernel = np.ones((k + 1, k + 1), np.uint8)
    top = cv2.dilate(H, kernel)
    # a step counts only when its top is a roof: walls beside roads, kerbs and bumps in the mesh of a road do not
    step = ((top - cv2.erode(H, kernel)) > STEP_HEIGHT) & ((top - ground_h) > ROOF_HEIGHT)
    r = int(math.ceil(EDGE_MARGIN / cell))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    edge = cv2.dilate(step.astype(np.uint8), disk) > 0
    out = roof | edge
    return out[pad:pad + n, pad:pad + n]


_vertices = None
_ground = None


def _init(vertices, ground):
    global _vertices, _ground
    _vertices, _ground = vertices, ground


def _run(job):
    tiles, x_min, y_min, tile, n = job
    out = []
    for tx, ty in tiles:
        try:
            r = raised_cells(_vertices, _ground, x_min + tx * tile, y_min + (ty + 1) * tile, tile, n)
            out.append((tx, ty, None if r is None else np.packbits(r.astype(np.uint8).ravel()).tobytes(), None))
        except Exception as e:
            out.append((tx, ty, None, "%s: %s" % (type(e).__name__, e)))
    return out


def write_cleared(output_file, masks_file, mesh_file, point_cloud, smrf, max_concurrency, log):
    """Write the masks of masks_file with roofs and roof edges left out (output_file). Returns True when written,
    False when the ground could not be made (the caller then uses masks_file as it is)."""
    started = time.time()
    ground_tif = os.path.join(os.path.dirname(output_file), "wire_masks_ground.tif")
    try:
        ground = ground_grid(point_cloud, ground_tif, smrf, log)
    except Exception as e:
        log.WARNING("[PATCH8] Cannot find the ground in %s (%s): overhead wires looked for on roofs too" % (point_cloud, e))
        return False
    if ground is None:
        log.WARNING("[PATCH8] No ground found in %s: overhead wires looked for on roofs too" % point_cloud)
        return False
    log.INFO("[PATCH8] Ground grid of %.0f m: %d x %d cells, %.0f %% of them with ground points, the rest filled from around them"
             % (GROUND_CELL, ground[0].shape[1], ground[0].shape[0], 100 * ground[4]))

    vertices = wire_masks.read_mesh_vertices(mesh_file)
    with open(masks_file, 'rb') as f:
        magic = f.read(8)
        if magic != wire_masks.MAGIC:
            log.WARNING("[PATCH8] %s is not a v2 masks file: overhead wires looked for on roofs too" % masks_file)
            return False
        x_min, y_min, tile, cell = struct.unpack('<dddd', f.read(32))
        nx, ny, n = struct.unpack('<iii', f.read(12))
    tiles = [(tx, ty) for ty in range(ny) for tx in range(nx)]
    workers = max(1, min(max_concurrency, 8))
    per_job = max(1, int(math.ceil(len(tiles) / float(workers * 4))))
    jobs = [(tiles[i:i + per_job], x_min, y_min, tile, n) for i in range(0, len(tiles), per_job)]
    raised = {}
    failed = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(vertices, ground)) as pool:
        for result in pool.map(_run, jobs):
            for tx, ty, bits, error in result:
                if error is not None:
                    failed.append((tx, ty, error))
                elif bits is not None:
                    raised[(tx, ty)] = np.frombuffer(bits, dtype=np.uint8)
    if failed:
        log.WARNING("[PATCH8] Roofs not found on %d tile(s), wires looked for there on roofs too (first: tile %d,%d, %s)"
                    % (len(failed), failed[0][0], failed[0][1], failed[0][2]))

    mb = (n * n + 7) // 8
    checked_before = checked_after = 0
    wire_before = wire_after = 0
    tmp = output_file + ".tmp"
    with open(masks_file, 'rb') as src, open(tmp, 'wb') as dst:
        dst.write(src.read(8 + 32 + 12))
        (nnames,) = struct.unpack('<i', src.read(4))
        dst.write(struct.pack('<i', nnames))
        for _ in range(nnames):
            (length,) = struct.unpack('<H', src.read(2))
            dst.write(struct.pack('<H', length))
            dst.write(src.read(length))
        (count,) = struct.unpack('<q', src.read(8))
        dst.write(struct.pack('<q', count))
        for _ in range(count):
            head = src.read(12)
            tx, ty, _view = struct.unpack('<iii', head)
            wire = np.frombuffer(src.read(mb), dtype=np.uint8)
            checked = np.frombuffer(src.read(mb), dtype=np.uint8)
            r = raised.get((tx, ty))
            checked_before += int(np.unpackbits(checked).sum())
            wire_before += int(np.unpackbits(wire & checked).sum())
            if r is not None:
                keep = np.bitwise_not(r)
                wire = wire & keep
                checked = checked & keep
            checked_after += int(np.unpackbits(checked).sum())
            wire_after += int(np.unpackbits(wire & checked).sum())
            dst.write(head)
            dst.write(wire.tobytes())
            dst.write(checked.tobytes())
    os.replace(tmp, output_file)
    with open(output_file + ".txt", "w") as f:
        f.write(HEADER + "\n" + smrf_header(*smrf) + "\n")

    roof_share = float(np.mean([np.unpackbits(r)[:n * n].mean() for r in raised.values()])) if raised else 0.0
    log.INFO("[PATCH8] Roofs and roof edges left out of the overhead wires: %.0f %% of the mesh, %.0f %% of the checked photo "
             "area; wires now on %.1f %% of the checked photo area (was %.1f %%), took %.1f min" % (
                 100 * roof_share, 100 * (1 - checked_after / max(1.0, float(checked_before))),
                 100 * wire_after / max(1.0, float(checked_after)), 100 * wire_before / max(1.0, float(checked_before)),
                 (time.time() - started) / 60.0))
    return True
