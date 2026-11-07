"""
PATCH4: keep every GCP visible in the orthophoto.

The texturing of the orthophoto picks, for every face of the mesh, the view that looks most straight down at it and
drops views whose colour differs from most other views. A GCP marked in only a few photos (painted later, covered by
a parked car on other days, too dark on another flight) then loses to the photos that do not show it.

For every GCP this module chooses one of the photos in which the GCP was marked (the mark proves it is visible there)
and writes a square around the GCP for texrecon's --forced_views option, which textures the faces inside the square
from that photo only. Cameras and geometry are not touched: the GCP is shown where the reconstruction puts it.

The candidates are compared on the same piece of ground: the square around the GCP is cut out of every marked photo
at the same scale and orientation (around the mark, with the camera's local projection). On that common patch:
- sharpness: variance of the Laplacian; at the common scale the photo noise of dark photos is averaged away,
- clean: how much the patch differs from the median of all candidates (a person, a car or a shadow that is only in
  some photos stands out); 1 for the cleanest candidate,
- straight down: viewing angle from vertical, 0 at MAX_VIEWING_ANGLE,
- centre: the mark's distance from the image centre (tie-breaker).

[PATCH10] With photos of several surveys (months, see surveys.py) the square comes from the newest survey with a marked
photo scoring at least min_score; only when none of the newest does, the best photo of any survey. Each survey's photos
are compared with each other for "clean" (the ground may have changed between surveys). A square whose photo is older
than the newest survey seeing the GCP's spot within max_angle is an older survey's square: it gets size_older instead
(0 = no square: the GCP still places the survey, but the orthophoto shows the newest survey there).
"""

import os
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from modules import log

# Candidates cut out and compared per GCP (the best ones by viewing angle and position in the image)
MAX_CANDIDATES = 10

# Score weights
WEIGHT_SHARPNESS = 0.35
WEIGHT_CLEAN = 0.35
WEIGHT_VERTICAL = 0.3
WEIGHT_CENTRE = 0.1

# Viewing angle (from vertical) at which the straight-down factor reaches 0
MAX_VIEWING_ANGLE = 45.0

# Resolution of the common patch, in metres per pixel
PATCH_RESOLUTION = 0.02


def quaternion_to_rotation(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def read_nvm_cameras(nvm_file):
    """
    :return dict { image file name -> (focal in pixels, world to camera rotation, camera centre) } in the mesh coordinates
    """
    with open(nvm_file) as f:
        lines = list(map(str.strip, f.read().split("\n")))

    if len(lines) < 3 or lines[0] != "NVM_V3":
        raise Exception("%s does not seem to be a valid NVM file" % nvm_file)

    cameras = {}
    num_images = int(lines[2])
    for l in lines[3:3 + num_images]:
        p = l.split()
        if len(p) < 9:
            continue
        rotation = quaternion_to_rotation(*map(float, p[2:6]))
        cameras[os.path.basename(p[0])] = (float(p[1]), rotation, np.array([float(p[6]), float(p[7]), float(p[8])]))
    return cameras


def nvm_name_of(filename, cameras):
    """
    The undistorted image keeps the photo's name plus an extension (282-01_x.JPG -> 282-01_x.JPG.tif)
    """
    if filename in cameras:
        return filename
    for name in cameras:
        if name.startswith(filename + "."):
            return name
    return None


def ground_to_pixel_jacobian(rotation, centre, focal_px, g):
    """
    Pixels per metre east and north at ground point g, as a 2 x 2 matrix (columns: east, north)
    """
    def project(x):
        c = rotation.dot(x - centre)
        return np.array([focal_px * c[0] / c[2], focal_px * c[1] / c[2]])
    step = 0.05
    p0 = project(g)
    return np.column_stack([(project(g + np.array([step, 0.0, 0.0])) - p0) / step,
                            (project(g + np.array([0.0, step, 0.0])) - p0) / step])


def common_patch(image_path, px, py, jacobian, size):
    """
    The square of size x size metres around the mark at (px, py), north up, PATCH_RESOLUTION metres per pixel,
    or None when the photo cannot be read or the square is not inside it
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    n = int(round(size / PATCH_RESOLUTION))
    s = PATCH_RESOLUTION
    # Patch pixel (j, i) -> ground offset (east, north) -> image pixel
    a = jacobian
    m = np.array([[a[0, 0] * s, -a[0, 1] * s, px + a[0, 0] * (s / 2 - size / 2) + a[0, 1] * (size / 2 - s / 2)],
                  [a[1, 0] * s, -a[1, 1] * s, py + a[1, 0] * (s / 2 - size / 2) + a[1, 1] * (size / 2 - s / 2)]])

    # The square must be inside the photo
    corners = np.array([[0, 0, 1], [n, 0, 1], [0, n, 1], [n, n, 1]], dtype=float).dot(m.T)
    h, w = image.shape[:2]
    if corners[:, 0].min() < 0 or corners[:, 1].min() < 0 or corners[:, 0].max() >= w or corners[:, 1].max() >= h:
        return None

    # Blur by the scale change first, so that downsampling averages instead of picking single (noisy) pixels
    x0, y0 = int(corners[:, 0].min()), int(corners[:, 1].min())
    x1, y1 = int(corners[:, 0].max()) + 2, int(corners[:, 1].max()) + 2
    crop = image[y0:y1, x0:x1]
    pixels_per_cell = math.sqrt(abs(np.linalg.det(a))) * s
    if pixels_per_cell > 1.0:
        crop = cv2.GaussianBlur(crop, (0, 0), 0.5 * pixels_per_cell)
    m[0, 2] -= x0
    m[1, 2] -= y0
    return cv2.warpAffine(crop, m, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def standardized(patch):
    p = patch.astype(np.float32)
    return (p - p.mean(axis=(0, 1))) / (p.std(axis=(0, 1)) + 1.0)


def gcp_label(entries):
    for e in entries:
        if e.extras:
            return e.extras.split()[0]
    e = entries[0]
    return "gcp_%.3f_%.3f" % (e.x, e.y)


def newest_rank_seeing(g, cameras, photos, survey_of, max_angle):
    """[PATCH10] The newest survey (lowest rank) with a photo that sees ground point g within max_angle of straight down."""
    best = None
    for filename, photo in photos.items():
        rank = survey_of.get(filename)
        if rank is None or (best is not None and rank >= best) or not photo.width or not photo.height:
            continue
        name = nvm_name_of(filename, cameras)
        if name is None:
            continue
        focal, rotation, c = cameras[name]
        d = c - g
        distance = float(np.linalg.norm(d))
        if distance <= 0 or math.degrees(math.acos(max(-1.0, min(1.0, d[2] / distance)))) > max_angle:
            continue
        x = rotation.dot(g - c)
        if x[2] <= 0:
            continue
        focal_px = (photo.focal_ratio * max(photo.width, photo.height)) if photo.focal_ratio else focal
        u, v = focal_px * x[0] / x[2], focal_px * x[1] / x[2]
        if abs(u) <= photo.width / 2.0 and abs(v) <= photo.height / 2.0:
            best = rank
    return best


def write_forced_views(output_file, reconstruction, images_path, nvm_file, size, max_workers,
                       survey_of=None, labels=None, size_older=None, min_score=0.0, max_angle=25.0):
    """
    Writes the squares file for texrecon's --forced_views option.

    :param size side of the square around each GCP, in metres
    :param survey_of [PATCH10] filename -> survey rank (0 = the newest), with labels the survey names; None = one survey
    :param size_older [PATCH10] side of the square when its photo is older than the newest survey there (0 = none)
    :param min_score [PATCH10] the newest survey's best photo must score this much, else the best of any survey is taken
    :return number of squares written
    """
    several = survey_of is not None and labels is not None and len(labels) > 1
    if size_older is None or not several:
        size_older = size
    # the photos are compared on a square of this size (the GCP squares themselves may be smaller or none)
    cut_size = size if size > 0 else (size_older if size_older > 0 else 2.0)
    cameras = read_nvm_cameras(nvm_file)
    photos = {p.filename: p for p in reconstruction.photos}
    off_x, off_y = reconstruction.georef.utm_east_offset, reconstruction.georef.utm_north_offset

    # GCPs, each with the photos it is marked in
    gcps = {}
    for e in reconstruction.gcp.iter_entries():
        gcps.setdefault(e.coords_key(), []).append(e)

    candidates = {}
    for key, entries in gcps.items():
        e0 = entries[0]
        g = np.array([e0.x - off_x, e0.y - off_y, e0.z])
        found = []
        for e in entries:
            name = nvm_name_of(e.filename, cameras)
            photo = photos.get(e.filename)
            if name is None or photo is None or not photo.width or not photo.height:
                continue

            focal, rotation, c = cameras[name]
            d = c - g
            distance = float(np.linalg.norm(d))
            if distance <= 0:
                continue
            angle = math.degrees(math.acos(max(-1.0, min(1.0, d[2] / distance))))
            vertical = max(0.0, 1.0 - angle / MAX_VIEWING_ANGLE)

            w, h = photo.width, photo.height
            from_centre = math.hypot(e.px - w / 2.0, e.py - h / 2.0) / math.hypot(w / 2.0, h / 2.0)
            centre = max(0.0, 1.0 - from_centre)

            # The marks are in the original photo: its focal length in pixels, the camera's rotation from the reconstruction
            focal_px = (photo.focal_ratio * max(w, h)) if photo.focal_ratio else focal
            jacobian = ground_to_pixel_jacobian(rotation, c, focal_px, g)

            found.append({'entry': e, 'angle': angle, 'vertical': vertical, 'from_centre': from_centre,
                          'centre': centre, 'jacobian': jacobian, 'patch': None})

        found.sort(key=lambda c: c['vertical'] + 0.2 * c['centre'], reverse=True)
        candidates[key] = (g, found[:MAX_CANDIDATES], len(entries), len(found))

    # Cut the square out of the candidate photos
    jobs = [c for (_, found, _, _) in candidates.values() for c in found]
    def cut(c):
        e = c['entry']
        c['patch'] = common_patch(os.path.join(images_path, e.filename), e.px, e.py, c['jacobian'], cut_size)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        list(executor.map(cut, jobs))

    lines = ["# PATCH4: one square per GCP: label image_name xmin ymin xmax ymax (mesh coordinates = UTM minus the offset %s %s)" % (off_x, off_y)]
    written = 0
    for key, (g, found, num_marked, num_reconstructed) in sorted(candidates.items(), key=lambda kv: gcp_label(gcps[kv[0]])):
        label = gcp_label(gcps[key])
        found = [c for c in found if c['patch'] is not None]
        if not found:
            log.WARNING("[PATCH4] %s: none of its %s marked photos holds the whole square (%s in the reconstruction), left as is" % (label, num_marked, num_reconstructed))
            continue

        for c in found:
            c['sharpness'] = float(cv2.Laplacian(cv2.cvtColor(c['patch'], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            c['rank'] = survey_of.get(c['entry'].filename, 0) if several else 0
        best_sharpness = max(c['sharpness'] for c in found) or 1.0

        # [PATCH10] "clean" within each survey: the ground may have changed from one survey to the next
        for rank in set(c['rank'] for c in found):
            group = [c for c in found if c['rank'] == rank]
            if len(group) >= 3:
                z = [standardized(c['patch']) for c in group]
                median = np.median(np.stack(z), axis=0)
                for c, zc in zip(group, z):
                    c['clutter'] = float(np.abs(zc - median).mean())
                least = min(c['clutter'] for c in group)
                for c in group:
                    c['clean'] = (least + 0.05) / (c['clutter'] + 0.05)
            else:
                for c in group:
                    c['clean'] = 1.0

        for c in found:
            c['sharpness_rel'] = c['sharpness'] / best_sharpness
            c['score'] = (WEIGHT_SHARPNESS * c['sharpness_rel'] + WEIGHT_CLEAN * c['clean'] +
                          WEIGHT_VERTICAL * c['vertical'] + WEIGHT_CENTRE * c['centre'])

        # [PATCH10] the newest survey whose best photo scores at least min_score, else the best photo of any survey
        best = None
        for rank in sorted(set(c['rank'] for c in found)):
            candidate = max((c for c in found if c['rank'] == rank), key=lambda c: c['score'])
            if candidate['score'] >= min_score:
                best = candidate
                break
        below = best is None
        if below:
            best = max(found, key=lambda c: c['score'])
        e = best['entry']

        square = size
        older = False
        survey_note = ""
        if several:
            newest_here = newest_rank_seeing(g, cameras, photos, survey_of, max_angle)
            older = newest_here is not None and best['rank'] > newest_here
            survey_note = "survey %s%s%s, " % (labels[best['rank']],
                                                " (older than %s, the newest here)" % labels[newest_here] if older else "",
                                                ", no photo of a newer survey scores %.2f" % min_score if below and best['rank'] > 0 else "")
            if older:
                square = size_older
        summary = ("%sscore %.2f: sharpness %.2f, clean %.2f, %.1f deg from vertical, mark %.0f %% from the image centre; "
                   "%s of %s marked photos compared" % (survey_note, best['score'], best['sharpness_rel'], best['clean'], best['angle'],
                                                        best['from_centre'] * 100, len(found), num_marked))
        if square <= 0:
            log.INFO("[PATCH10] %s: no square (%s is 0), the GCP still places the survey (%s: %s)" % (
                label, "gcp-texture-size-older" if older else "gcp-texture-size", e.filename, summary))
            continue
        half = square / 2.0
        lines.append("%s %s %.3f %.3f %.3f %.3f" % (label, e.filename, g[0] - half, g[1] - half, g[0] + half, g[1] + half))
        lines.append("#   " + summary)
        log.INFO("[PATCH4] %s: %s (%s%s)" % (label, e.filename, summary, "; square %.1f m" % square if square != size else ""))
        written += 1

    with open(output_file, 'w') as f:
        f.write("\n".join(lines) + "\n")
    log.INFO("[PATCH4] %s GCP squares of %.1f x %.1f m written to %s" % (written, size, size, output_file))
    return written
