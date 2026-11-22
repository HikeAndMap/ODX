# [PATCH6] Photos with and without RTK in one run, with or without GCPs.
#
# 1. A photo that carries its own position accuracy (DJI RTK XMP, eBee XMP, or a geo file with accuracy columns) keeps it:
#    --gps-accuracy (and the new --gps-accuracy-vertical) apply only to the photos without one. Stock ODX overwrote the
#    accuracy of every photo with --gps-accuracy, RTK photos included. A DJI photo counts as RTK only with an RTK fixed
#    solution (RtkFlag 50, read in photo.py): single point (16) and float (34) report centimetres but are metres off.
# 2. Photos without their own accuracy get --gps-accuracy horizontally and --gps-accuracy-vertical vertically (0 = the same
#    as --gps-accuracy, as before). A consumer drone's absolute height is often tens of metres off, far more than its
#    horizontal position.
# 3. --rtk-z-offset is added to the heights of the RTK photos only: their heights hang from the height the RTK base
#    broadcasts, which may differ from the height system of the GCPs by a fixed amount. The other photos keep theirs (their
#    height error differs per flight, so no single offset fits them).
# 4. With a GCP file, stock ODX switches the photo positions off altogether (bundle_use_gps: no). When the run has RTK
#    photos (at least MIN_RTK_PHOTOS with an accuracy of RTK_MAX_METRES or better), their positions now stay in the bundle
#    adjustment next to the GCPs, and the photos without RTK are given an accuracy of NO_PULL_METRES, so that they do not
#    pull on positions at all (as before) - their GPS frame may differ from the GCPs' by metres.
# 5. Without a GCP file but with RTK photos as anchors, the heights of the photos without RTK do not count either
#    (vertical accuracy NO_PULL_METRES); their horizontal position still counts with --gps-accuracy.
# 6. With GCPs, the console shows how far the RTK photos' heights are from the heights the GCPs give them.

import statistics
import math

MIN_RTK_PHOTOS = 3
RTK_MAX_METRES = 1.0          # after ODX's doubling of the reported RTK standard deviation
NO_PULL_METRES = 10000.0
RTK_FIXED_FLAG = 50           # DJI RtkFlag: 0 no position, 16 single point, 34 float, 50 fixed
CHECK_WARN_METRES = 0.5       # the RTK photos' heights differ from the GCPs' by more than this: warn

SOURCE_PHOTO = 'photo'        # the accuracy came with the photo (XMP or geo file)
SOURCE_OPTION = 'option'      # the accuracy was set from --gps-accuracy / --gps-accuracy-vertical


def _has_gps(p):
    return getattr(p, 'latitude', None) is not None and getattr(p, 'longitude', None) is not None


def _not_fixed(p):
    """A DJI photo that reports RTK without a fixed solution."""
    flag = getattr(p, 'rtk_flag', None)
    return flag is not None and flag != RTK_FIXED_FLAG


def has_own_accuracy(p):
    if _not_fixed(p):
        return False
    source = getattr(p, 'gps_accuracy_source', None)
    if source is not None:
        return source == SOURCE_PHOTO
    # A photo list from before [PATCH6] (a restart of an older task): an accuracy this good can only have come with the photo
    return getattr(p, 'gps_xy_stddev', None) is not None and p.gps_xy_stddev <= RTK_MAX_METRES


def is_rtk(p):
    return has_own_accuracy(p) and p.gps_xy_stddev is not None and p.gps_xy_stddev <= RTK_MAX_METRES


def vertical_accuracy(gps_accuracy, gps_accuracy_vertical):
    return gps_accuracy_vertical if gps_accuracy_vertical and gps_accuracy_vertical > 0 else gps_accuracy


def mark_sources(photos):
    """Dataset stage, after the geo file and before any override: remember which photos brought their own accuracy."""
    for p in photos:
        if getattr(p, 'gps_accuracy_source', None) is None:
            own = getattr(p, 'gps_xy_stddev', None) is not None and not _not_fixed(p)
            p.gps_accuracy_source = SOURCE_PHOTO if own else None


def set_accuracy_without_own(photos, gps_accuracy, gps_accuracy_vertical):
    """Dataset stage: the photos without their own accuracy get the option values. Returns (kept, set) counts."""
    vertical = vertical_accuracy(gps_accuracy, gps_accuracy_vertical)
    kept = changed = 0
    for p in photos:
        if not _has_gps(p):
            continue
        if has_own_accuracy(p):
            kept += 1
        else:
            p.gps_xy_stddev = gps_accuracy
            p.gps_z_stddev = vertical
            p.gps_accuracy_source = SOURCE_OPTION
            changed += 1
    return kept, changed


def summary(photos, gps_accuracy, gps_accuracy_vertical):
    """The [PATCH6] console line about the photo positions."""
    own = [p for p in photos if _has_gps(p) and has_own_accuracy(p)]
    others = [p for p in photos if _has_gps(p) and not has_own_accuracy(p)]
    not_fixed = sum(1 for p in photos if _has_gps(p) and _not_fixed(p))
    parts = []
    if own:
        xy = statistics.median(p.gps_xy_stddev for p in own)
        z = statistics.median((p.gps_z_stddev if p.gps_z_stddev is not None else p.gps_xy_stddev) for p in own)
        rtk = sum(1 for p in own if is_rtk(p))
        parts.append("%s with their own accuracy, kept (median %.3f m horizontal, %.3f m vertical; %s of them RTK)" % (len(own), xy, z, rtk))
    if others:
        text = "%s without, %.1f m horizontal and %.1f m vertical" % (
            len(others), gps_accuracy, vertical_accuracy(gps_accuracy, gps_accuracy_vertical))
        if not_fixed:
            text += " (%s of them report RTK without a fix, RtkFlag not 50)" % not_fixed
        parts.append(text)
    if not parts:
        return "[PATCH6] Photo positions: no photo has a GPS position"
    return "[PATCH6] Photo positions: " + "; ".join(parts)


def rtk_count(photos):
    return sum(1 for p in photos if _has_gps(p) and is_rtk(p))


def has_rtk_anchors(photos):
    """True when the run has enough RTK photos for their positions to anchor it."""
    return rtk_count(photos) >= MIN_RTK_PHOTOS


def keep_gps_with_gcp(photos):
    """With a GCP file: True when the run has enough RTK photos for their positions to stay in the adjustment."""
    return has_rtk_anchors(photos)


def apply_rtk_z_offset(photos, offset):
    """--rtk-z-offset: added to the heights of the RTK photos only (in memory). Returns how many."""
    n = 0
    for p in photos:
        if _has_gps(p) and is_rtk(p) and getattr(p, 'altitude', None) is not None:
            p.altitude += offset
            n += 1
    return n


def silence_without_rtk(photos):
    """With a GCP file and RTK photos: the photos without RTK do not pull on positions. In memory only (images.json keeps
    their real values). Returns how many."""
    n = 0
    for p in photos:
        if _has_gps(p) and not is_rtk(p):
            p.gps_xy_stddev = p.gps_z_stddev = NO_PULL_METRES
            n += 1
    return n


def free_heights_without_rtk(photos):
    """Without a GCP file, with RTK photos as anchors: the heights of the photos without RTK do not count, their horizontal
    position does. In memory only. Returns how many."""
    n = 0
    for p in photos:
        if _has_gps(p) and not is_rtk(p):
            p.gps_z_stddev = NO_PULL_METRES
            n += 1
    return n


def rtk_differences(photos, centres):
    """centres: {filename: (lat, lon, alt)} of the reconstructed camera centres. Returns a list of (filename, horizontal m,
    height m) for the RTK photos, height = photo height minus reconstructed height (positive: the photo says higher)."""
    out = []
    for p in photos:
        if not (_has_gps(p) and is_rtk(p)) or getattr(p, 'altitude', None) is None:
            continue
        c = centres.get(p.filename)
        if c is None:
            continue
        lat, lon, alt = c
        north = (p.latitude - lat) * 111320.0
        east = (p.longitude - lon) * 111320.0 * math.cos(math.radians(lat))
        out.append((p.filename, math.hypot(east, north), p.altitude - alt))
    return out


def check_text(diffs, used):
    """The [PATCH6] console line of the height check, and whether to warn."""
    if not diffs:
        return None, False
    dz = statistics.median(d[2] for d in diffs)
    dxy = statistics.median(d[1] for d in diffs)
    lo = min(d[2] for d in diffs)
    hi = max(d[2] for d in diffs)
    how = "used together with the GCPs" if used else "not used in the adjustment, so an independent check"
    text = "[PATCH6] RTK photo heights vs the GCPs: median %+.2f m (%+.2f to %+.2f), horizontal median %.2f m, %s RTK photos (%s)" % (
        dz, lo, hi, dxy, len(diffs), how)
    warn = abs(dz) > CHECK_WARN_METRES
    if warn:
        text += "; more than %.1f m: check --rtk-z-offset (the RTK base's height system differs from the GCPs')" % CHECK_WARN_METRES
    return text, warn


def log_rtk_check(opensfm_path, photos, used):
    """OpenSfM stage, with GCPs, while reconstruction.json is still topocentric: log the height check."""
    from opensfm.dataset import DataSet
    from modules import log

    ds = DataSet(opensfm_path)
    reference = ds.load_reference()
    centres = {}
    for rec in ds.load_reconstruction():
        for shot_id, shot in rec.shots.items():
            x, y, z = shot.pose.get_origin()
            centres[shot_id] = reference.to_lla(x, y, z)
    text, warn = check_text(rtk_differences(photos, centres), used)
    if text is None:
        return
    if warn:
        log.WARNING(text)
    else:
        log.INFO(text)


# [PATCH14] Several surveys, several drones, GCPs marked in only some of them.
#
# With a GCP file stock ODX switches every photo's GPS off (bundle_use_gps: no): the GCPs place the model. A survey whose photos
# have no GCP mark - a newer flight over an older, controlled survey, often with another drone - is then held only by its matches
# with the other surveys. When those are weak or one-sided (another season, another camera, overlap along one edge) the survey
# can turn as a whole: at Burnham Park (2026-11) a Matrice 4E flight was folded 34 degrees around one edge and ended up to 210 m
# below its own GPS. Now:
# 1. A survey is the photos of one month and one camera (make + model).
# 2. With a GCP file, the photos of a survey that has no GCP mark keep their GPS in the bundle adjustment with their own accuracy
#    (--gps-accuracy without RTK). The photos of surveys with GCP marks do not pull on positions (accuracy NO_PULL_METRES, as
#    [PATCH6] already did next to RTK photos); RTK photos keep theirs.
# 3. Each camera's GPS may sit in another height system than the GCPs (a consumer drone's barometric height, the Matrice 4E's
#    ellipsoidal height: tens of metres apart). OpenSfM's GPS bias (bundle_compensate_gps_bias) solves an offset per camera; it is
#    limited to a shift (bundle_gps_bias_translation_only, our OpenSfM fork), because a rotation would let the survey tilt again.
# 4. After the reconstruction, the console says per survey how far its cameras are from their GPS and how much the survey is
#    tilted against its GPS - a warning above SURVEY_TILT_WARN_DEGREES.

SURVEY_TILT_WARN_DEGREES = 3.0
SURVEY_MIN_PHOTOS = 10
SURVEY_MIN_SPREAD_METRES = 20.0   # a survey narrower than this in both directions has no plane to compare


def survey_key(p):
    """[PATCH14] The survey of a photo: its month (as [PATCH10] groups them) and its camera."""
    from modules import surveys
    camera = ("%s %s" % (getattr(p, 'camera_make', '') or '', getattr(p, 'camera_model', '') or '')).strip() or 'unknown camera'
    return "%s %s" % (surveys.survey_label(surveys.capture_time(p)), camera)


def marked_filenames(gcp):
    """[PATCH14] The photos with a GCP mark in the GCP file (a GCPFile), or an empty set."""
    if gcp is None or not gcp.exists():
        return set()
    return {e.filename for e in gcp.iter_entries()}


def surveys_without_marks(photos, marked):
    """[PATCH14] The surveys (keys) that have photos with GPS but no GCP mark in any of their photos, in order of their keys."""
    with_marks = {survey_key(p) for p in photos if p.filename in marked}
    return sorted({survey_key(p) for p in photos if _has_gps(p)} - with_marks)


def keep_unmarked_surveys(photos, marked):
    """[PATCH14] With a GCP file: True when a survey without GCP marks keeps its GPS (and the GPS bias is solved per camera)."""
    return len(surveys_without_marks(photos, marked)) > 0


def silence_marked_surveys(photos, marked):
    """[PATCH14] With a GCP file and surveys without marks: the photos of the surveys WITH marks, RTK photos excepted, do not pull
    on positions (the GCPs place them). In memory only. Returns how many."""
    unmarked = set(surveys_without_marks(photos, marked))
    n = 0
    for p in photos:
        if _has_gps(p) and not is_rtk(p) and survey_key(p) not in unmarked:
            p.gps_xy_stddev = p.gps_z_stddev = NO_PULL_METRES
            n += 1
    return n


def unmarked_summary(photos, marked):
    """[PATCH14] The console line: which surveys keep their GPS."""
    unmarked = surveys_without_marks(photos, marked)
    counts = {k: sum(1 for p in photos if _has_gps(p) and survey_key(p) == k) for k in unmarked}
    return "[PATCH14] Surveys without GCP marks keep their GPS in the bundle adjustment (with a height offset solved per camera): %s" % (
        ", ".join("%s (%s photos)" % (k, counts[k]) for k in unmarked))


def _plane_normal(points):
    """Unit normal of the plane through points (n x 3) and the two largest spreads (m), or None when too few or too narrow."""
    import numpy as np
    a = np.array(points, dtype=float)
    if len(a) < 3:
        return None
    a = a - a.mean(axis=0)
    w, v = np.linalg.eigh(a.T.dot(a) / len(a))
    spreads = np.sqrt(np.maximum(w, 0.0))
    if spreads[1] * 2 < SURVEY_MIN_SPREAD_METRES:
        return None
    n = v[:, 0]
    return n / np.linalg.norm(n) * (1 if n[2] >= 0 else -1)


def survey_differences(photos, centres_xyz, gps_xyz):
    """[PATCH14] Per survey: photos, median horizontal and vertical distance of the cameras from their GPS (GPS minus camera,
    positive = the GPS says higher) and the tilt between the plane of the cameras and the plane of their GPS positions (None when
    there is no plane). centres_xyz / gps_xyz: {filename: (x, y, z)} in one local metric frame."""
    import numpy as np
    out = []
    by_survey = {}
    for p in photos:
        if p.filename in centres_xyz and p.filename in gps_xyz:
            by_survey.setdefault(survey_key(p), []).append(p.filename)
    for key in sorted(by_survey):
        names = by_survey[key]
        if len(names) < SURVEY_MIN_PHOTOS:
            continue
        c = np.array([centres_xyz[n] for n in names])
        g = np.array([gps_xyz[n] for n in names])
        d = g - c
        horizontal = float(np.median(np.hypot(d[:, 0], d[:, 1])))
        vertical = float(np.median(d[:, 2]))
        nc, ng = _plane_normal(c), _plane_normal(g)
        tilt = None
        if nc is not None and ng is not None:
            tilt = float(np.degrees(np.arccos(min(1.0, abs(float(nc.dot(ng)))))))
        out.append((key, len(names), horizontal, vertical, tilt))
    return out


def survey_check_lines(diffs, marked_keys):
    """[PATCH14] Console lines (text, warn) of the survey check."""
    lines = []
    for key, n, horizontal, vertical, tilt in diffs:
        role = "placed by its GCP marks" if key in marked_keys else "placed by its own GPS"
        tilt_text = "tilt not measurable (too narrow)" if tilt is None else "tilted %.1f deg against its GPS" % tilt
        warn = tilt is not None and tilt > SURVEY_TILT_WARN_DEGREES
        text = "[PATCH14] Survey %s, %s photos (%s): cameras %.1f m from their GPS horizontally, GPS %+.1f m in height, %s" % (
            key, n, role, horizontal, vertical, tilt_text)
        if warn:
            text += " - MORE THAN %.0f DEG: this survey did not fit the others; check its photos (orientation, overlap) before using the result" % SURVEY_TILT_WARN_DEGREES
        lines.append((text, warn))
    return lines


def log_survey_check(opensfm_path, photos, marked):
    """[PATCH14] OpenSfM stage, while reconstruction.json is still topocentric: log the survey check."""
    from opensfm.dataset import DataSet
    from modules import log

    ds = DataSet(opensfm_path)
    reference = ds.load_reference()
    centres, gps = {}, {}
    for rec in ds.load_reconstruction():
        for shot_id, shot in rec.shots.items():
            centres[shot_id] = tuple(shot.pose.get_origin())
    for p in photos:
        if _has_gps(p) and getattr(p, 'altitude', None) is not None:
            gps[p.filename] = tuple(reference.to_topocentric(p.latitude, p.longitude, p.altitude))
    marked_keys = {survey_key(p) for p in photos if p.filename in marked}
    for text, warn in survey_check_lines(survey_differences(photos, centres, gps), marked_keys):
        if warn:
            log.WARNING(text)
        else:
            log.INFO(text)
