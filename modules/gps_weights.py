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
