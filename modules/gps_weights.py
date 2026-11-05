# [PATCH6] Photos with and without RTK in one run, with or without GCPs.
#
# 1. A photo that carries its own position accuracy (DJI RTK XMP, eBee XMP, or a geo file with accuracy columns) keeps it:
#    --gps-accuracy (and the new --gps-accuracy-vertical) apply only to the photos without one. Stock ODX overwrote the
#    accuracy of every photo with --gps-accuracy, RTK photos included.
# 2. Photos without their own accuracy get --gps-accuracy horizontally and --gps-accuracy-vertical vertically (0 = the same
#    as --gps-accuracy, as before). A consumer drone's absolute height is often tens of metres off, far more than its
#    horizontal position.
# 3. With a GCP file, stock ODX switches the photo positions off altogether (bundle_use_gps: no). When the run has RTK
#    photos (at least MIN_RTK_PHOTOS with an accuracy of RTK_MAX_METRES or better), their positions now stay in the bundle
#    adjustment next to the GCPs, and the photos without RTK are given an accuracy of NO_PULL_METRES, so that they do not
#    pull on positions at all (as before) - their GPS frame may differ from the GCPs' by metres.

import statistics

MIN_RTK_PHOTOS = 3
RTK_MAX_METRES = 1.0          # after ODX's doubling of the reported RTK standard deviation
NO_PULL_METRES = 10000.0

SOURCE_PHOTO = 'photo'        # the accuracy came with the photo (XMP or geo file)
SOURCE_OPTION = 'option'      # the accuracy was set from --gps-accuracy / --gps-accuracy-vertical


def _has_gps(p):
    return getattr(p, 'latitude', None) is not None and getattr(p, 'longitude', None) is not None


def has_own_accuracy(p):
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
            p.gps_accuracy_source = SOURCE_PHOTO if getattr(p, 'gps_xy_stddev', None) is not None else None


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
    parts = []
    if own:
        xy = statistics.median(p.gps_xy_stddev for p in own)
        z = statistics.median((p.gps_z_stddev if p.gps_z_stddev is not None else p.gps_xy_stddev) for p in own)
        rtk = sum(1 for p in own if is_rtk(p))
        parts.append("%s with their own accuracy, kept (median %.3f m horizontal, %.3f m vertical; %s of them RTK)" % (len(own), xy, z, rtk))
    if others:
        parts.append("%s without, %.1f m horizontal and %.1f m vertical" % (
            len(others), gps_accuracy, vertical_accuracy(gps_accuracy, gps_accuracy_vertical)))
    if not parts:
        return "[PATCH6] Photo positions: no photo has a GPS position"
    return "[PATCH6] Photo positions: " + "; ".join(parts)


def rtk_count(photos):
    return sum(1 for p in photos if _has_gps(p) and is_rtk(p))


def keep_gps_with_gcp(photos):
    """With a GCP file: True when the run has enough RTK photos for their positions to stay in the adjustment."""
    return rtk_count(photos) >= MIN_RTK_PHOTOS


def silence_without_rtk(photos):
    """With a GCP file and RTK photos: the photos without RTK do not pull on positions. In memory only (images.json keeps
    their real values). Returns how many."""
    n = 0
    for p in photos:
        if _has_gps(p) and not is_rtk(p):
            p.gps_xy_stddev = p.gps_z_stddev = NO_PULL_METRES
            n += 1
    return n
