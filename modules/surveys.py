# [PATCH10] Several surveys of the same area: the orthophoto shows the newest one.
#
# An area may have been flown more than once (2023, 2025, 2026). All photos go into the reconstruction - the older ones still
# hold features of what has not changed, and their GCPs place the newer survey - but the texturing used to pick, face by face,
# whichever photo looked most straight down, so a building came out half from one year and half from another.
#
# Photos are grouped into surveys by the month they were taken (EXIF DateTimeOriginal; a survey of one area never runs
# twice in a month). The newest survey is rank 0. Every older photo gets a role, worked out from the reconstruction:
# - edge: part of its footprint lies outside the ground a newer survey sees within MAX_ANGLE of straight down. It textures
#   like any photo there (near the edge of a newer survey the older nadir photos take over), and joins the point cloud.
# - reserve: its whole footprint lies inside that ground. It still helps to place the survey (feature matching, its GCPs),
#   and it paints only its GCP squares (PATCH4) and the ground under a vehicle that every newer photo has (PATCH7).
# texrecon reads the surveys and roles (--view_surveys) and paints each face from the newest survey that sees it within
# MAX_ANGLE. After texturing, the faces' photos give the source layer: which survey, day and flight painted which ground,
# as a GeoPackage of polygons next to the report (odm_report/orthophoto_sources.gpkg) - kept for every run, one survey or more.

import os
import math
import struct
from datetime import datetime, timezone

import numpy as np

SOURCES_FILE = "orthophoto_sources.gpkg"
SOURCES_CELL = 0.25          # metres per cell of the source raster
FLIGHT_GAP = 5 * 60          # seconds without a photo that start a new flight
FOOTPRINT_SAMPLES = 5        # footprint points per image side checked for the reserve role
GROUND_NEIGHBOURS = 64       # mesh vertices around a camera whose median height is the ground under it
LABELING_MAGIC = b"HAMLAB01"


def capture_time(photo):
    """The photo's capture time as the camera clock gave it (ODX reads DateTimeOriginal as if it were UTC), or None."""
    if getattr(photo, 'utc_time', None) is None:
        return None
    return datetime.fromtimestamp(photo.utc_time / 1000.0, timezone.utc)


def survey_label(t):
    return t.strftime("%Y-%m") if t is not None else "undated"


def group_surveys(photos):
    """
    :return (labels, survey_of): the survey labels, newest first (photos without a date last), and filename -> rank
    """
    months = sorted({survey_label(capture_time(p)) for p in photos if capture_time(p) is not None}, reverse=True)
    labels = months + (["undated"] if any(capture_time(p) is None for p in photos) else [])
    rank = {l: i for i, l in enumerate(labels)}
    return labels, {p.filename: rank[survey_label(capture_time(p))] for p in photos}


def flights_of(photos):
    """
    Photos taken without a gap of FLIGHT_GAP seconds are one flight.
    :return list of dicts (survey, date, start, end, photos) sorted by time, and filename -> index into that list
    """
    timed = sorted(((capture_time(p), p.filename) for p in photos if capture_time(p) is not None), key=lambda t: t[0])
    flights, index_of = [], {}
    for t, name in timed:
        if not flights or (t - flights[-1]['end']).total_seconds() > FLIGHT_GAP or t.date() != flights[-1]['end'].date():
            flights.append({'survey': survey_label(t), 'date': t.strftime("%Y-%m-%d"), 'start': t, 'end': t, 'photos': 0})
        flights[-1]['end'] = t
        flights[-1]['photos'] += 1
        index_of[name] = len(flights) - 1
    return flights, index_of


def view_photo_names(views, photos):
    """NVM view name -> photo filename (the undistorted image keeps the photo's name plus an extension)."""
    names = {p.filename for p in photos}
    result = {}
    for v in views:
        n = v['name']
        if n in names:
            result[n] = n
            continue
        for cut in range(len(n) - 1, 0, -1):
            if n[cut] == '.' and n[:cut] in names:
                result[n] = n[:cut]
                break
    return result


def write_view_surveys(output_file, reconstruction, nvm_file, mesh_file, max_angle, log):
    """
    Writes texrecon's --view_surveys file when the photos hold more than one survey.
    :return the number of surveys (the file is only written, and only needed, with 2 or more)
    """
    from modules import wire_masks
    from scipy.spatial import cKDTree

    labels, survey_of = group_surveys(reconstruction.photos)
    if len(labels) < 2:
        log.INFO("[PATCH10] One survey (%s): textured as usual" % (labels[0] if labels else "no photos"))
        return len(labels)

    views = wire_masks.read_nvm(nvm_file)
    photo_of = view_photo_names(views, reconstruction.photos)
    ranks = np.array([survey_of.get(photo_of.get(v['name']), 0) for v in views])
    centres = np.array([v['C'] for v in views])

    vertices = wire_masks.read_mesh_vertices(mesh_file)
    if len(vertices) < GROUND_NEIGHBOURS:
        raise Exception("the mesh %s has too few vertices" % mesh_file)
    _, near = cKDTree(vertices[:, :2]).query(centres[:, :2], k=GROUND_NEIGHBOURS)
    ground = np.median(vertices[near, 2], axis=1)
    reach = np.maximum(0.0, centres[:, 2] - ground) * math.tan(math.radians(max_angle))

    # The ground each survey sees within max_angle: a disc of radius reach around every camera (flat ground).
    newer_trees = {}
    def covered_by_newer(points, rank):
        if rank not in newer_trees:
            newer = np.nonzero(ranks < rank)[0]
            newer_trees[rank] = (newer, cKDTree(centres[newer, :2]) if len(newer) else None)
        newer, t = newer_trees[rank]
        if t is None:
            return False
        r_max = float(reach[newer].max())
        for p in points:
            hits = t.query_ball_point(p, r_max)
            if not any(np.hypot(*(p - centres[newer[h], :2])) <= reach[newer[h]] for h in hits):
                return False
        return True

    reserve = np.zeros(len(views), dtype=bool)
    for i, v in enumerate(views):
        if ranks[i] == 0:
            continue
        w, h = wire_masks.image_size(v['path'])
        us = np.linspace(0, w - 1, FOOTPRINT_SAMPLES)
        vs = np.linspace(0, h - 1, FOOTPRINT_SAMPLES)
        points = []
        for u in us:
            for q in vs:
                d = v['R'].T.dot(np.array([(u - w / 2.0 + 0.5) / v['f'], (q - h / 2.0 + 0.5) / v['f'], 1.0]))
                if d[2] >= -1e-6:
                    points = None  # looks up to or above the horizon: its footprint has no end
                    break
                t = (ground[i] - v['C'][2]) / d[2]
                points.append(v['C'][:2] + t * d[:2])
            if points is None:
                break
        reserve[i] = points is not None and covered_by_newer(points, ranks[i])

    lines = ["# [PATCH10] view surveys v1, max angle %.1f deg: image_name rank reserve label (rank 0 = the newest survey; "
             "reserve 1 = an older photo inside a newer survey)" % max_angle]
    for i, v in enumerate(views):
        lines.append("%s %d %d %s" % (v['name'], ranks[i], 1 if reserve[i] else 0, labels[ranks[i]]))
    with open(output_file, 'w') as f:
        f.write("\n".join(lines) + "\n")

    parts = []
    for r, label in enumerate(labels):
        n = int((ranks == r).sum())
        if r == 0:
            parts.append("%s (newest) %d photos" % (label, n))
        else:
            n_reserve = int((reserve & (ranks == r)).sum())
            parts.append("%s %d photos (%d at the edge, %d reserve)" % (label, n, n - n_reserve, n_reserve))
    log.INFO("[PATCH10] %d surveys: %s; reserve = an older photo inside a newer survey, used to place the survey, "
             "for its GCP squares and under vehicles only" % (len(labels), "; ".join(parts)))
    return len(labels)


def survey_ranks(reconstruction):
    """filename -> survey rank (0 = the newest), and the labels, for the other patches (GCP squares, wires)."""
    labels, survey_of = group_surveys(reconstruction.photos)
    return survey_of, labels


def read_labeling_faces(prefix):
    """texrecon --write_labeling output: (triangles (n, 3, 2) float32 mesh x, y; labels (n,) uint32: 0 none, k = view k - 1; view images)."""
    with open(prefix + "_labeling_faces.bin", "rb") as f:
        if f.read(8) != LABELING_MAGIC:
            raise Exception("%s_labeling_faces.bin is not a labeling faces file" % prefix)
        n = struct.unpack("<q", f.read(8))[0]
        data = np.fromfile(f, dtype=np.dtype([('xy', '<f4', (6,)), ('label', '<u4')]), count=n)
    with open(prefix + "_labeling_views.txt") as f:
        images = [l.rstrip("\n") for l in f if l.strip()]
    return data['xy'].reshape(-1, 3, 2), data['label'], images


def write_sources(output_file, prefix, reconstruction, log):
    """
    The source layer: which survey, day and flight painted which ground of the orthophoto, as polygons in the
    orthophoto's coordinate system. :return the number of polygons written
    """
    import cv2
    from osgeo import gdal, ogr, osr

    triangles, labels, images = read_labeling_faces(prefix)
    flights, flight_of = flights_of(reconstruction.photos)
    survey_names, _ = group_surveys(reconstruction.photos)
    names = {p.filename for p in reconstruction.photos}

    def photo_name(path):
        n = os.path.basename(path)
        if n in names:
            return n
        for cut in range(len(n) - 1, 0, -1):
            if n[cut] == '.' and n[:cut] in names:
                return n[:cut]
        return None

    # value per view: its flight + 1 (0 = none)
    view_value = np.zeros(len(images) + 1, dtype=np.int32)
    for k, path in enumerate(images):
        f = flight_of.get(photo_name(path))
        view_value[k + 1] = f + 1 if f is not None else 0
    values = view_value[np.minimum(labels, len(images)).astype(np.int64)]

    keep = values > 0
    if not keep.any():
        log.WARNING("[PATCH10] No face of the orthophoto has a dated photo: no source layer written")
        return 0
    tri = triangles[keep].astype(np.float64)
    values = values[keep]

    x0, y0 = tri[..., 0].min(), tri[..., 1].min()
    x1, y1 = tri[..., 0].max(), tri[..., 1].max()
    nx = int(math.ceil((x1 - x0) / SOURCES_CELL)) + 1
    ny = int(math.ceil((y1 - y0) / SOURCES_CELL)) + 1
    raster = np.zeros((ny, nx), dtype=np.uint16)
    shift = 4  # sub-cell precision of cv2.fillPoly
    px = np.round((tri[..., 0] - x0) / SOURCES_CELL * (1 << shift)).astype(np.int32)
    py = np.round((y1 - tri[..., 1]) / SOURCES_CELL * (1 << shift)).astype(np.int32)
    pts = np.stack([px, py], -1)
    # Painted flight by flight; where faces overlap (walls of the 2.5D mesh) the last one drawn stays.
    for value in np.unique(values):
        sel = pts[values == value]
        cv2.fillPoly(raster, [p.reshape(3, 1, 2) for p in sel], int(value), lineType=cv2.LINE_8, shift=shift)

    off_x, off_y = reconstruction.georef.utm_east_offset, reconstruction.georef.utm_north_offset
    srs = osr.SpatialReference()
    srs.ImportFromWkt(reconstruction.georef.srs.to_wkt())

    mem = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_UInt16)
    mem.SetGeoTransform((x0 + off_x - SOURCES_CELL / 2.0, SOURCES_CELL, 0.0, y1 + off_y + SOURCES_CELL / 2.0, 0.0, -SOURCES_CELL))
    mem.SetProjection(srs.ExportToWkt())
    band = mem.GetRasterBand(1)
    band.WriteArray(raster)
    band.SetNoDataValue(0)

    if os.path.exists(output_file):
        os.remove(output_file)
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(output_file)
    layer = ds.CreateLayer("sources", srs, ogr.wkbPolygon)
    for name, kind in (("value", ogr.OFTInteger), ("survey", ogr.OFTString), ("newest", ogr.OFTInteger),
                       ("date", ogr.OFTString), ("flight_start", ogr.OFTString), ("flight_end", ogr.OFTString),
                       ("flight_photos", ogr.OFTInteger), ("area_m2", ogr.OFTReal)):
        layer.CreateField(ogr.FieldDefn(name, kind))
    gdal.Polygonize(band, band, layer, 0, [], callback=None)

    newest = survey_names[0] if survey_names else ""
    count = 0
    area_by_survey = {}
    layer.ResetReading()
    fids = [feature.GetFID() for feature in layer]
    ds.StartTransaction()
    for fid in fids:
        feature = layer.GetFeature(fid)
        v = feature.GetField("value")
        if not v:
            layer.DeleteFeature(fid)
            continue
        fl = flights[v - 1]
        area = feature.GetGeometryRef().GetArea()
        feature.SetField("survey", fl['survey'])
        feature.SetField("newest", 1 if fl['survey'] == newest else 0)
        feature.SetField("date", fl['date'])
        feature.SetField("flight_start", fl['start'].strftime("%H:%M:%S"))
        feature.SetField("flight_end", fl['end'].strftime("%H:%M:%S"))
        feature.SetField("flight_photos", fl['photos'])
        feature.SetField("area_m2", area)
        layer.SetFeature(feature)
        area_by_survey[fl['survey']] = area_by_survey.get(fl['survey'], 0.0) + area
        count += 1
    ds.CommitTransaction()
    ds = None

    total = sum(area_by_survey.values()) or 1.0
    log.INFO("[PATCH10] Source layer: %d areas, %s, written to %s" % (count, ", ".join(
        "%s %.1f %%" % (s, 100.0 * a / total) for s, a in sorted(area_by_survey.items(), reverse=True)), output_file))
    return count
