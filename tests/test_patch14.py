import math
import unittest
from datetime import datetime, timezone

from modules.photo import normalize_dji_gimbal, normalize_degrees
from modules import gps_weights


def rotation(yaw, pitch, roll):
    """ZYX Euler angles in degrees as a 3x3 rotation (lists)."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    rz = [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
    ry = [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
    rx = [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
    mul = lambda a, b: [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mul(mul(rz, ry), rx)


class FakePhoto:
    def __init__(self, name, month, model, lat, lon, alt=1500.0, make='DJI'):
        self.filename = name
        self.camera_make = make
        self.camera_model = model
        self.latitude = lat
        self.longitude = lon
        self.altitude = alt
        self.utc_time = datetime(2026 if month >= 11 else 2025, month, 5, tzinfo=timezone.utc).timestamp() * 1000.0
        self.gps_xy_stddev = 3.0
        self.gps_z_stddev = 3.0
        self.rtk_flag = None
        self.gps_accuracy_source = gps_weights.SOURCE_OPTION


class FakeGcp:
    def __init__(self, names):
        self.names = names

    def exists(self):
        return True

    def iter_entries(self):
        for n in self.names:
            yield type('E', (), {'filename': n})()


class TestPatch14Gimbal(unittest.TestCase):
    def test_matrice_4e_straight_down_is_the_same_camera_with_roll_0(self):
        # Burnham Park 2026-11-05: GimbalYaw +147.3, GimbalPitch -90, GimbalRoll +180 (aircraft yaw -33.1)
        pitch, roll, flipped = normalize_dji_gimbal(-90.0, 180.0)
        yaw = normalize_degrees(147.3 + 180.0)
        self.assertTrue(flipped)
        self.assertAlmostEqual(pitch, -90.0)
        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(yaw, -32.7, places=6)
        a, b = rotation(147.3, -90.0, 180.0), rotation(yaw, pitch, roll)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(a[i][j], b[i][j], places=9)

    def test_oblique_and_negative_roll_are_the_same_camera(self):
        for gy, p, r in [(147.3, -60.0, 180.0), (30.0, -45.0, -180.0), (-100.0, -20.0, 179.0)]:
            p2, r2, flipped = normalize_dji_gimbal(p, r)
            self.assertTrue(flipped)
            self.assertLessEqual(abs(r2), 90.0)
            a, b = rotation(gy, p, r), rotation(normalize_degrees(gy + 180.0), p2, r2)
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(a[i][j], b[i][j], places=9)

    def test_mavic_and_mini_are_not_touched(self):
        self.assertEqual(normalize_dji_gimbal(-90.0, 0.0), (-90.0, 0.0, False))
        self.assertEqual(normalize_dji_gimbal(-89.9, 0.5), (-89.9, 0.5, False))
        self.assertEqual(normalize_dji_gimbal(None, 180.0), (None, 180.0, False))


class TestPatch14Surveys(unittest.TestCase):
    def photos(self):
        older = [FakePhoto("mavic_%02d.JPG" % i, 5, "L1D-20c", 16.41 + i * 1e-4, 120.59, make='Hasselblad') for i in range(20)]
        newer = [FakePhoto("m4e_%02d.JPG" % i, 11, "M4E", 16.41 + i * 1e-4, 120.59) for i in range(20)]
        return older, newer

    def test_a_survey_without_marks_keeps_its_gps_and_the_marked_one_is_silenced(self):
        older, newer = self.photos()
        marked = gps_weights.marked_filenames(FakeGcp(["mavic_03.JPG", "mavic_07.JPG"]))
        photos = older + newer
        self.assertEqual(gps_weights.surveys_without_marks(photos, marked), ["2026-11 DJI M4E"])
        self.assertTrue(gps_weights.keep_unmarked_surveys(photos, marked))
        self.assertEqual(gps_weights.silence_marked_surveys(photos, marked), 20)
        self.assertTrue(all(p.gps_xy_stddev == gps_weights.NO_PULL_METRES for p in older))
        self.assertTrue(all(p.gps_xy_stddev == 3.0 for p in newer))

    def test_every_survey_marked_changes_nothing(self):
        older, newer = self.photos()
        marked = {"mavic_03.JPG", "m4e_05.JPG"}
        self.assertFalse(gps_weights.keep_unmarked_surveys(older + newer, marked))

    def test_a_folded_survey_is_reported_as_tilted(self):
        older, newer = self.photos()
        photos = older + newer
        gps, centres = {}, {}
        tilt = math.radians(34.0)
        for i, p in enumerate(photos):
            x, y = (i % 5) * 40.0, (i // 5 % 4) * 40.0
            gps[p.filename] = (x, y, 50.0)
            if p in newer:
                # folded about the line y = 0, and 47 m higher in its GPS (another height system)
                centres[p.filename] = (x, y * math.cos(tilt), 50.0 - 47.0 + y * math.sin(tilt))
            else:
                centres[p.filename] = (x, y, 50.0)
        diffs = {d[0]: d for d in gps_weights.survey_differences(photos, centres, gps)}
        self.assertAlmostEqual(diffs["2026-11 DJI M4E"][4], 34.0, places=3)
        self.assertAlmostEqual(diffs["2025-05 Hasselblad L1D-20c"][4], 0.0, places=6)
        lines = gps_weights.survey_check_lines(list(diffs.values()), {"2025-05 Hasselblad L1D-20c"})
        warned = [text for text, warn in lines if warn]
        self.assertEqual(len(warned), 1)
        self.assertIn("2026-11 DJI M4E", warned[0])


if __name__ == '__main__':
    unittest.main()
