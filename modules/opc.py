# [PATCH15] The OpenPointClass model is part of this repository (hikeandmap/models/openpointclass), checked by its SHA-256,
# so that classifying the point cloud never depends on a download. Upstream downloaded it at every first use of a node
# from GitHub; on 2026-11-23 that failed on our server (release-assets.githubusercontent.com: Max retries exceeded) and
# the warning itself then failed ("name 'model_url' is not defined"), so the point cloud was left with the ground
# classes of SMRF only. The download is kept as a fallback.
#
# Model: vehicles-vegetation-buildings.zip of OpenPointClass release v1.1.3 (uav4geo, AGPL-3.0,
# https://github.com/uav4geo/OpenPointClass/releases/tag/v1.1.3); its model.bin unchanged.

import hashlib
import os
from modules.ai import get_model
from modules import log
from modules.system import run
from modules import io

MODEL_URL = "https://github.com/uav4geo/OpenPointClass/releases/download/v1.1.3/vehicles-vegetation-buildings.zip"
MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "hikeandmap", "models", "openpointclass",
                          "vehicles-vegetation-buildings-v1.1.3.bin")
MODEL_SHA256 = "2960ef1036bb66c42ec908fb5160f396062a9baf5a7ca9a08ba4644976d798c2"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path():
    """The model in this repository when it is there and unchanged; else the upstream download (None when that fails too)."""
    path = os.path.abspath(MODEL_FILE)
    if os.path.isfile(path):
        digest = _sha256(path)
        if digest == MODEL_SHA256:
            return path
        log.WARNING("[PATCH15] %s has SHA-256 %s, expected %s: not used" % (path, digest, MODEL_SHA256))
    else:
        log.WARNING("[PATCH15] %s is missing" % path)
    log.INFO("[PATCH15] Downloading the OpenPointClass model instead")
    return get_model("openpointclass", MODEL_URL, "v1.0.0", name="model.bin")


def classify(point_cloud, max_threads=8):
    """Classifies the point cloud in place. True when it was classified."""
    tmp_output = io.related_file_path(point_cloud, postfix=".classified")
    if os.path.isfile(tmp_output):
        os.remove(tmp_output)

    try:
        model = model_path()
        if model is None:
            log.WARNING("Cannot classify using OpenPointClass: the model could not be loaded from %s nor downloaded from %s"
                        % (os.path.abspath(MODEL_FILE), MODEL_URL))
            return False

        log.INFO("[PATCH15] Classifying with the OpenPointClass model %s" % model)
        run('pcclassify "%s" "%s" "%s" -u -s 2,64' % (point_cloud, tmp_output, model), env_vars={'OMP_NUM_THREADS': max_threads})

        if os.path.isfile(tmp_output):
            os.remove(point_cloud)
            os.rename(tmp_output, point_cloud)
            return True
        log.WARNING("Cannot classify using OpenPointClass (no output generated)")
        return False

    except Exception as e:
        log.WARNING("Cannot classify using OpenPointClass: %s" % str(e))
        return False
