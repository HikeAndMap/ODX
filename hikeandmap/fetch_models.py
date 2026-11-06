# [PATCH7] Puts the YOLOX models of modules/vehicle_boxes.py into the node image when it is built, each checked by its
# SHA-256, so that a run never depends on downloading them. Run from the Dockerfile: python3 hikeandmap/fetch_models.py
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from modules import vehicle_boxes  # noqa: E402

target = os.path.join(ROOT, "storage", "models", vehicle_boxes.MODEL_NAMESPACE, vehicle_boxes.MODEL_VERSION.replace(".", "_"))
os.makedirs(target, exist_ok=True)

for name, url, sha256 in vehicle_boxes.MODELS:
    path = os.path.join(target, name)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == sha256:
                print("[PATCH7] %s already in the image" % name)
                continue
    print("[PATCH7] Downloading %s" % url)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    with open(tmp, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != sha256:
        os.remove(tmp)
        sys.exit("[PATCH7] %s has SHA-256 %s, expected %s: not used" % (name, digest, sha256))
    os.replace(tmp, path)
    print("[PATCH7] %s checked (%s bytes)" % (name, os.path.getsize(path)))
