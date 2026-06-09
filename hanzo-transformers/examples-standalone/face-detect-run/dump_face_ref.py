#!/usr/bin/env python3
"""Dump face_alignment reference: SFD raw box, 68 landmarks, and the MuseTalk
nose-centered bbox, per frame of sun.mp4. Mirrors musetalk_dub.py exactly."""
import os, sys, glob, json, subprocess
import numpy as np, cv2, torch
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
_orig = torch.load
torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _orig(*a, **k))[1]
import face_alignment

REPO = os.path.expanduser("~/work/zen-dub-run/zen-dub")
VIDEO = os.path.join(REPO, "data/video/sun.mp4")
FFMPEG = os.path.expanduser("~/work/zen-dub-run/bin/ffmpeg")
OUT = os.path.expanduser("~/work/zen-dub-run/facedump")
FRAMES = os.path.join(OUT, "frames")
os.makedirs(FRAMES, exist_ok=True)
for f in glob.glob(os.path.join(FRAMES, "*.png")):
    os.remove(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, flush=True)

# extract frames identically to the dub
subprocess.run([FFMPEG, "-v", "fatal", "-y", "-i", VIDEO,
                "-start_number", "0", f"{FRAMES}/%08d.png"], check=True)
img_list = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
print("n frames", len(img_list), flush=True)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                  flip_input=False, device=device)

# dump EVERY frame; cheap enough and lets us check temporal drift fully
recs = []
last_good = None
for idx, p in enumerate(img_list):
    fr = cv2.imread(p)
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    preds, _, det = fa.get_landmarks(rgb, return_bboxes=True)
    if not preds:
        recs.append({"frame": idx, "ok": False,
                     "musetalk_bbox": list(last_good) if last_good else [0, 0, 0, 0]})
        continue
    lm = preds[0].astype(np.int32)  # 68x2
    half_y = int(lm[29][1])
    chin_y = int(np.max(lm[:, 1]))
    half_dist = chin_y - half_y
    upper = max(0, half_y - half_dist)
    x1 = int(np.min(lm[:, 0])); x2 = int(np.max(lm[:, 0]))
    y1 = int(upper); y2 = int(chin_y)
    if x2 - x1 <= 0 or y2 - y1 <= 0 or x1 < 0:
        bbox = list(last_good) if last_good else [0, 0, 0, 0]
    else:
        last_good = (x1, y1, x2, y2)
        bbox = [x1, y1, x2, y2]
    sfd = det[0].tolist() if len(det) else None  # [x1,y1,x2,y2,score]
    recs.append({"frame": idx, "ok": True,
                 "sfd_box": sfd,
                 "nose": [int(lm[29][0]), int(lm[29][1])],
                 "lm_xmin": x1, "lm_xmax": x2, "chin_y": chin_y,
                 "musetalk_bbox": bbox,
                 "landmarks": lm.tolist()})
    if idx < 3 or idx % 100 == 0:
        print(f"frame {idx}: sfd={sfd} mtbbox={bbox}", flush=True)

with open(os.path.join(OUT, "ref_face.json"), "w") as f:
    json.dump({"video": VIDEO, "w": int(fr.shape[1]), "h": int(fr.shape[0]),
               "n": len(img_list), "records": recs}, f)
print("WROTE", os.path.join(OUT, "ref_face.json"), "records", len(recs), flush=True)
