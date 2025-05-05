#!/usr/bin/env python3
"""
Expression‑controlled OSC + Syphon streamer with normalized eye/mouth coords
──────────────────────────────────────────────────────────────────────────
• webcam → MediaPipe Face Landmarker (blend‑shapes + face center + pose)
• exponential‑moving‑average smoothing
• OSC out, one message per label:
    /face/browInnerUp         <float>
    … (all blend‑shape TARGETS) …
    /face/center/x            <float>
    /face/center/y            <float>
    /face/center/z            <float>
    /face/pose/yaw            <float>
    /face/pose/pitch          <float>
    /face/pose/roll           <float>
    /face/eye/left/x          <float>  (0…1)
    /face/eye/left/y          <float>  (0…1)
    /face/eye/right/x         <float>  (0…1)
    /face/eye/right/y         <float>  (0…1)
    /face/mouth/x             <float>  (0…1)
    /face/mouth/y             <float>  (0…1)
• Syphon: live RGBA camera feed published as “FaceOSCPy”
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc import udp_client

# ─── Syphon (macOS) ───────────────────────────────────────────────────
import syphon
from syphon.utils.raw   import create_mtl_texture
from syphon.utils.numpy import copy_image_to_mtl_texture

# ────────────────────────── CONFIG ──────────────────────────
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-assets/"
    "face_landmarker_v2_with_blendshapes.task"
)
MODEL_FILE = "face_landmarker_v2_with_blendshapes.task"

TARGETS = [
    "browInnerUp", "mouthFunnel",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthLowerDownLeft1", "mouthLowerDownRight1",
    "mouthSmileRight", "mouthSmileLeft",
    "eyeSquintLeft", "eyeSquintRight",
    "jawOpen",
    "eyeBlinkLeft", "eyeBlinkRight",
    "mouthFrownLeft", "mouthFrownRight",
    "eyeLookInRight", "eyeLookInLeft",
    "eyeLookDownLeft", "eyeLookUpLeft",
]

ALPHA_BLEND   = 0.6
ALPHA_CENTER  = 0.6
ALPHA_POSE    = 0.6
ALPHA_TRACK   = 0.6  # smoothing for normalized coords

OSC_IP        = "127.0.0.1"
OSC_PORT      = 8001
WEBCAM_INDEX  = 0
SYPHON_NAME   = "FaceOSCPy"

# landmark indices for eye corners and mouth corners
LE_OUTER, LE_INNER = 33, 133
RE_INNER, RE_OUTER = 362, 263
MO_LEFT, MO_RIGHT = 61, 291
# ─────────────────────────────────────────────────────────────

# download model if missing
if not os.path.exists(MODEL_FILE):
    print("▶ Downloading Face Landmarker model …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

# init Face Landmarker
BaseOptions = python.BaseOptions
FLOpts      = vision.FaceLandmarkerOptions
RunMode     = vision.RunningMode
landmarker  = vision.FaceLandmarker.create_from_options(
    FLOpts(
        base_options=BaseOptions(model_asset_path=MODEL_FILE),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=RunMode.VIDEO,
    )
)

# OSC client
osc = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# smoothing buffers
smooth_blend   = {name: 0.0 for name in TARGETS}
smooth_center  = np.zeros(3, dtype=np.float32)
smooth_pose    = np.zeros(3, dtype=np.float32)
smooth_leye    = np.zeros(2, dtype=np.float32)
smooth_reye    = np.zeros(2, dtype=np.float32)
smooth_mouth   = np.zeros(2, dtype=np.float32)

def ema(prev, new, alpha):
    return alpha * prev + (1 - alpha) * new

def matrix_to_euler(R: np.ndarray):
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    yaw   =  np.arctan2(R[1, 0], R[0, 0])
    roll  =  np.arctan2(R[2, 1], R[2, 2])
    return np.degrees([yaw, pitch, roll])

# open webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
timestamp_ms = 0

# setup Syphon
ret, first = cap.read()
if not ret:
    raise RuntimeError("Unable to read from webcam")
h, w = first.shape[:2]
sy_server = syphon.SyphonMetalServer(SYPHON_NAME)
texture   = create_mtl_texture(sy_server.device, w, h)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # publish Syphon frame
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        copy_image_to_mtl_texture(rgba, texture)
        sy_server.publish_frame_texture(texture)

        # MediaPipe prep
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # inference
        result = landmarker.detect_for_video(mp_img, int(timestamp_ms))
        timestamp_ms += 1000 / fps

        if not result.face_blendshapes or not result.face_landmarks:
            cv2.imshow("cam", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # blend‑shapes
        raw_bs = {b.category_name: b.score for b in result.face_blendshapes[0]}
        for name in TARGETS:
            smooth_blend[name] = ema(smooth_blend[name], raw_bs.get(name,0.0), ALPHA_BLEND)
            osc.send_message(f"/face/{name}", float(smooth_blend[name]))

        # face‑center
        pts = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float32)
        raw_center = pts.mean(axis=0)
        smooth_center = ema(smooth_center, raw_center, ALPHA_CENTER)
        osc.send_message("/face/center/x", float(smooth_center[0]))
        osc.send_message("/face/center/y", float(smooth_center[1]))
        osc.send_message("/face/center/z", float(smooth_center[2]))

        # head‑pose
        mat = np.array(result.facial_transformation_matrixes[0]).reshape(4,4)[:3,:3]
        yaw, pitch, roll = matrix_to_euler(mat)
        raw_pose = np.array([yaw,pitch,roll],dtype=np.float32)
        smooth_pose = ema(smooth_pose, raw_pose, ALPHA_POSE)
        osc.send_message("/face/pose/yaw",   float(smooth_pose[0]))
        osc.send_message("/face/pose/pitch", float(smooth_pose[1]))
        osc.send_message("/face/pose/roll",  float(smooth_pose[2]))

        # left‑eye normalized center
        lm = result.face_landmarks[0]
        le = np.array([
            (lm[LE_OUTER].x + lm[LE_INNER].x) / 2,
            (lm[LE_OUTER].y + lm[LE_INNER].y) / 2
        ], dtype=np.float32)
        smooth_leye = ema(smooth_leye, le, ALPHA_TRACK)
        osc.send_message("/face/eye/left/x", float(smooth_leye[0]))
        osc.send_message("/face/eye/left/y", float(smooth_leye[1]))

        # right‑eye normalized center
        re = np.array([
            (lm[RE_OUTER].x + lm[RE_INNER].x) / 2,
            (lm[RE_OUTER].y + lm[RE_INNER].y) / 2
        ], dtype=np.float32)
        smooth_reye = ema(smooth_reye, re, ALPHA_TRACK)
        osc.send_message("/face/eye/right/x", float(smooth_reye[0]))
        osc.send_message("/face/eye/right/y", float(smooth_reye[1]))

        # mouth normalized center
        mo = np.array([
            (lm[MO_LEFT].x + lm[MO_RIGHT].x) / 2,
            (lm[MO_LEFT].y + lm[MO_RIGHT].y) / 2
        ], dtype=np.float32)
        smooth_mouth = ema(smooth_mouth, mo, ALPHA_TRACK)
        osc.send_message("/face/mouth/x", float(smooth_mouth[0]))
        osc.send_message("/face/mouth/y", float(smooth_mouth[1]))

        # preview
        cv2.imshow("cam", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    sy_server.stop()
    landmarker.close()
