# backend/main.py

from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
from sam2.build_sam import build_sam2  # type: ignore
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore

from requests.auth import HTTPBasicAuth
import requests
import uvicorn
import torch
import base64
import io
import numpy as np
import pickle
from typing import List
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
from jinja2 import Template
import logging
import threading
import uuid
import os
import cv2
import zlib
import json
from dotenv import load_dotenv
from pathlib import Path

from sement_utils import SAM_Mask
from sement_utils import load_from_drops, load_locally

# Keep False anyway for now. Not fully implemented (yet?), and not even sure it's worth implementing.
RUNS_ON_N2EB = False
# Right now, it load/uploads everything using requests.get/post() with Drops. Not sure there's any need to change this behavior.

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger('uvicorn')


# For Segment Anything import
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# For SAM 2
load_dotenv()
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH')
MODEL_CONFIG = os.getenv('MODEL_CONFIG')
print('Model config =', MODEL_CONFIG)
print('Checkpoint path =', CHECKPOINT_PATH)

if not os.path.exists(MODEL_CONFIG):
    raise FileNotFoundError(f"Model config file not found: {MODEL_CONFIG}")
elif not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Model checkpoint file not found: {CHECKPOINT_PATH}")
else:
    logger.info(os.path.abspath(CHECKPOINT_PATH))
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


logger.info(f"Loading SAM model using device: {device}...")

sam2_model = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=device)
sam2_model.eval()

logger.info(f"Loaded SAM2 model on {device}.")

app = FastAPI(title="SAM Backend", description="Runs on Mac with MPS.")

# Unused for now, but might be useful in the future.
# This is a lock to prevent multiple threads from accessing the same session data at the same time.
segmentation_lock = threading.Lock()

# We'll keep a global dictionary of session_id -> session_data
# session_data might include: { 'predictor': SamPredictor, 'image_array': np.ndarray, ... }
SESSIONS = {}


##########################
# DATA MODELS
##########################

class CreateSessionRequest(BaseModel):
    image_path: str


class CreateSessionResponse(BaseModel):
    session_id: str
    width: int
    height: int


class InteractiveRequest(BaseModel):
    session_id: str
    # Each point is [x, y], label is 1 (foreground) or 0 (background)
    points: List[List[float]]
    labels: List[int]
    multimask: bool = False


class AutoRequest(BaseModel):
    session_id: str


class ReleaseSessionRequest(BaseModel):
    session_id: str


class MaskResponse(BaseModel):
    masks: List[SAM_Mask]

##################
# Core Endpoints
##################
# 4 endpoints:
# /init_session: load image for req.path, create and store embedding, returns session ID.
# /predict_interactive:Given req.session_id, req.points. req.labels, produce a mask (3 actually) and return it
#  --> how to handle masks? Generate in backend, send to frontend, erase.
# /predict_automatic: same as above w/o points and sends lots of masks
# /release_session


@app.get('/')
def serve_sessions():
    """
    Serves an HTML page displaying the SESSIONS dictionary.
    """
    with open("backend/templates/sessions.html") as f:
        template = Template(f.read())

    html_content = template.render(sessions=SESSIONS)
    return HTMLResponse(content=html_content)


@app.post("/init_session", response_model=CreateSessionResponse)
def init_session(req: CreateSessionRequest):
    """
    Load an image from the server's filesystem, compute the SAM embedding,
    serialize it, and return it to the frontend.
    """
    # Load image
    image_path = req.image_path
    try:
        image_array = load_image(image_path)

    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Could not open image: {e}")
    logger.info('Image loaded. Computing embedding...')
    # Create predictor and compute image embedding
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_array)  # This does embedding internally

    # Generate a session ID for multi users compatibility
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "predictor": predictor,
        "image_array": image_array,
    }

    height, width = image_array.shape[:2]
    return CreateSessionResponse(
        session_id=session_id,
        width=width,
        height=height
    )


@app.post("/predict_interactive", response_model=MaskResponse)
def predict_interactive(req: InteractiveRequest):
    """
    Given a session ID and user-drawn points/labels, produce a mask (or masks).
    """
    # Fetch session_data
    session_data = SESSIONS.get(req.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    predictor = session_data["predictor"]

    # Convert request data
    input_points = np.array(req.points)  # shape [N, 2]
    input_labels = np.array(req.labels)  # shape [N]
    logger.info('Computing a single mask...')

    # Perform prediction
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )
    # If multimask_output=True, you get multiple candidate masks (masks.shape = [3, h, w])
    # If false, you get a single mask (masks.shape = [1, h, w])
    # For ambiguous prompts such as a single point, it is recommended to use multimask_output=True even if only a single mask is desired;
    # the best single mask can be chosen by picking the one with the highest score returned in scores. This will often result in a better mask.
    # We'll return them as a list of flattened binary arrays (int 0/1).
    # For simplicity, we'll round/threshold them to 0/1.

    mask = {}
    logger.info(f'Mask computed, len(masks) = {len(masks)}')
    mask['segmentation'] = masks[0].astype(bool)
    mask['area'] = np.sum(mask['segmentation'])
    mask['stability_score'] = scores
    mask['point_coords'] = input_points.tolist()
    mask['shape'] = mask['segmentation'].shape

    # Below are placeholders (optional)
    mask['bbox'] = [0, 0, 0, 0]
    mask['predicted_iou'] = -1.0
    mask['crop_box'] = [0, 0, 0, 0]

    sam_mask = SAM_Mask(**mask)
    return MaskResponse(masks=[sam_mask])


@app.post("/predict_automatic", response_model=MaskResponse)
def predict_automatic(req: AutoRequest):
    """
    Use SamAutomaticMaskGenerator on the original image (already loaded).
    Return multiple masks.
    """
    session_data = SESSIONS.get(req.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    image_array = session_data["image_array"]
    logger.info('Auto generating all masks...')
    # Create the automatic mask generator each time or store it in session if you prefer
    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        # You can tweak these params
        points_per_batch=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        # etc...
    )

    generated_masks = generator.generate(image_array)

    for m in generated_masks:
        m['shape'] = m['segmentation'].shape
        m['segmentation'] = m["segmentation"].astype(bool)

    logger.info(f'{len(generated_masks)} masks generated.')

    return MaskResponse(masks=generated_masks)


@app.post("/release_session")
def release_session(req: ReleaseSessionRequest):
    """
    Free up memory by removing the stored SamPredictor and image from SESSIONS.
    """
    if req.session_id in SESSIONS:
        del SESSIONS[req.session_id]
        return {"status": "released"}
    else:
        raise HTTPException(status_code=404, detail="Invalid session_id")


def load_image(path: str):
    """
    Loads an image from Drops stored at the given path.
    """
    # if RUNS_ON_N2EB:
    #     if not path.startswith('/drops'):
    #         path = '/drops/' + path
    #     img = load_locally(path)
    img_content = load_from_drops(path)
    img_array = np.frombuffer(img_content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR_RGB)
    return img


if __name__ == "__main__":
    # If you want to run the backend server directly (not via uvicorn CLI):
    uvicorn.run(app, host="0.0.0.0", port=8001)
