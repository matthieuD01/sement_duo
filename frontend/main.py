import uvicorn
import os
import pickle
import requests
from typing import List
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import logging
from requests.auth import HTTPBasicAuth
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import time
from utils.drops_utils import *
from utils.masks import SAM_Mask
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger('uvicorn')

# --------------------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# --------------------------------------------------------------------------------------

load_dotenv()

# Access the variables
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')
DROPS_URL = os.getenv('DROPS_URL')

# Where all masks are stored as fname.pkl
MASKS_ROOT = os.getenv('MASKS_ROOT')
# The URL to the backend server (with port number)
BACKEND_URL = os.getenv('BACKEND_URL')


# --------------------------------------------------------------------------------------
# FASTAPI APP
# --------------------------------------------------------------------------------------

app = FastAPI(title="SAM Frontend")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --------------------------------------------------------------------------------------
# DATA MODELS
# --------------------------------------------------------------------------------------


class InitSessionRequest(BaseModel):
    image_path: str


class InitSessionResponse(BaseModel):
    session_id: str
    width: int
    height: int


class ReleaseSessionRequest(BaseModel):
    session_id: str


class InteractiveRequest(BaseModel):
    session_id: str
    # Each point is [x, y], label is 1 (foreground) or 0 (background)
    points: List[List[float]]
    labels: List[int]
    multimask: bool = False


class AutoRequest(BaseModel):
    session_id: str


class SaveMasksRequest(BaseModel):
    image_name: str
    image_path: str
    masks: List[SAM_Mask]


class LoadMasksRequest(BaseModel):
    image_path: str


class MaskResponse(BaseModel):
    # Return a single mask
    masks: List[SAM_Mask]  # each mask is 2D

# --------------------------------------------------------------------------------------
# FRONTEND ENDPOINTS
# --------------------------------------------------------------------------------------

# Homepage


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main page. We pass the list of images to the template.
    """
    return templates.TemplateResponse("index2.html", {"request": request})

# Drops page duplicate, for use as file explorer in the homepage.


@app.get("/drops/{path:path}", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render a clone of Drops page. We pass the list of images to the template.
    """
    return templates.TemplateResponse("drops_page.html", {"request": request})


SESSIONS = {}
# Initiates a new segmentation session with a given image path with the backend.


@app.post("/init_session_front")
def init_session(req: InitSessionRequest):
    """
    Calls the backend to create a new segmentation session with a given image path.
    Returns { session_id, width, height }.
    """

    # Call backend /init_session
    payload = {"image_path": req.image_path}
    logger.info(req.image_path)
    resp = requests.post(f"{BACKEND_URL}/init_session",
                         json=payload, timeout=300)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    rs = resp.json()
    session_id = rs['session_id']
    SESSIONS[session_id] = {'last_updated': time.time(), 'masks': [], 'image_path': req.image_path,
                            'selected': [], }
    logger.info(f'New session with ID {session_id}')
    return resp.json()  # { session_id, width, height }
    # return {'session_id': '123456'}


@app.post("/predict_interactive")
def predict_interactive(req: InteractiveRequest):
    """
    Calls the backend /predict_interactive with a list of (x,y) points and labels.
    'points' is a flattened list: [x1, y1, x2, y2, ...].
    'labels' is parallel: [1, 0, 1, ...].
    """
    payload = {
        "session_id": req.session_id,
        "points": req.points,
        "labels": req.labels,
        "multimask": req.multimask
    }
    resp = requests.post(
        f"{BACKEND_URL}/predict_interactive", json=payload, timeout=300
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    mask_list = [SAM_Mask(**mask_dict) for mask_dict in data['masks']]
    return MaskResponse(masks=mask_list)


@app.post("/predict_automatic")
def predict_automatic(req: AutoRequest):
    """
    Calls the backend /predict_automatic, returning automatically generated masks.
    """
    logger.info('Received predict_auto request. Waiting for backend...')
    payload = {"session_id": req.session_id}
    session = SESSIONS[req.session_id]
    session['last_updated'] = time.time()
    resp = requests.post(
        f"{BACKEND_URL}/predict_automatic", json=payload, timeout=300)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    # resp_data = resp.json()

    # auto_response = AutoBackendResponse(**resp.json())
    # masks_temp = auto_response.masks
    # logger.info(
    #     f'Received backend response, with {len(masks_temp)} masks')
    # masks_response = []
    # for m in masks_temp:
    #     md = m.model_dump()
    #     compressed = base64.b64decode(md['segmentation'])
    #     packed = zlib.decompress(compressed)
    #     md['segmentation'] = np.unpackbits(
    #         np.frombuffer(packed, dtype=np.uint8)).astype(bool)
    #     mask = _masks.Mask(**md)
    #     masks_response.append(mask_to_transfer(mask))

    # logger.info('Backend result received. Transferring to client.')
    data = resp.json()
    mask_list = [SAM_Mask(**mask_dict) for mask_dict in data['masks']]
    return MaskResponse(masks=mask_list)


@app.post("/save_masks")
def save_masks(req: SaveMasksRequest):
    """
    Takes a list of masks (req.masks) for a given image, compresses them,
    and writes them to a .pkl on disk. The file name uses .pkl instead of .png or .jpg.
    """
    try:
        # if not os.path.exists(full_path):
        #     raise HTTPException(
        #         status_code=404, detail="Image not found on frontend side.")
        base, ext = os.path.splitext(req.image_name)
        masks_file = base + ".pkl"
        # masks = []
        # for m in req.masks:  # store as list of lists
        #     m2 = _masks.Mask(**m)
        #     m2.compress()
        #     masks.append(m2)
        drops_path = MASKS_ROOT + masks_file
        logger.info(f'Saving to {drops_path}')
        data = pickle.dumps(req.masks)
        response = upload_to_drops(drops_path=drops_path, data=data)
        logger.info('Response: ', response.status_code, response.text)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail=f"Failed to save masks: {response.text}")
        return {"status": "ok", "saved_file": masks_file}

    except Exception as e:
        logger.error(f"Error saving masks to Drops: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to save masks: {str(e)}")


@app.get("/load_masks/{image_name}", response_model=MaskResponse)
def load_masks(image_name: str):
    """
    Load previously saved masks for a given image.
    Return them as a list of lists (flattened 0/1 arrays).
    """
    full_path = MASKS_ROOT + image_name
    # # if not os.path.exists(full_path):
    # #     raise HTTPException(
    # #         status_code=404, detail="Image not found on frontend side.")

    base, ext = os.path.splitext(full_path)
    masks_file = base + ".pkl"

    # if not os.path.exists(masks_file):
    #     return LoadMasksResponse(masks=[])
    fn1, fn2 = get_unique_filename(
        full_path=masks_file)
    if fn1 == fn2:  # Masks file not found
        logger.info(f'Masks not found. Returning empty list.')
        return MaskResponse(masks=[])
    masks_file = MASKS_ROOT + fn1
    logger.info(f'Fetching masks from {masks_file}...')
    masks_bytes = load_from_drops(masks_file)
    masks_pkl = pickle.loads(masks_bytes)
    # raw_data = zlib.decompress(compressed_masks)
    # masks = pickle.loads(raw_data)  # list of lists
    logger.info(f'{len(masks_pkl)} masks found.')
    return MaskResponse(masks=masks_pkl)


# @app.post('/delete_masks')
@app.get("/images/{path:path}")
def serve_image(path: str):
    logger.info(f'Image path is {path}')
    content = load_image(path)
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
    logger.info(f'Image retrieved. img_array = {img_array}')
    if path.lower().endswith('.png'):
        media_type = "image/png"
    elif path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        media_type = "image/jpeg"
    elif path.lower().endswith('.tiff'):
        pil_image = Image.open(BytesIO(content))
        output_buffer = BytesIO()
        pil_image.save(output_buffer, format="PNG")
        content = output_buffer.getvalue()
        media_type = "image/png"
    else:
        media_type = "image/jpg"
    return Response(content=content, media_type=media_type)

@app.get("/imagesNoBanner/{path:path}")
def serve_image(path: str):
    logger.info(f'Image path is {path}')
    content = load_image(path)
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
    logger.info(f'Image retrieved. img_array = {img_array}')
    if path.lower().endswith('.png'):
        media_type = "image/png"
    elif path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        media_type = "image/jpeg"
    elif path.lower().endswith('.tiff'):
        pil_image = Image.open(BytesIO(content))
        output_buffer = BytesIO()
        pil_image.save(output_buffer, format="PNG")
        content = output_buffer.getvalue()
        media_type = "image/png"
    else:
        media_type = "image/jpg"
    return Response(content=content, media_type=media_type)


@app.post("/release_session")
def release_session(req: ReleaseSessionRequest):
    """
    Tells the backend to free memory for the specified session_id.
    """
    if req.session_id in SESSIONS:
        payload = {"session_id": req.session_id}
        resp = requests.post(f"{BACKEND_URL}/release_session",
                             json=payload, timeout=300)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        del SESSIONS[req.session_id]
        logger.info(f'Released session {req.session_id}')

    return resp.json()  # { "status": "released" }


def load_image(img_path: str):
    """
    Loads an image from Drops stored at the given path.
    """
    base_url = "https://drops.steingart.ceec.echem.io/"
    if not img_path.startswith(base_url):
        img_path = base_url + img_path
    logger.info(f'Img path is {img_path}')
    # 2. Authenticate and request file list
    response = requests.get(img_path)

    if response.status_code == 200:
        return response.content
    else:
        print(
            f'Image not found, server response status code is {response.text}, {response.status_code}')
        raise Exception(f"Error retrieving image: {response.text}")


async def forward_request(request: Request, endpoint: str, path: str):
    url = f"{DROPS_URL}{endpoint}"
    auth = HTTPBasicAuth(USERNAME, PASSWORD)
    path = '/'.join(path.split(','))
    # logger.info(
    #    f'Attemping {request.method} request, with url {url}, path={path}')
    response = requests.post(url, auth=auth, json={"search": f"{path}"})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code,
                            detail="Error forwarding request")

    return JSONResponse(content=response.json(), status_code=response.status_code)

# Endpoint to proxy GET requests


@app.get("/proxy/list/{path:path}")
async def proxy_get_list(request: Request, drops_path: str):
    return await forward_request(request, "/list/", path=drops_path)

# Endpoint to proxy POST requests


@app.post("/proxy/list/{path:path}")
async def proxy_post_list(request: Request, path: str):
    return await forward_request(request, "/list/", path=path)


# def mask_to_transfer(m: _masks.Mask) -> MaskTransfer:
#     """
#     Convert a Mask dataclass with compressed segmentation
#     into a Pydantic MaskTransfer for JSON serialization.
#     """
#     # 1) Ensure it's compressed
#     if not m.compressed:
#         m.compress()

#     # 2) Now m.segmentation is compressed bytes
#     compressed_bytes = m.segmentation
#     b64_str = base64.b64encode(compressed_bytes).decode("utf-8")

#     return MaskTransfer(
#         segmentation_b64=b64_str,
#         area=m.area,
#         point_coords=m.point_coords,
#         bbox=m.bbox,
#         crop_box=m.crop_box,
#         stability_score=m.stability_score,
#         predicted_iou=m.predicted_iou,
#         shape=m.shape
#     )


# --------------------------------------------------------------------------------------
# LAUNCH
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
