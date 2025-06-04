sement_duo is a webapp implementation of the [_Segment Anything_](https://github.com/facebookresearch/segment-anything) model from Meta (hereby refered to as SAM). SAM is a general purpose segmentation algorithm that works with any image. To be accurate, we currently use [SAM 2.1](https://github.com/facebookresearch/sam2), an updated version of SAM2 (SAM 2 suppots videos and is faster than SAM). 

It is implemented here with a "dual" architecture:
- A backend that runs locally on your device, and does the heavy lifting. It loads SAM and makes all the computation related to masking.
- A frontend that is in charge of receiving and requesting masks from the backend, of saving and loading masks from disk. It establishes the web interface for users to interact. The web interface has a file explorer based on Drops. When an image is selected in the file explorer, it displays it, asks the backend to compute an embedding that will be used for masking, and allows the user to create masks.

The dual architecture is needed when running on Apple Silicon because Docker currently does not support Apple Silicon's MPS (GPU acceleration). So the backend runs locally, and the frontend runs containerized in Docker.

# How to install

1. Create a Python 3.11 environment (with Conda, venv, etc.) and activate it.
2. Install the latest versions of torch. See https://pytorch.org/get-started/locally/. On Mac, you can run this command (automatically installs with MPS):
`pip3 install torch torchvision torchaudio`
3. Download SAM 2 from Git and install it (do not forget the '.'):
```
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

4. Verify that SAM 2 is installed correctly and that Pytorch uses the MPS:
```
import torch
import sam2
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```

5. Navigate to the folder where you want to download everything, and run:
`git clone https://gogs.ceec.echem.io/matthieu/sement_duo.git`
Then:
`pip install -r requirements.txt`

6. If you want to use custom/different SAM 2 endpoints, feel free to download them [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description). Download both config and checkpoints. Otherwise, the large config/checkpoints are already in '/backend/sam2_checkpoint'. 

7. Create a .env file within said `sement_duo` folder with the keys:
```
USERNAME=""
PASSWORD=""
DROPS_URL="https://drops.steingart.ceec.echem.io/"

MODEL_CONFIG =    ""
CHECKPOINT_PATH = ""
MASKS_ROOT = "/matthieu/code/sement_duo/masks/"
BACKEND_URL = "http://localhost:8001"
```
Where MODEL_CONFIG and CHECKPOINT_PATH are the *absolute* paths to these files. Username and password are your log-ins to Drops. MASKS_ROOT is where the masks will be saved for all images.

Both frontend/backend use the same utils functions found in the `utils`. If you need to separate the frontend and backend, copy/paste the `utils` folder into the backend/frontend folders and modify the imports in both `main.py`.

# How to run
1. To run the backend and frontend, go to the `sement_duo` base folder, and in Terminal/cmd/VS Code, with the proper Python env activated:

`uvicorn backend.main:app --port 8001`

`uvicorn frontend.main:app --port 8000`

2. Then, navigate to localhost:8000 and have fun.

# How to use
Assuming everything is running, you can now go to the frontend URL. You'll be greeted with a Drops-like file explorer on the left. 

From there, you can navigate to whichever folder you want. By default, it greets you in `/drops/matthieu/`. You can navigate to another base file by typing in the input below your folder path without the '/drops' (e.g. "/labdaq" will go to `/drops/labdaq`). Don't forget to start with a '/'.

Once you navigated to your folder, select your image. It will load the image in the center, compute an embedding and load previously saved masks if they exist. While the embedding is being computed or any work is in progress, the buttons below will be grayed out. The interface lets you:
- Zoom in/out your image by using your mouse wheel or pinching your trackpad.
- Move the image around by left-clicking (maintained) and dragging.
- See the current masks in the right panel. Selected masks are highlighted in that panel and will be visible with a green overlay on the image. if there are a lot of masks, selecting them all at the same time will make the page lag a bit while it computes the overlays.
- Generate a single mask. By holding "Shift", a left-click will set the current mouse position as a point to include inside the mask (green points), a right-click will tell the model to exclude the point from the mask (red points). Then, click "Get single mask". The more points you use, the better the mask usually.
- Undo the last point you set down with the corresponding button below
- Automatically generate masks for the whole image with "Auto Generate Masks". *This is a heavy operation that will take some time* (at least 30s) and may not generate the results expected. If you have only a few masks to generate, generate them one-by-one. 
- Delete all currently selected masks by pressing "Delete". 
- Save all masks (selected or not) to Drops with the "Save Masks" button. Masks are not dynamically saved and deleted as you go, but only when you click that button, so don't forget it. 

If you reload/quit the page prior to saving, then all progress will be lost.

# Some important details

Currently, whenever a user selects a new image, a new embedding is computed and stored in the backend along with a session ID. There is no garbage collection as of yet, so every once in a while, you might want to go to 'localhost:8001' (or whatever port you chose for the backend),  and release all sessions. 

All file loading/saving is made to drops, using the `requests` package. If necessary, you can load files from your local session by going to `utils/drops_utils.py` and create new functions that mimic the behavior of `get_dir_contents()`, `load_from_drops()`, `upload_to_drops()` and modify the references in both `main.py` (`get_dir_contents()` is also used in `get_unique_filename()` so you might want to change that reference too). Ultimately, I'll probably implement this myself with a `USE_LOCAL_FILES` bool in the .env to choose how you want to use it. 

Each time the "Save Masks" button is clicked, the current set of masks will be saved to Drops in the folder designated by `MASKS_ROOT` in frontend `main.py`. They are saved with name "{img_name}_{suffix}.pkl", where suffix is incremented by 1 each time a new set of masks is saved for the given image. Eg. if image name is "my_image.jpg", then it will be saved as `my_image.pkl`, then `my_image_1.pkl`, then `my_image_2.pkl`, etc.. It is like this for 2 reasons: 
1. Drops doesn't allow you to overwrite files.
2. This way, you can generate different sets of masks for each image if you need. Currently, if 2 different images have the same name, their masks will not be differentiated. 

# Future improvements
1. SAM 2 supports videos, aka it can track objects in a video automatically. We could implement this, at least in the backend.
2. I will be making a script to import masks from a given image. The masks can be imported in Python as a Mask object (see utils/masks.py). However, masks.py is not available in Pithy, hence why a dedicated script would be useful.
3. For SEM images, there is a banner to remove, I will do it.
3. Currently, the interface can only load the lastest set of masks for a given image. We might want to be able to load any set of saved masks, as well as to add custom suffix when saving for more clarity.
4. "Auto-generate Masks" button uses default settings for now. An advanced mode where the user can set its own parameters for auto generation might be useful, especially when different scales are at play.
5. Nothing stops us from using different segmenting algos nor from training our own. SAM 2 allows for post-training  (maybe even for training from the start, but not sure that'd be the way to go). For example. see [this](https://github.com/BAMresearch/automatic-sem-image-segmentation) and [that](https://www.sciencedirect.com/science/article/pii/S2352492823008188)