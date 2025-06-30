# ------------------------------------------------------------------------------------------------
# Set-up
# ------------------------------------------------------------------------------------------------

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"using device: {device}")
    
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    print("No GPU available")
    exit()





# ------------------------------------------------------------------------------------------------
# Loading the SAM 2 video predictor
# ------------------------------------------------------------------------------------------------

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))





# ------------------------------------------------------------------------------------------------
# Select an example video
# ------------------------------------------------------------------------------------------------

# We assume that the video is stored as a list of JPEG frames with filenames like `<frame_index>.jpg`.

# For your custom videos, you can extract their JPEG frames using ffmpeg (https://ffmpeg.org/) as follows:
# ```
# ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
# ```
# where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks ffmpeg to start the JPEG file from `00000.jpg`.

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "notebooks/videos/bedroom"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.show()





# ------------------------------------------------------------------------------------------------
# Initialize the inference state
# ------------------------------------------------------------------------------------------------

# SAM 2 requires stateful inference for interactive video segmentation, so we need to initialize an **inference state** on this video.

# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state` (as shown in the progress bar below).

inference_state = predictor.init_state(video_path=video_dir)





# ------------------------------------------------------------------------------------------------
### Example 2: Segment an object using box prompt
# ------------------------------------------------------------------------------------------------

# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `reset_state`.

predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array([300, 0, 500, 400], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)
# Note : when add new prompt, both existing prompt and new prompt require to be sent to the model with 'add_new_points_or_box'.

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.show()
# Then, to get the masklet throughout the entire video, we propagate the prompts using the `propagate_in_video` API.



# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }



# Render the segmentation results in real-time
import cv2

for frame_idx in range(len(frame_names)):
    # 원본 프레임 로딩
    frame_path = os.path.join(video_dir, frame_names[frame_idx])
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            color = (0, 255, 0)
            alpha = 0.4

            # ✅ 안전한 마스크: 2D, uint8, 값 0 또는 255
            mask = (out_mask > 0).astype(np.uint8) * 255
            mask = np.squeeze(mask)  # 혹시라도 (H, W, 1)일 경우 대비

            if np.count_nonzero(mask) == 0:
                continue  # 마스크가 비어있으면 contour 생략

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = frame.copy()
            cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 화면에 표시
    cv2.imshow("SAM2 Segmentation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(30)  # 30ms 대기 → 약 33fps

    if key == 27:  # ESC key to break
        break

cv2.destroyAllWindows()




