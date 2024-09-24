import torch
import requests
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image, ImageOps
from diffusers import DiffusionPipeline
from transparent_background import Remover

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

np.random.seed(3)

device = torch.device("cuda")
sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)



#TODO: Load Yahoo photo-background-generation Model
model_id = "yahoo-inc/photo-background-generation"
pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
pipeline = pipeline.to('cuda')

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def yahoo_bg_generator(img, prompt):
    # model_id = "yahoo-inc/photo-background-generation"
    # pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
    # pipeline = pipeline.to('cuda')

    seed = 0
    # img = Image.open(image_path)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = resize_with_padding(img, (512, 512))

    # Load background detection model
    remover = Remover() # default setting
    remover = Remover(mode='base') # nightly release checkpoint

    # Get foreground mask
    fg_mask = remover.process(img, type='map') # default setting - transparent background


    seed = 13
    mask = ImageOps.invert(fg_mask)
    img = resize_with_padding(img, (512, 512))
    generator = torch.Generator(device='cuda').manual_seed(seed)
    
    cond_scale = 1.0
    with torch.autocast("cuda"):
        controlnet_image = pipeline(
            prompt=prompt, image=img, mask_image=mask, control_image=mask, num_images_per_prompt=1, generator=generator, num_inference_steps=20, guess_mode=False, controlnet_conditioning_scale=cond_scale
        ).images[0]

    return controlnet_image

def show_mask(image, mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    alpha_expanded = np.expand_dims(mask, axis=2)
    rgba_img = np.concatenate((image, alpha_expanded*255), axis=2)
    ax.imshow(rgba_img)
    # ax.imshow(image)
    return rgba_img

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        show_mask(image, mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            pass
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.show()
        break # Just show mask 1

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # 關閉圖表
    plt.close()
    # 使用 PIL 打開圖像
    img = Image.open(buf)
    return img

#TODO: Load SAM2
def SAM(image_path, coords_str):

    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)

    # Parse coordinates
    # coords = coords_str.strip('[]').split(',')
    # x = int(float(coords[0].strip()))
    # y = int(float(coords[1].strip()))
    # print(f"Type of x and y is: {type(x)}")
    # input_point = np.array([[x, y]])
    # input_label = np.array([1])
    coords_str = coords_str.strip('[]')
    list_of_lists = []
    for sublist in coords_str.split('], ['):
        # Split the sublist into individual numbers and convert them to integers
        elements = [int(x) for x in sublist.split(', ')]
        list_of_lists.append(elements)
    input_points = np.array(list_of_lists)
    input_labels = np.tile(np.array([1]), len(input_points))
    print(input_labels)
    masks, scores, logits = predictor.predict(point_coords=input_points,
                                              point_labels=input_labels,
                                              multimask_output=True,)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    img = show_masks(image, masks, scores, point_coords=input_points, input_labels=input_labels, borders=True)

    return img

if __name__ == "__main__":
    img = Image.open("test.jpg")
    img = yahoo_bg_generator(img=img, prompt = 'A dark swan in a bedroom')
    img.save("yahoo_bg_generator_result.jpg")
