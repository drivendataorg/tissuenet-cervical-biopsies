import cv2
import numpy as np
from PIL import Image as PImage
from openslide import OpenSlide


def get_tissue_mask(wsi_path, read_level, down_sample, up_level=3):
    slide = OpenSlide(wsi_path)
    mag_level = int(np.log2(down_sample) + read_level)
    ratio = 1 if mag_level <= slide.level_count-1 else 2**(mag_level - slide.level_count+1)
    level = mag_level if mag_level <= slide.level_count-1 else slide.level_count-1

    img_RGB = np.array(slide.read_region((0, 0),
                       level-up_level,
                       slide.level_dimensions[level-up_level]))
    
    black_points = (img_RGB[:, :, 0] < 20) & (img_RGB[:, :, 0] < 20)  & (img_RGB[:, :, 0] < 20)
    img_RGB[:, :, 0][black_points] = 255.
    img_RGB[:, :, 1][black_points] = 255.
    img_RGB[:, :, 2][black_points] = 255.
    
    slide_lv = cv2.cvtColor(np.array(img_RGB), cv2.COLOR_RGBA2RGB)
    slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
    slide_lv = slide_lv[:, :, 1]
    
    _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tissue_mask[tissue_mask != 0] = 1
    
    if (1 in np.mean(tissue_mask, axis=0)) or (1 in np.mean(tissue_mask, axis=1)):
        slide_lv[slide_lv <= 20] = 0
        _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        tissue_mask[tissue_mask != 0] = 1
    
    new_h = int(np.ceil(tissue_mask.shape[1] / 2**up_level / ratio))
    new_w = int(np.ceil(tissue_mask.shape[0] / 2**up_level / ratio))

    tissue_mask = np.array(tissue_mask, np.uint8)
    tissue_mask_pil = np.array(PImage.fromarray(tissue_mask).resize((new_h, new_w)))
    tissue_mask_cv2 = np.array(cv2.resize(tissue_mask, (new_h, new_w)))
    
    result = np.array(tissue_mask_pil | tissue_mask_cv2, np.uint8)
    return result
