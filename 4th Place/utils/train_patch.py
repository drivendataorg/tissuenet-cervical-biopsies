import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely import wkt
from openslide import OpenSlide

from utils.heatmap import crop_img
from utils.tissue_mask import get_tissue_mask


def get_one_at_same_level(wsi_path, geometry, crop_size, level):
    slide = OpenSlide(wsi_path)
    geo = wkt.loads(geometry)
    center_point_x = int((geo.bounds[0] + geo.bounds[2]) / 2)
    center_point_y = slide.level_dimensions[0][1] - int((geo.bounds[1] + geo.bounds[3]) / 2)
    
    x_start = int(center_point_x-crop_size/2*2**level)
    y_start = int(center_point_y-crop_size/2*2**level)
    
    return x_start, y_start



def get_labeled_df(train_anno, valid_list, level, crop_size, base_path):
    region_df = pd.DataFrame([], columns=['region', 'label', 'is_valid'])
    for idx in tqdm(range(len(train_anno))):
        filename = train_anno.iloc[idx]['filename']
        geometry = train_anno.iloc[idx]['geometry']

        wsi_path = os.path.join(base_path, filename)
        x_start, y_start = get_one_at_same_level(wsi_path, geometry, crop_size, level)
        label = train_anno.iloc[idx]['annotation_class']

        region = ','.join([wsi_path, str(int(x_start)), str(int(y_start)), str(int(level)), str(int(crop_size))]) 
        is_valid = True if filename in valid_list else False

        region_df.loc[idx] = [region, label, is_valid]
    
    return region_df


def get_zero_df(zero_list, valid_list, read_level, down_sample, up_level, patch_numbers, crop_size, base_path):
    df = []
    mark = 0
    for idx, item in tqdm(enumerate(zero_list)):
        if item in valid_list:
            continue

        wsi_path = base_path + item
        tis = get_tissue_mask(wsi_path, read_level, down_sample, up_level)
        
        if np.mean(tis) == 0:
            print(item)
            continue
        
        _, _, region_df = crop_img(tis, wsi_path, read_level, down_sample, crop_size)

        num = int(len(region_df) / patch_numbers)
        if num == 0:
            None
        else:
            region_df = region_df[::num][:patch_numbers]
        df = region_df if mark==0 else df.append(region_df)
        
        mark += 1
    df['label'] = df['label'].astype(int)
    return df
