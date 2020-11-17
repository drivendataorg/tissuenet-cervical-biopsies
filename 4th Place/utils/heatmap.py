import fastai
import numpy as np
import pandas as pd
from openslide import OpenSlide
from fastai.vision.all import *
from utils.tissue_mask import get_tissue_mask

def load_image(fn, mode=None):
    "Open and load a `PIL.Image` and convert to `mode`"

    wsi_name, x, y, level, crop_size = fn.split(',')
    x=int(x); y=int(y); level=int(level); crop_size=int(crop_size)
    
    im = OpenSlide(wsi_name[2:]).read_region((x, y), 
                                             level, 
                                             (crop_size, crop_size))
    
    im = im._new(im.im)
    return im.convert(mode) if mode else im

fastai.vision.core.load_image = load_image


def crop_img(tissue_mask, wsi_path, read_level, down_sample, img_size):
    wsi = OpenSlide(wsi_path)
    y_list, x_list = np.where(tissue_mask == 1)
    num_points = len(x_list)
    
    x_max, y_max = wsi.level_dimensions[0]
    mag_ratio = 2**read_level*down_sample
    crop_ratio = 2**read_level*img_size
    
    region_list = []
    x_start_list = []
    y_start_list = []
    
    for i in range(num_points):
        
        x_start = int(x_list[i]*mag_ratio-crop_ratio/2)
        y_start = int(y_list[i]*mag_ratio-crop_ratio/2)
        
        x_start = 0 if x_start < 0 else x_start
        y_start = 0 if y_start < 0 else y_start
        
        x_start = x_max-crop_ratio if x_start+crop_ratio>x_max else x_start
        y_start = y_max-crop_ratio if y_start+crop_ratio>y_max else y_start
        
        x_start_list.append(int(x_start))
        y_start_list.append(int(y_start))
 
        region_list.append(','.join([wsi_path, 
                                     str(int(x_start)),
                                     str(int(y_start)), 
                                     str(int(read_level)), 
                                     str(int(img_size))]))
    
    fake_region_label_list = np.random.randint(0, 1, len(region_list)).reshape(-1, 1)
    region_list = np.array(region_list).reshape(-1, 1)
    
    region_df = pd.DataFrame(np.concatenate([region_list, fake_region_label_list], axis=1), columns=['region', 'label'])
    return x_start_list, y_start_list, region_df


def get_heatmap(wsi_path, tissue_mask, x_start_list, y_start_list, region_df, model_path, model, img_size, infer_bs_size): 
    data = ImageDataLoaders.from_df(region_df, 
                                   size=img_size,
                                   valid_pct=0,
                                   batch_tfms=[Normalize.from_stats(*imagenet_stats)],
                                   bs=infer_bs_size,
                                   seed=1, item_tfms=Resize(img_size))
    
    learn = cnn_learner(data, model, metrics=error_rate, n_out=4, pretrained=False)
    learn.model_dir = '.'
    learn.load(model_path)
    
    p, t = learn.get_preds(ds_idx=0, reorder=True)
    p = p.numpy()
    
    test_x_y_points = ['_'.join(str(item).split(',')[1:3]) for item in learn.dls.train_ds.items['region']]
    
    except_h, except_w = tissue_mask.shape
    probs_map = np.zeros((4, except_h, except_w))
    y_list, x_list = np.where(tissue_mask == 1)
    
    for y_x in zip(y_start_list, x_start_list, y_list, x_list):
        y, x, y_ori, x_ori = y_x
        probs_map[:, y_ori, x_ori] = p[test_x_y_points.index(str(x) + '_' + str(y))]
    return probs_map


def process_one_tif(wsi_path, down_sample, read_level, model_path, model, img_size, infer_bs_size):
    tissue_mask = get_tissue_mask(wsi_path, 
                                  read_level, 
                                  down_sample)
    
    x_start_list, y_start_list, region_df = crop_img(tissue_mask, 
                                                     wsi_path, 
                                                     read_level, 
                                                     down_sample, 
                                                     img_size)
    probs_map = get_heatmap(wsi_path, 
                            tissue_mask, 
                            x_start_list, 
                            y_start_list, 
                            region_df, 
                            model_path,
                            model, 
                            img_size, 
                            infer_bs_size)
    
    result = np.concatenate([np.array(tissue_mask)[np.newaxis, :, :], probs_map], axis=0)
    return result
    