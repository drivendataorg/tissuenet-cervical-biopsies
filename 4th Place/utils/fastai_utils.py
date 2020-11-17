import fastai
from openslide import OpenSlide
from fastai.vision.all import *

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


def get_weight(train_df):
    vc = train_df.label.value_counts()
    weight_1 = vc[0] / vc[1]
    weight_2 = vc[0] / vc[2]
    weight_3 = vc[0] / vc[3]
    
    return torch.Tensor([1., weight_1, weight_2, weight_3]).cuda()


def get_databunch(train_df, img_size, bs_size):
    path = './'
    data = ImageDataLoaders.from_df(
               train_df, 
               path,
               valid_col='is_valid',
               size=img_size,
               batch_tfms=[Normalize.from_stats(*imagenet_stats), *aug_transforms(do_flip=True, 
                                                                              flip_vert=True, 
                                                                              max_lighting=0.2,
                                                                              max_rotate=10., 
                                                                              max_zoom=1.2,
                                                                              max_warp=0.2) ],
               bs=bs_size,
               seed=1, item_tfms=Resize(img_size))
    
    print('train_ds: ', data.train_ds)
    print('valid_ds: ', data.valid_ds)
    print('train_wsi_num: ', len(set([item.split(',')[0].split('/')[-1] for item in data.train_ds.items['region'].tolist()])))
    print('valid_wsi_num: ', len(set([item.split(',')[0].split('/')[-1] for item in data.valid_ds.items['region'].tolist()])))
    
    return data


def train_schedule(model, data, weight, lr, epochs):
    learn = cnn_learner(data, 
                        model,
                        metrics=error_rate,
                        loss_func=CrossEntropyLossFlat(weight=weight))
    
    learn.fine_tune(epochs, freeze_epochs=1)

    return learn


def train(train_df, model, lr, epochs, img_size, bs_size):
    weight = get_weight(train_df)
    data = get_databunch(train_df, img_size, bs_size)
    learn = train_schedule(model, data, weight, lr, epochs)
    return learn
    