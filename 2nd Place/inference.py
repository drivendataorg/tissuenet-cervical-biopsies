import numpy as np, pandas as pd
import cv2,PIL,math
import os,sys,shutil,time,random,gc,warnings,logging

from fastai.vision.all import *
from fastai.callback.cutmix import *
from torchvision.models.resnet import ResNet, Bottleneck
from resnest.torch import resnest50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 4 
bs = 2

mean = [0.485, 0.456, 0.406]
std =  [0.229, 0.224, 0.225]
mean = torch.tensor(mean)
std = torch.tensor(std)

PARAMS = {}
PARAMS['n_tiles'] = 0
PARAMS['max_nth_tile'] = 0
PARAMS['sz'] = 0

RANDOM_STATE = 41
def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
fix_seed(RANDOM_STATE)


class TissueTiles(tuple):
    @classmethod
    def create(cls, fns): return cls(tuple(PILImage.create(f) for f in fns))

def TissueTilesBlock(): 
    return TransformBlock(type_tfms=TissueTiles.create, batch_tfms=None)

class TissueGetItems():
    def __init__(self, path_col, tissue_id_col, label_col, max_nth_tile, shuffle=False):
        self.path = path_col
        self.tissue_id = tissue_id_col
        self.label = label_col #tissue label
        self.shuffle = shuffle
        self.max_nth_tile = max_nth_tile
        
    def __call__(self, df):
        data = []
        for tissue_id in progress_bar(df[self.tissue_id].unique()):
            tiles = df[(df[self.tissue_id]==tissue_id) & (df['tile_id']<self.max_nth_tile)]
            fns = tiles[self.path].tolist()
            lbl = tiles[self.label].values[0]
            data.append([*fns, lbl])
        return data

def create_batch(data):
    n_tiles = PARAMS['n_tiles']
    max_nth_tile = PARAMS['max_nth_tile']
    
    xs, ys = [], []
    for d in data:
        img = d[0]
        n_available = len(img)
        ixs = []
        choice_range = min(n_available,max_nth_tile)
        if n_available<n_tiles:
          ixs = list(np.random.choice(range(choice_range),size=n_tiles,replace=True))
        else:
           ixs = list(np.random.choice(range(choice_range),size=n_tiles,replace=False))
        ixs = sorted(list(ixs))
        ##normalize
        img = tuple([ ((img[i]/255.) -  mean[...,None,None])/std[...,None,None] for i in ixs])
        xs.append(img)
        ys.append(d[1])

    xs = torch.cat([TensorImage(torch.cat([im[None] for im in x], dim=0))[None] for x in xs], dim=0)
    ys = torch.cat([y[None] for y in ys], dim=0)
    return TensorImage(xs), TensorCategory(ys)

def show_tile_batch(max_rows = 2, max_cols = 5):
  xb, yb = dls.one_batch()
  fig, axes = plt.subplots(ncols=max_cols, nrows=max_rows, figsize=(12,6), dpi=120)
  for i in range(max_rows):
      xs, ys = xb[i], yb[i]
      for j, x in enumerate(xs):
          if j== max_cols:
            break
          x = x.cpu()
          x = mean[...,None,None]+x*std[...,None,None]
          axes[i,j].imshow(x.permute(1,2,0).numpy())
          axes[i,j].set_title(ys.item())
          axes[i,j].axis('off')

class SequenceTfms(Transform):
    def __init__(self, tfms): 
        self.tfms = tfms
        
    def encodes(self, x:TensorImage): 
        
        bs, seq_len, ch, rs, cs = x.shape
        x = x.view(bs, seq_len*ch, rs, cs)
        x = compose_tfms(x, self.tfms)
        x = x.view(bs, seq_len, ch, rs, cs) 
        return x
    
class BatchTfms(Transform):
    def __init__(self, tfms): 
        self.tfms = tfms
        
    def encodes(self, x:TensorImage): 
        bs, seq_len, ch, rs, cs = x.shape
        x = x.view(bs*seq_len, ch, rs, cs)
        x = compose_tfms(x, self.tfms)
        
        x = x.view(bs, seq_len, ch, rs, cs) 
        return x


"""### Model"""
class Model(nn.Module):
    def __init__(self,arch,norm,pre=False):
        super().__init__()
        if arch == 'ResX50':
          m = ResNet(Bottleneck,  [3, 4, 6, 3], groups=32,width_per_group=4)
          print('Loaded Model RexNextssl')

        elif arch == 'ResS50':
          m = resnest50(pretrained=pre)
          print('Loaded model ResNest')

        blocks = [*m.children()]
        enc = blocks[:-2]
        self.enc = nn.Sequential(*enc)
        C = blocks[-1].in_features
        head = [AdaptiveConcatPool2d(),
                Flatten(), #bs x 2*C
                nn.Linear(2*C,512),
                Mish()
                ]
        if norm == 'GN':
          head.append(nn.GroupNorm(32,512))
          print('Group Norm')
        elif norm == 'BN':
          head.append(nn.BatchNorm1d(512))
          print('Batch Norm')
        else:
          print('No Norm')
        
        head.append(nn.Dropout(0.5))
        head.append(nn.Linear(512,NUM_CLASSES-1))
        self.head = nn.Sequential(*head)                              

    def forward(self, *x):
        shape = x[0].shape 
        n = shape[1]## n_tiles
        x = torch.stack(x,1).view(-1,shape[2],shape[3],shape[4])    
        x = self.enc(x)
        shape = x.shape        
        #concatenate the output of tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        x = self.head(x)
        return x
        


class BinBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, inputs, targets):
        targ_bin  = torch.zeros_like(inputs)
        for i in range(targets.shape[0]):
          targ_bin[i,:targets[i]] = 1
        targ_bin = targ_bin.to(device)
        loss = self.criterion(inputs, targ_bin)
        return loss

importance_weights = torch.ones(NUM_CLASSES-1, dtype=torch.float).to(device)
# importance_weights = torch.tensor([1., 0.7, 1.]).to(device)

def label_to_levels(label, num_classes):
    levels = [1]*label + [0]*(num_classes - 1 - label)
    levels = torch.tensor(levels, dtype=torch.float32)
    return levels

def loss_fn2(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)

class CORAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = loss_fn2
    def forward(self, inputs, targets):
        levels = []
        for label in targets:
            levels_from_label = label_to_levels(label, num_classes=NUM_CLASSES)
            levels.append(levels_from_label)
        levels = torch.stack(levels).to(device)
        loss = loss_fn2(inputs, levels, imp=importance_weights)
        return loss

def get_dataloader(df_test):
  n_tiles = PARAMS['n_tiles']
  max_nth_tile = PARAMS['max_nth_tile']
  sz = PARAMS['sz']
  print(f'inference size: {sz}')

  def splitter(lst):
    df = pd.DataFrame([l[0] for l in lst]).rename(columns={0:'path'})
    df = pd.merge(df,df_test).drop_duplicates(subset=['tissue_id']).sort_values(by=['tissue_id']).reset_index(drop=True)
    train = df[df.is_valid!=1].index.to_list()
    valid = df[df.is_valid==1].index.to_list()
    return train,valid

  affine_tfms, light_tfms = aug_transforms(flip_vert=False, max_rotate=25.0, max_zoom=1.1, max_warp=0., 
                        p_affine=0.75, p_lighting=0., xtra_tfms=None)

  def get_x(t):
    return t[:-1]
  def get_y(t):
    return t[-1]
  dblock = DataBlock(
      blocks    = (TissueTilesBlock, CategoryBlock),
      get_items = TissueGetItems('path', 'tissue_id', 'label',max_nth_tile), 
      get_x     = get_x,
      get_y     = get_y,
      splitter  = splitter,
      item_tfms  = Resize(sz),
      batch_tfms = [SequenceTfms([affine_tfms]), 
                    # BatchTfms([brightness, contrast]),
                    ]
      )

  dls = dblock.dataloaders(df_test, bs=bs, create_batch=create_batch)
  return dls

def do_inference(df_test,dir_model,df_page_config):
  page_preds = []
  for _,config in df_page_config.iterrows():
    
    print(f'%%%%%% running inference for {config.model_name}')
    ##update params
    PARAMS['n_tiles'] = int(config.N)
    PARAMS['max_nth_tile'] = int(config.N_MAX)
    PARAMS['sz'] = int(config.sz)

    ##set seed
    ##fix_seed(int(config.seed)) 
    ##create learner
    dls = get_dataloader(df_test)
    model = Model(config.arch,config.norm)
    if config.loss=='CRL':
      print ('CORAL loss')
      loss_func = CORAL()
    elif config.loss=='BCE':
      print ('BCE loss')
      loss_func = BinBCELoss()
    
    learn = Learner(dls,model,loss_func=loss_func,cbs=[BnFreeze]).to_fp16()
    learn.clip_grad = 1.0
    learn.model_dir = dir_model
    learn.load(f'{config.model_name}')
    learn.model.eval()

    if config.do_tta:
      preds = learn.tta(ds_idx=1)[0]
    else:
       preds = learn.get_preds(ds_idx=1)[0]
    #if weight!=1
    preds = preds*config.weight
    print('preds: ',preds.shape)
    page_preds.append(preds)
    gc.collect()
  return page_preds