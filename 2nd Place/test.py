import numpy as np, pandas as pd
import cv2,PIL,pyvips
import skimage.io as sk 
import os,sys,glob,shutil,time,random,gc,warnings,logging,math, multiprocessing,argparse
from datetime import timedelta
from fastai.vision.all import *

from inference import *
from utils import *
from wsi import slide,filters,tiles,util

START_TIME = time.time()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--file_meta")
args = parser.parse_args()

FILE_META = args.file_meta#'data/test_metadata.csv'
DIR_TILES = './workspace/tiles/test'
DIR_MODEL = './workspace/models' 
CONFIG = 'config.csv'

PAGE_IX_MULS = {1:16,2:8,3:4,4:2}

DIR_INPUT = {}
DIR_INPUT[48] = f'{DIR_TILES}/48/'
DIR_INPUT[64] = f'{DIR_TILES}/64/'

PAGES_TO_EXTRACT = {}
PAGES_TO_EXTRACT[48] = [2,3,4]
PAGES_TO_EXTRACT[64] = [2,3,4]

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

logger.info('done initial setup')


def get_preds_for_page(df_test,dir_output,df_page_config,page):
  ## generate placeholder training set
  train_fnames, train_tiles = create_dummy_train(n=4,width=64,height=64,dest=f'{dir_output}/{page}')
  ## create df_test
  df_test = df_test.append(pd.DataFrame(train_fnames).rename(columns={0:'filename'}))
  df_test['tissue_id'] = df_test.filename.str.split('.').str[0].values
  df_test = df_test.reset_index(drop=True)

  path_tiles = f'{dir_output}/{page}/'
  fnames = [p for p in os.listdir(path_tiles) if p.split('.')[1]=='jpeg']
  df1 = pd.DataFrame(fnames).rename(columns={0:'tissue_id'})
  df1['path'] = path_tiles + df1['tissue_id']
  df1['tissue_id'] = df1.tissue_id.str.rsplit('_',n=1,expand=True)[0]
  df1['tile_id'] = df1['path'].str.split('_').str[-1].str.split('.',expand=True)[0].astype(np.int16)

  n_tiles = []
  for ix, row in df_test.iterrows():
    fn = row.tissue_id
    tiles = [f for f in fnames if fn in f]
    n_tiles.append(len(tiles))
  df_test[f'n_tiles'] = n_tiles
  #create df_test train/ valid(test) splits
  df_test = pd.merge(df_test,df1,on='tissue_id').sort_values(by=['tissue_id','tile_id']).reset_index(drop=True)
  df_test['label'] = 0
  df_test['is_valid'] = 1
  df_test.loc[df_test.filename.isin(train_fnames),'is_valid'] = 0
    
  preds = do_inference(df_test,DIR_MODEL,df_page_config)
  return preds

df_test_main = pd.read_csv(FILE_META)
df_test_main = df_test_main[['filename']]
df_test_main['tissue_id'] = df_test_main.filename.str.split('.').str[0].values
logger.info('loaded test file')


##load models
models = sorted([m.split('.')[0] for m in os.listdir(DIR_MODEL) if m.split('.')[-1] == 'pth'])
logger.info(f'{len(models)} models available: {models}')
#model/ data config
df_config = pd.read_csv(CONFIG).dropna()
df_config = df_config[df_config['sub']].reset_index(drop=True)
logger.info(f'{len(df_config)} models in config: {df_config.model_name.to_list()}')

##DO INFERENCE
PREDICTIONS = []
##Inference for D=48
BASE_SZ = 48
for page in PAGES_TO_EXTRACT[BASE_SZ]:
  start_time_task = time.time()
  logger.info(f'$$$$$$$$ GETTING PREDS FOR DS: {BASE_SZ}, PAGE: {page} $$$$$$$$$')
  df_test = df_test_main.copy()
  df_page_config = df_config[(df_config.page==page) & (df_config.D==BASE_SZ) ]
  preds = get_preds_for_page(df_test,DIR_INPUT[BASE_SZ], df_page_config, page)
  PREDICTIONS.append(preds)

  elapsed = time.time() - start_time_task
  logger.info(f'$$$$$$$$ DONE GETTING PREDS FOR DS: {BASE_SZ}, PAGE: {page} TIME: {timedelta(seconds=elapsed)}')
  elapsed = time.time() - START_TIME
  logger.info(f'$$$$$$$$ TOTAL TIME ELAPSED: {timedelta(seconds=elapsed)}')
  gc.collect()

##Inference for D=64
BASE_SZ = 64
for page in PAGES_TO_EXTRACT[BASE_SZ]:
  start_time_task = time.time()
  logger.info(f'$$$$$$$$ GETTING PREDS FOR DS: {BASE_SZ}, PAGE: {page} $$$$$$$$$')
  df_test = df_test_main.copy()
  df_page_config = df_config[(df_config.page==page) & (df_config.D==BASE_SZ) ]
  preds = get_preds_for_page(df_test,DIR_INPUT[BASE_SZ], df_page_config, page)
  PREDICTIONS.append(preds)

  elapsed = time.time() - start_time_task
  logger.info(f'$$$$$$$$ DONE GETTING PREDS FOR DS: {BASE_SZ}, PAGE: {page} TIME: {timedelta(seconds=elapsed)}')
  elapsed = time.time() - START_TIME
  logger.info(f'$$$$$$$$ TOTAL TIME ELAPSED: {timedelta(seconds=elapsed)}')
  gc.collect()


##compile predictions
PREDICTIONS = [item for sublist in PREDICTIONS for item in sublist]
logger.info(f'predictions count: {len(PREDICTIONS)}')

df_test = df_test_main.copy()
df_test['tissue_id'] = df_test.filename.str.split('.').str[0].values
df_test = df_test.sort_values(by=['tissue_id']).reset_index(drop=True) ##sort to match predictions

preds = torch.stack(PREDICTIONS).sum(0)
preds = torch.sigmoid(preds).sum(1).round().numpy().astype(np.int8)
logger.info(f'UNIQUE PREDS: {np.unique(preds)}')
preds=np.clip(preds,0,3).astype(np.int8)
logger.info(f'CLIPPED PREDS: {np.unique(preds)}')

##create dataframe
df_submission = pd.DataFrame(columns=['filename']+[str(i) for i in range(NUM_CLASSES)])
df_submission['filename'] = df_test['filename'].values
df_submission =  df_submission.fillna(0)
for ix,row in df_submission.iterrows():
  df_submission.iloc[[ix],preds[ix]+1] = 1
  
logger.info(f'inference df shape {df_submission.shape}')

df_submission = df_submission.set_index('filename')
df_submission = df_submission.reindex(index=df_test_main['filename'])
df_submission = df_submission.reset_index()

logger.info(f'submission shape: {df_submission.shape}')
logger.info(f'submission types: {df_submission.dtypes}')

df_submission.to_csv('submission.csv',index=False)

elapsed = time.time() - START_TIME
logger.info(f'ALL DONE! TOTAL TIME: {timedelta(seconds=elapsed)}')
