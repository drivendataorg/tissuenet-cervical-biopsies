import numpy as np, pandas as pd
import cv2,PIL,pyvips
import skimage.io as sk 
import os,sys,glob,shutil,time,random,gc,warnings,logging,math, multiprocessing, argparse
from datetime import timedelta

from wsi import slide,filters,tiles,util

START_TIME = time.time()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--dir_input_tif")
parser.add_argument("--file_meta")
parser.add_argument("--dir_output")
args = parser.parse_args()

DIR_INPUT_TIF = args.dir_input_tif
FILE_INPUT_CSV = args.file_meta#'data/train_metadata_eRORy1H.csv'
DIR_OUTPUT_TILES = f'./workspace/tiles/{args.dir_output}/'# ./workspace/tiles/train/

PAGE_IX_MULS = {1:16,2:8,3:4,4:2}
DIR_OUTPUT = {}
DIR_OUTPUT[48] = f'{DIR_OUTPUT_TILES}/48/'
DIR_OUTPUT[64] = f'{DIR_OUTPUT_TILES}/64/'

PAGES_TO_EXTRACT = {}
PAGES_TO_EXTRACT[48] = [2,3,4]
PAGES_TO_EXTRACT[64] = [2,3,4]

for page in PAGES_TO_EXTRACT[48]:
    os.makedirs(f'{DIR_OUTPUT[48]}/{page}',exist_ok=True)

for page in PAGES_TO_EXTRACT[64]:
    os.makedirs(f'{DIR_OUTPUT[64]}/{page}',exist_ok=True)

slide.SRC_TRAIN_DIR = args.dir_input_tif
RANDOM_STATE = 41
def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
fix_seed(RANDOM_STATE)

logger.info('done initial setup')

def save_tiles_for_page(cur_page,name,image_path,df_tissue_tiles,dir_output,logger):
    patch_size = PATCH_SIZES_ACT[cur_page]
    slide = pyvips.Image.new_from_file(image_path, page=cur_page)
    RES_MUL = PAGE_IX_MULS[cur_page] #2**(base_page-cur_page)
    for idx, row in df_tissue_tiles.iterrows():
        if row.tile_id==MAX_TILES_PER_PAGE[cur_page]: ##generated maximum tiles for page, exit
            break
        y = row['Row Start']
        x = row['Col Start']

        if (y<0 or x<0):
            warnings.warn(f"bad coords for {name} x:{x} y:{y}", RuntimeWarning)  
        
        
        x1 = max(0,x)*RES_MUL
        y1 = max(0,y)*RES_MUL
        
        region_width = region_height = patch_size#PATCH_SIZES_ACT[cur_page]
        if x1 + region_width >slide.width:
            logger.info(f'reducing {name} since {x1} + {region_width} >{slide.width}')
            region_width = slide.width - x1
        if y1 + region_height >slide.height:
            logger.info(f'reducing {name} since {y1} + {region_height} >{slide.height}')
            region_height = slide.height - y1
        try:
            #method 2
            region = pyvips.Region.new(slide).fetch(x1, y1, region_width, region_height)
            bands = 3
            img = np.ndarray(
                buffer=region,
                dtype=np.uint8,
                shape=(region_height, region_width, bands))
            
            img = PIL.Image.fromarray(img)
            img.save(f'{dir_output}/{cur_page}/{name}_{idx}.jpeg', quality=90)
        except Exception as ex:
            logger.info(f'Failed for {name}. x: {x}, y: {y} x1: {x1}, y1: {y1} reg_w: {region_width}, reg_h: {region_height} ')
            logger.info(f'slide width: {slide.width} height: {slide.height}  cur_page: {cur_page}' )
            logger.info(f'exc: {ex}')
            logger.info(f"{os.popen('df -h').read()}")
       

def gen_tiles(DIR_INPUT_TIF,dir_output,df_tile_data,pages_to_extract):
    ix=-1
    for name,df in list(df_tile_data.groupby('tissue_id')):
        ix+=1
        logger.info(f'processing {ix}: {name}')
        image_path = f'{DIR_INPUT_TIF}/{name}.tif'
        df = df.sort_values(by='tile_id').reset_index(drop=True)
        for page in pages_to_extract:
            save_tiles_for_page(page,name,image_path,df,dir_output,logger)


def generate_tiles_for_slide_list(slide_names,dir_output,pages_to_extract):
  for slide_name in slide_names:
    # ##generate tiles
    df = pd.read_csv(f'{slide.TILE_DATA_DIR}/{slide_name}-tile_data.csv',skiprows=14).sort_values(by='Score',ascending=False).reset_index(drop=True)
      #filter scores
    df1 = df[df.Score>0]
    if len(df1)>=1:
      df = df1
    else:
      logger.info(f'Ignoring Score: {slide_name}')
    
    df['tile_id'] = df.index
    df['tissue_id'] = slide_name
    df['filename'] = df['tissue_id'] + '.tif'
    gen_tiles(DIR_INPUT_TIF,dir_output,df,pages_to_extract)
  

def multiprocess_generate_tiles(dir_output,pages_to_extract):
  slides_list = list(df_input.tissue_id.unique())
  num_slides = len(slides_list)

  num_processes = min(multiprocessing.cpu_count(),5)
  pool = multiprocessing.Pool(num_processes)

  if num_processes > num_slides:
    num_processes = num_slides
  slides_per_process = num_slides / num_processes

  tasks = []
  for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * slides_per_process + 1
    end_index = num_process * slides_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    sublist = slides_list[start_index - 1:end_index]
    tasks.append((sublist,dir_output,pages_to_extract))
    logger.info(f"Task # {num_process} Process slides {sublist}")
    
  # start tasks
  results = []
  for t in tasks:
    results.append(pool.apply_async(generate_tiles_for_slide_list, t))

  for result in results:
    _ = result.get()

df_input = pd.read_csv(FILE_INPUT_CSV)
df_input = df_input[df_input.filename.isin([f for f in os.listdir(DIR_INPUT_TIF) if f.split('.')[-1]=='tif'])].reset_index(drop=True)
df_input = df_input[['filename']]
df_input['tissue_id'] = df_input.filename.str.split('.').str[0].values
logger.info('loaded training file')


##Generate tiles
logger.info('************** GENERATING MASKS *********************')

NAMES = [n.split('.')[0] for n in df_input.filename.values]
df_submission = pd.DataFrame()

n_files = len(NAMES)
filters.multiprocess_apply_filters_to_images(image_name_list=NAMES)
elapsed = time.time() - START_TIME
logger.info(f'######### DONE GENERATING MASKS ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

##48
BASE_SZ = 48
tiles.TILE_SIZE_BASE = BASE_SZ
slide.TILE_DATA_DIR = os.path.join(slide.BASE_DIR, f"tile_data/{BASE_SZ}")
slide.TOP_TILES_DIR = os.path.join(slide.BASE_DIR, f"top_tiles/{BASE_SZ}")

MAX_TILES_PER_PAGE = {1:24,2:48,3:96,4:128} #maximum number of tiles to extract per page
PATCH_SIZES_ACT = {1:768,2:384,3:192,4:96}#patch size to extract for each page

logger.info(f'********* GENERATING TILE META {BASE_SZ} **********')
tiles.multiprocess_filtered_images_to_tiles(image_list=NAMES, display=False, save_summary=False, save_data=True, save_top_tiles=False)
elapsed = time.time() - START_TIME
logger.info(f'######### DONE GENERATING TILE META {BASE_SZ} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

logger.info(f'********* GENERATING TILES {BASE_SZ} **********')

multiprocess_generate_tiles(dir_output=DIR_OUTPUT[BASE_SZ],pages_to_extract=PAGES_TO_EXTRACT[BASE_SZ])

elapsed = time.time() - START_TIME
logger.info(f'######### DONE GENERATING TILES {BASE_SZ} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

##64
BASE_SZ = 64
tiles.TILE_SIZE_BASE = BASE_SZ
slide.TILE_DATA_DIR = os.path.join(slide.BASE_DIR, f"tile_data/{BASE_SZ}")
slide.TOP_TILES_DIR = os.path.join(slide.BASE_DIR, f"top_tiles/{BASE_SZ}")

MAX_TILES_PER_PAGE = {2:48,3:64,4:128} #maximum number of tiles to extract per page
PATCH_SIZES_ACT = {2:512,3:256,4:128}#patch size to extract for each page

logger.info(f'********* GENERATING TILE META {BASE_SZ} **********')
tiles.multiprocess_filtered_images_to_tiles(image_list=NAMES, display=False, save_summary=False, save_data=True, save_top_tiles=False)
elapsed = time.time() - START_TIME
logger.info(f'######### DONE GENERATING TILE META {BASE_SZ} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

logger.info(f'********* GENERATING TILES {BASE_SZ} **********')

multiprocess_generate_tiles(dir_output=DIR_OUTPUT[BASE_SZ],pages_to_extract=PAGES_TO_EXTRACT[BASE_SZ])

elapsed = time.time() - START_TIME
logger.info(f'######### DONE GENERATING TILES {BASE_SZ} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()
