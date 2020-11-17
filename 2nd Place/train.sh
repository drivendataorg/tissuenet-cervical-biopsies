#dir_input_tif - directory with train TIF files
#file_meta - path to train metadata csv file
#dir_output - leave default. Tiles will be saved to ./workspace/tiles/train 


python preprocess.py --dir_input_tif data/ --file_meta data/train_metadata_eRORy1H.csv --dir_output train
python train.py --train_labels ./data/train_labels.csv --model_id 1
python train.py --train_labels ./data/train_labels.csv --model_id 2
python train.py --train_labels ./data/train_labels.csv --model_id 3
python train.py --train_labels ./data/train_labels.csv --model_id 4
python train.py --train_labels ./data/train_labels.csv --model_id 5
python train.py --train_labels ./data/train_labels.csv --model_id 6