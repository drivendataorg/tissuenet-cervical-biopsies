#dir_input_tif - directory with test TIF files
#file_meta - path to test metadata csv file
#dir_output - leave default. Tiles will be saved to ./workspace/tiles/test

python preprocess.py --dir_input_tif data/ --file_meta data/test_metadata.csv --dir_output test
python test.py --file_meta ./data/test_metadata.csv