TissueNet 2nd Place Solution
==============================
## [TissueNet: Detect Lesions in Cervical Biopsies](https://www.drivendata.org/competitions/67/competition-cervical-biopsy)

## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.  
    ├── config.csv         <- Contains model parameters used during training/ inference.
    ├── download_models.sh <- Download pretrained models to ./workspace/models
    ├── environment.yml   <- The conda environment file for reproducing the analysis environment
    ├── test.sh            <- Preprocess test set and run inference
    ├── train.sh           <- Preprocess training set and train models
--------

### Training

```sh train.sh ```
This will preprocess the training set and train 6 models with 5 fold CV (total 30 models).

Model weights will be saved to ./workspace/models

You may need to change these arguments in the train.sh file
* `dir_input_tif` - directory with train TIF files
* `file_meta` - path to train train metadata csv file

### Inference 
(Optional) Download pretrained models used in the best submission. *Caution: this will replace models saved from above training*
``` 
sh download_models.sh
```
Models will be saved to ./workspace/models

Run inference

```sh test.sh ```

This will preprocess the test set, run inference and output `submission.csv` file. 

You may need to change these arguments in the test.sh file
* `dir_input_tif` - directory with test TIF files
* `file_meta` - path to test metadata csv file

