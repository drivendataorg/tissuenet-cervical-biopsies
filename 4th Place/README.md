# Readme

1. Requirement 

   See requirements.yml

2. Config

   All you need config are in 'Need config' block from Train.ipynb and Inference.ipynb

3. File description

   1. Train.ipynb: All steps of training.
   2. Inference.ipynb: All steps of inference.
   3. utils - extract_feature_probsmap.py: extrac feature from probsmap.
   4. utils - fastai_utils.py: some function for train with fastai.
   5. utils - train_patch.py: some function for train to get train data. 
   6. utils - heatmap.py: some function to generate heatmap.
   7. utils - ml_model.py: some function for wsi mechine learning classification model.
   8. utils - tissue_mask.py: generate tissue mask.

4. Results

   About 0.93 on private dateset A, a little high on private dataset B than A.

5. Hardware

   A RTX 2080Ti and A Intel E5-2697 v4

6. Time

   About 3 hours for training. (1.5h to train a classification model, 1.5h to generate heatmaps)

   About 1.5 hours to inference 1000 samples.

7. Optimizer Time

   1. Choose a sampler deeplearning model, like resnte50.
   2. Choose a high downsample ratio.