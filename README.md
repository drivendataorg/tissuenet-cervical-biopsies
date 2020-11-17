[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://s3.amazonaws.com/drivendata-public-assets/sfp_comp_image.jpg)


# TissueNet: Detect Lesions in Cervical Biopsies

## Goal of the Competition

A biopsy is a sample of tissue examined at a microscopic level to diagnose cancer or signs of pre-cancer. While most diagnoses are still made with photonic microscopes, digital pathology has developed considerably over the past decade as it has become possible to digitize slides into "virtual slides" or "whole slide images" (WSIs). These heavy image files contain all the information required to diagnose lesions as malignant or benign.

Making this diagnosis is no easy task. It requires specialized training and careful examination of microscopic tissue. Approaches in machine learning are already able to help analyze WSIs by measuring or counting areas of the image under a pathologist's supervision. In addition, computer vision has shown some potential to classify tumor subtypes, and in time may offer a powerful tool to aid pathologists in making the most accurate diagnoses.

This challenge focuses on epithelial lesions of the uterine cervix, and features a unique collection of thousands of WSIs collected from medical centers across France. The lesions in slides like these are most often benign (class 0), but some others have low malignant potential (class 1) or high malignant potential (class 2), and others may already be invasive cancers (class 3).

Using this unique dataset, the objective is to detect the most severe epithelial lesions of the uterine cervix present in these biopsy images.


## What's in this Repository

This repository contains code from winning competitors in the [TissueNet: Detect Lesions in Cervical Biopsies](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
---   | ---          | ---          | ---           | ---
1     | Tribvn-Healthcare [brice_tayart](https://www.drivendata.org/users/brice_tayart/)*    | 0.933 | 0.948 | In a first stage, use the annotated regions to train a DenseNet to classify lesion class. Train multiple models with different parameters, and ensemble their predictions. In a second stage, run the classifiers in a sliding window over all the tissue in the slide to generate heatmaps for each class. Summarize the heatmaps as a set of features, and pass those feature to an SVM for slide-level classification.
2     | [karelds](https://www.drivendata.org/users/karelds/)    | 0.922 | 0.935 | A multiple instance learning approach: for each slide, select the top N most informative tiles based on an existing heuristic that favors tiles containing more tissue, more hematoxylin relative to eosin stain, and higher color saturation. To accommodate features are different spatial scales, individual models were trained using the 2nd, 3rd, and 4th levels of the image pyramid. Selected tiles are passed through a convolutional neural network to generate image features, which are pooled and classified at the level of an individual slide.
3     | [kbrodt](https://www.drivendata.org/users/kbrodt/)    | 0.918 | 0.934 | Divide the slide into _N_ x _N_ tiles and score each tile based on pixel intensity. Stitch the top _K_ tiles together into a single image. For multiple values of _N_ and _K_, train a model using the EfficientNet-B0 architecture to predict lesion class from the stitched image. During inference, ensemble the model results by averaging their outputs.
4     | LifeIs2Short [AndrewTal](https://www.drivendata.org/users/AndrewTal/) [debut_kele](https://www.drivendata.org/users/debut_kele/) [amelie](https://www.drivendata.org/users/amelie/)  | 0.931 | 0.933 | Use the annotated regions to train a DenseNet 201 to classify lesion class. Segment each slide into tissue and non-tissue regions. Use the classifier to generate a probability heatmap for all tissue regions in a slide and summarize the heatmaps as a feature vector. Train an ensemble of classifiers (random forest, LGBM, XGB, AdaBoost, and Gradient boosting) to make a slide-level prediction from the feature vectors.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

*The first place participants declined to release their code.
