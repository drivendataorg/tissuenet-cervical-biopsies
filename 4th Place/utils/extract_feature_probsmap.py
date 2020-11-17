import cv2
import numpy as np
import scipy.stats.stats as st
from skimage.measure import label, regionprops

MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS = 0, 1, 2, 3, 4

class extractor_features(object):
    def __init__(self, probs_map, tissue_mask):
        self._probs_map = probs_map
        self._tissue_mask = tissue_mask

    def get_region_props(self, probs_map_threshold):
        labeled_img = label(probs_map_threshold)
        return regionprops(labeled_img, intensity_image=self._probs_map)

    def probs_map_set_p(self, threshold):
        probs_map_threshold = np.array(self._probs_map)

        probs_map_threshold[probs_map_threshold < threshold] = 0
        probs_map_threshold[probs_map_threshold >= threshold] = 1

        return probs_map_threshold

    def get_num_probs_region(self, region_probs):
        return len(region_probs)

    def get_tumor_region_to_tissue_ratio(self, region_props):
        tissue_area = cv2.countNonZero(self._tissue_mask) + 1
        tumor_area = 0

        n_regions = len(region_props)
        for index in range(n_regions):
            tumor_area += region_props[index]['area']

        return float(tumor_area) / tissue_area

    def get_largest_tumor_index(self, region_props):

        largest_tumor_index = -1
        largest_tumor_area = -1

        n_regions = len(region_props)
        for index in range(n_regions):
            if region_props[index]['area'] > largest_tumor_area:
                largest_tumor_area = region_props[index]['area']
                largest_tumor_index = index

        return largest_tumor_index

    def f_area_largest_tumor_region_t50(self):
        pass

    def get_longest_axis_in_largest_tumor_region(self,
                                                 region_props,
                                                 largest_tumor_region_index):
        largest_tumor_region = region_props[largest_tumor_region_index]
        return max(largest_tumor_region['major_axis_length'],
                   largest_tumor_region['minor_axis_length'])

    def get_average_prediction_across_tumor_regions(self, region_props):
        # close 255
        if len(region_props) > 0:
            region_mean_intensity = [region.mean_intensity for region in region_props]
            return np.mean(region_mean_intensity)
        else:
            return 0

    def get_feature(self, region_props, n_region, feature_name):
        feature = [0] * 5
        if n_region > 0:
            feature_values = [region[feature_name] for region in region_props]
            feature[MAX] = np.log1p(format_2f(np.max(feature_values)))
            feature[MEAN] = np.log1p(format_2f(np.mean(feature_values)))
            feature[VARIANCE] = np.log1p(format_2f(np.var(feature_values)))
            feature[SKEWNESS] = format_2f(st.skew(np.array(feature_values)))
            feature[KURTOSIS] = format_2f(st.kurtosis(np.array(feature_values)))

        return feature

def format_2f(number):
    return float("{0:.2f}".format(number))

def compute_features(extractor):
    features = []  # all features

    probs_map_threshold_p90 = extractor.probs_map_set_p(0.9)  # 0.9
    probs_map_threshold_p50 = extractor.probs_map_set_p(0.5)  # 0.5

    region_props_p90 = extractor.get_region_props(probs_map_threshold_p90)
    region_props_p50 = extractor.get_region_props(probs_map_threshold_p50)

    f_count_tumor_region = np.log1p(extractor.get_num_probs_region(region_props_p90))  # 1
    features.append(f_count_tumor_region)

    f_percentage_tumor_over_tissue_region = extractor.get_tumor_region_to_tissue_ratio(region_props_p90)  # 2
    features.append(f_percentage_tumor_over_tissue_region)

    largest_tumor_region_index_t50 = extractor.get_largest_tumor_index(region_props_p50)
    try:
        f_area_largest_tumor_region_t50 = np.log1p(region_props_p50[largest_tumor_region_index_t50].area)  # 3
    except:
        f_area_largest_tumor_region_t50 = 0
    features.append(f_area_largest_tumor_region_t50)
    try:
        f_longest_axis_largest_tumor_region_t50 = np.log1p(
            extractor.get_longest_axis_in_largest_tumor_region(region_props_p50,
                                                               largest_tumor_region_index_t50))  # 4
    except:
        f_longest_axis_largest_tumor_region_t50 = 0

    features.append(f_longest_axis_largest_tumor_region_t50)

    f_pixels_count_prob_gt_90 = np.log1p(cv2.countNonZero(probs_map_threshold_p90))  # 5
    features.append(f_pixels_count_prob_gt_90)

    f_avg_prediction_across_tumor_regions = extractor.get_average_prediction_across_tumor_regions(region_props_p90)  # 6
    features.append(f_avg_prediction_across_tumor_regions)

    f_area = extractor.get_feature(region_props_p90, f_count_tumor_region, 'area')  # 7,8,9,10,11
    features += f_area

    f_perimeter = extractor.get_feature(region_props_p90, f_count_tumor_region, 'perimeter')  # 12,13,14,15,16
    features += f_perimeter

    f_eccentricity = extractor.get_feature(region_props_p90, f_count_tumor_region, 'eccentricity')  # 17,18,19,20,21
    features += f_eccentricity

    f_extent_t50 = extractor.get_feature(region_props_p50, len(region_props_p50), 'extent')  # 22,23,24,25,26
    features += f_extent_t50

    f_solidity = extractor.get_feature(region_props_p90, f_count_tumor_region, 'solidity')  # 27,28,29,30,31
    features += f_solidity

    return features


def get_probsmap_feature(probs_map, mode='Train'):
    probs_map = np.load(probs_map) if mode=='Train' else probs_map
    pred = np.argmax(probs_map[1:, :, :], axis=0)
    feature_1 = np.sum(pred == 1)  # Class 0 patch number
    feature_2 = np.sum(pred == 2)  # Class 1 patch number
    feature_3 = np.sum(pred == 3)  # Class 2 patch number

    probs_map = np.concatenate([probs_map, np.sum(probs_map[2:], axis=0, keepdims=True)], axis=0)
    probs_map = np.concatenate([probs_map, np.sum(probs_map[3:5], axis=0, keepdims=True)], axis=0)

    features_list = []
    for i in range(probs_map.shape[0] - 1):
        extractor = extractor_features(probs_map[i + 1], probs_map[0])
        feature = compute_features(extractor)
        features_list.append(feature)

    return np.array(features_list).reshape(-1).tolist() + [feature_1, feature_2, feature_3]