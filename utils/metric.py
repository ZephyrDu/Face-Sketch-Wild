import numpy as np
from skimage.measure import compare_ssim
import image_similarity_measures
import os
import cv2 as cv
from PIL import Image

def FSIM(gt_img, test_img):
    """Calculate FSIM score.
    -------------------------
    """
    tmp_score=image_similarity_measures.evaluation(
        org_img_path=gt_img,
        pred_img_path=test_img,
        metrics='test_img')
    return tmp_score


def SSIM(gt_img, test_img):
    """Calculate ssim score using skimage toolkit.
    """
    test_img = np.array(test_img).astype(np.uint8)
    gt_img = np.array(gt_img).astype(np.uint8)
    tmp_score = compare_ssim(gt_img, test_img, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return tmp_score


def avg_score(test_dir, gt_dir, metric_name='ssim', smooth=False, sigma=75, verbose=False):
    """
    Read images from two folders and calculate the average score.
    """
    metric_name = metric_name.lower()
    all_score = []
    for name in sorted(sorted(os.listdir(gt_dir))):
        test_img = Image.open(os.path.join(test_dir, name)).convert('L')
        gt_img = Image.open(os.path.join(gt_dir, name)).convert('L')
        if smooth:
            test_img = cv.bilateralFilter(np.array(test_img),7,sigma,sigma)

        if metric_name == 'ssim':
            tmp_score = SSIM(gt_img, test_img)
        elif metric_name == 'fsim':
            tmp_score = FSIM(gt_img, test_img)
        if verbose:
            print('Image: {}, Metric: {}, Smooth: {}, Score: {}'.format(name, metric_name, smooth, tmp_score))
        all_score.append(tmp_score)
    return np.mean(np.array(all_score))

