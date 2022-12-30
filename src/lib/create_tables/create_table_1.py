from __future__ import division

from sklearn.metrics import confusion_matrix
import numpy as np
from skimage import io
import contextlib
import os
import cv2
import time
import sys

from tensorflow.keras.models import load_model

from models.unet import get_premade_unet
from inference import infer, fit_image
from external.tchoulack import get_tchoulack

def calc_IoU(output, gt):
    """
    Calculate IoU score
    :param: output: predicted binary mask
    :param: gt: ground truth binary mask
    :return: IoU score
    """
    intersection = np.logical_and(output, gt)
    union = np.logical_or(output, gt)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calc_dice(output, gt):
    """
    Calculate dice coefficient
    :param: output: predicted binary mask
    :param: gt: ground truth binary mask
    :return: Dice coefficient
    """
    dice = np.sum(output[gt==1])*2.0 / (np.sum(output) + np.sum(gt))
    return dice

def calc_s(output, gt):
    """
    Calculate sensitivity and specificity
    :param: output: predicted binary mask
    :param: gt: ground truth binary mask
    :return: sensitivity, specificity
    """
    TN, FP, FN, TP = confusion_matrix(gt.flatten(), output.flatten()).ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity

def tchoulack_metric_wrapper(img_paths, gt_paths):
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    specificity_scores = []
    times = []  

    for img_path, gt_path in zip(img_paths, gt_paths):
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)[:, :, 0]

        start_time = time.time()
        output = get_tchoulack(img_path, img)
        end_time = time.time()

        # create binary mask
        gt[gt>0] = 1
        output[output>0] = 1

        # calculate metrics
        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, specificity = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
        execution_time = (end_time - start_time) * 1000
        times.append(execution_time)
    
    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(specificity_scores)/len(specificity_scores), sum(times)/len(times)]

def adaptive_rpca_metric_wrapper(gt_paths):
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    specificity_scores = []
    
    mask_paths = "../data/generatedMasks184/"
    mask_paths = [mask_paths + f for f in os.listdir(mask_paths)]
    mask_paths.sort()

    for mask_path, gt_path in zip(mask_paths, gt_paths):
        mask = cv2.imread(mask_path)[:, :, 0]
        gt = cv2.imread(gt_path)[:, :, 0]

        # create binary mask
        gt[gt>0] = 1
        mask[mask>0] = 1

        # calculate metrics
        dice_scores.append(calc_dice(mask, gt))
        IoU_scores.append(calc_IoU(mask, gt))
        sensitivity, specificity = calc_s(mask, gt)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
    
    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(specificity_scores)/len(specificity_scores)]

def modified_dncnn_metric_wrapper(img_paths, gt_paths):

    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    specificity_scores = []
    times = []  

    model = load_model('../data/FCN_baseline.h5')

    # warm up model with dummy data for 100 iterations
    print("Warming up model...")
    original_stdout = sys.stdout
    with contextlib.redirect_stdout(open(os.devnull, 'w')): # mute unnecessary tensorflow output
        for i in range(100):
            img = np.random.rand(1, 256, 256, 3)
            model.predict(img)
    sys.stdout = original_stdout

    for img_path, gt_path in zip(img_paths, gt_paths):
        img = io.imread(img_path)
        img = img.astype('float') / 255.0
        img = np.expand_dims(img, axis=0)

        gt = cv2.imread(gt_path)[:, :, 0]
        gt[gt>0] = 1

        original_stdout = sys.stdout

        with contextlib.redirect_stdout(open(os.devnull, 'w')): # mute unnecessary tensorflow output
            start_time = time.time()
            specular_mask = model.predict(img)
            end_time = time.time()

        sys.stdout = original_stdout

        specular_mask[specular_mask > 0.6] = 1
        specular_mask[specular_mask <= 0.6] = 0
        specular_mask = specular_mask[0, :, :, 0]
        specular_mask = specular_mask.astype(np.uint8)
        
        output = specular_mask

        # calculate metrics
        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, specificity = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
        execution_time = (end_time - start_time) * 1000
        times.append(execution_time)

    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(specificity_scores)/len(specificity_scores), sum(times)/len(times)]


def model_metric_wrapper(model, img_paths, gt_paths, use_dilation):
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    specificity_scores = []
    times = []  

    # create dummy inferences to warm up model
    print("Warming up model...")
    for i in range(100):
        img = cv2.imread(img_paths[0])
        infer(img, model, use_enhanced=True, use_threshold=True, use_model=True, use_dilation=use_dilation)

    print("Calculating metrics...")
    for img_path, gt_path in zip(img_paths, gt_paths):
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        gt = fit_image(gt)[:, :, 0]
        gt[gt>0] = 1

        start_time = time.time()
        output = infer(img, model, use_enhanced=True, use_threshold=True, use_model=True, use_dilation=False)
        end_time = time.time()
        output[output>0] = 1

        # calculate metrics
        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, specificity = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
        execution_time = (end_time - start_time) * 1000
        times.append(execution_time)

    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(specificity_scores)/len(specificity_scores), sum(times)/len(times)]

def make_table_1():
    img_path = '../data/CVC-all/img/'
    gt_path = '../data/CVC-all/mask/'

    img_paths = [img_path + f for f in os.listdir(img_path)]
    gt_paths = [gt_path + f for f in os.listdir(gt_path)]
    img_paths.sort()
    gt_paths.sort()

    model = get_premade_unet()
    
    print("Making table_1... (Note: This may take a while)")

    with open('../../tables/table_1/table_1_output.txt', 'w') as f:
        with contextlib.redirect_stdout(f): # redirect print statements to text file
            
            # Tchoulack Algorithm
            print("Working on Tchoulack Algorithm...")
            dice_t, iou_t, sensitivity_t, _, time_t = tchoulack_metric_wrapper(img_paths, gt_paths)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_t, iou_t, sensitivity_t, time_t))

            # Adaptive RPCA
            # Note: time noted in paper was calculated in MATLAB code
            print("Working on Adaptive RPCA...")
            dice_r, iou_r, sensitivity_r, _ = adaptive_rpca_metric_wrapper(gt_paths)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}'.format(dice_r, iou_r, sensitivity_r))

            # Modified DnCNN
            print("Working on Modified DnCNN...")
            dice_d, iou_d, sensitivity_d, _, time_d = modified_dncnn_metric_wrapper(img_paths, gt_paths)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_d, iou_d, sensitivity_d, time_d))

            # SpecReFlow Det (-)
            print("Working on SpecReFlow Det (-)...")
            dice_det_, iou_det_, sensitivity_det_, _, time_det_ = model_metric_wrapper(model, img_paths, gt_paths, use_dilation=False)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_det_, iou_det_, sensitivity_det_, time_det_))

            # SpecReFlow Det
            print("Working on SpecReFlow Det...")
            dice_det, iou_det, sensitivity_det, _, time_det = model_metric_wrapper(model, img_paths, gt_paths, use_dilation=True)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_det, iou_det, sensitivity_det, time_det))

    print("Table 1 Complete")

    

    