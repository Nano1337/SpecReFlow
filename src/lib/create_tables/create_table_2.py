import numpy as np
from skimage import io
import contextlib
import os
import cv2
import time
import sys

from create_tables.create_table_1 import calc_dice, calc_IoU, calc_s
from models.unet import get_premade_unet
from inference import infer, fit_image

def assess_raw_model(img_paths, gt_paths, model):
    """
    Assess the raw images using the model
    :param: img_paths: list of paths to images
    :param: gt_paths: list of paths to ground truth masks
    :param: model: model to use
    :return: list of dice scores, IoU scores, sensitivity scores, and times
    """
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    times = []  

    print("Warming up model...")
    for i in range(100):
        img = cv2.imread(img_paths[0])
        infer(img, model, use_enhanced=False, use_threshold=False, use_model=True, use_dilation=False)


    for img_path, gt_path in zip(img_paths, gt_paths):
        img = cv2.imread(img_path)
        gt = fit_image(cv2.imread(gt_path))[:, :, 0]
        gt[gt>0] = 1

        start_time = time.time()
        output = infer(img, model, use_enhanced=False, use_threshold=False, use_model=True, use_dilation=False)
        end_time = time.time()

        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, _ = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        times.append(end_time - start_time)

    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(times)/len(times)]

def assess_raw_thresholding(img_paths, gt_paths): 
    """
    Assess the raw images using thresholding
    :param: img_paths: list of paths to images
    :param: gt_paths: list of paths to ground truth masks
    :return: list of dice scores, IoU scores, sensitivity scores, and times
    """
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    times = []  

    for img_path, gt_path in zip(img_paths, gt_paths):
        img = cv2.imread(img_path)
        gt = fit_image(cv2.imread(gt_path))[:, :, 0]
        gt[gt>0] = 1

        start_time = time.time()
        output = infer(img, use_enhanced=False, use_threshold=True, use_model=False, use_dilation=False)
        end_time = time.time()

        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, _ = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        times.append(end_time - start_time)

    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(times)/len(times)]

def assess_raw_model_thresholding(img_paths, gt_paths, model):
    """
    Assess the raw images using the model and thresholding
    :param: img_paths: list of paths to images
    :param: gt_paths: list of paths to ground truth masks
    :param: model: model to use
    :return: list of dice scores, IoU scores, sensitivity scores, and times
    """
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    times = []  

    print("Warming up model...")
    for i in range(100):
        img = cv2.imread(img_paths[0])
        infer(img, model, use_enhanced=False, use_threshold=False, use_model=True, use_dilation=False)


    for img_path, gt_path in zip(img_paths, gt_paths):
        img = cv2.imread(img_path)
        gt = fit_image(cv2.imread(gt_path))[:, :, 0]
        gt[gt>0] = 1

        start_time = time.time()
        output = infer(img, model, use_enhanced=False, use_threshold=True, use_model=True, use_dilation=False)
        end_time = time.time()

        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, _ = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        times.append(end_time - start_time)

    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(times)/len(times)]

def assess_preprocessed_model(img_paths, gt_paths, model):
    """
    Assess the preprocessed images using the model
    :param: img_paths: list of paths to images
    :param: gt_paths: list of paths to ground truth masks
    :param: model: model to use
    :return: list of dice scores, IoU scores, sensitivity scores, and times
    """
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    times = []  

    print("Warming up model...")
    for i in range(100):
        img = cv2.imread(img_paths[0])
        infer(img, model, use_enhanced=True, use_threshold=False, use_model=True, use_dilation=False)


    for img_path, gt_path in zip(img_paths, gt_paths):
        img = cv2.imread(img_path)
        gt = fit_image(cv2.imread(gt_path))[:, :, 0]
        gt[gt>0] = 1

        start_time = time.time()
        output = infer(img, model, use_enhanced=True, use_threshold=False, use_model=True, use_dilation=False)
        end_time = time.time()

        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, _ = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        times.append(end_time - start_time)

    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(times)/len(times)]

def assess_preprocessed_thresholding(img_paths, gt_paths):
    """
    Assess the preprocessed images using the model and thresholding
    :param: img_paths: list of paths to images
    :param: gt_paths: list of paths to ground truth masks
    :param: model: model to use
    :return: list of dice scores, IoU scores, sensitivity scores, and times
    """
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    times = []  

    for img_path, gt_path in zip(img_paths, gt_paths):
        img = cv2.imread(img_path)
        gt = fit_image(cv2.imread(gt_path))[:, :, 0]
        gt[gt>0] = 1

        start_time = time.time()
        output = infer(img, use_enhanced=True, use_threshold=True, use_model=False, use_dilation=False)
        end_time = time.time()

        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, _ = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        times.append(end_time - start_time)

    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(times)/len(times)]

def assess_preprocessed_model_thresholding(img_paths, gt_paths, model):
    """
    Assess the preprocessed images using the model and thresholding
    :param: img_paths: list of paths to images
    :param: gt_paths: list of paths to ground truth masks
    :param: model: model to use
    :return: list of dice scores, IoU scores, sensitivity scores, and times
    """
    dice_scores = []
    IoU_scores = []
    sensitivity_scores = []
    times = []  

    print("Warming up model...")
    for i in range(100):
        img = cv2.imread(img_paths[0])
        infer(img, model, use_enhanced=True, use_threshold=False, use_model=True, use_dilation=False)


    for img_path, gt_path in zip(img_paths, gt_paths):
        img = cv2.imread(img_path)
        gt = fit_image(cv2.imread(gt_path))[:, :, 0]
        gt[gt>0] = 1

        start_time = time.time()
        output = infer(img, model, use_enhanced=True, use_threshold=True, use_model=True, use_dilation=False)
        end_time = time.time()

        dice_scores.append(calc_dice(output, gt))
        IoU_scores.append(calc_IoU(output, gt))
        sensitivity, _ = calc_s(output, gt)
        sensitivity_scores.append(sensitivity)
        times.append(end_time - start_time)

    return [sum(dice_scores)/len(dice_scores), sum(IoU_scores)/len(IoU_scores), sum(sensitivity_scores)/len(sensitivity_scores), sum(times)/len(times)]

def make_table_2():
    img_path = '../data/CVC-all/img/'
    gt_path = '../data/CVC-all/mask/'

    img_paths = [img_path + f for f in os.listdir(img_path)]
    gt_paths = [gt_path + f for f in os.listdir(gt_path)]
    img_paths.sort()
    gt_paths.sort()

    model = get_premade_unet()

    print("Making table_2... (Note: This may take a minute or two)")

    with open('../../tables/table_2/table_2_output.txt', 'w') as f:
        with contextlib.redirect_stdout(f): # redirect print statements to text file
            print("Working on Raw Images...")

            print("Assessing using Light U-net")
            dice_rm, iou_rm, sensitivity_rm, time_rm = assess_raw_model(img_paths, gt_paths, model)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_rm, iou_rm, sensitivity_rm, time_rm*1000))

            print("Assessing using Thresholding")
            dice_rt, iou_rt, sensitivity_rt, time_rt = assess_raw_thresholding(img_paths, gt_paths)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_rt, iou_rt, sensitivity_rt, time_rt*1000))

            print("Assessing using Light U-net + Thresholding")
            dice_rmt, iou_rmt, sensitivity_rmt, time_rmt = assess_raw_model_thresholding(img_paths, gt_paths, model)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_rmt, iou_rmt, sensitivity_rmt, time_rmt*1000))

            print("Working on Preprocessed Images...")

            print("Assessing using Light U-net")
            dice_pm, iou_pm, sensitivity_pm, time_pm = assess_preprocessed_model(img_paths, gt_paths, model)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_pm, iou_pm, sensitivity_pm, time_pm*1000))

            print("Assessing using Thresholding")
            dice_pt, iou_pt, sensitivity_pt, time_pt = assess_preprocessed_thresholding(img_paths, gt_paths)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_pt, iou_pt, sensitivity_pt, time_pt*1000))

            print("Assessing using Light U-net + Thresholding")
            dice_pmt, iou_pmt, sensitivity_pmt, time_pmt = assess_preprocessed_model_thresholding(img_paths, gt_paths, model)
            print('Dice: {:.4f}, IoU: {:.4f}, Sensitivity: {:.4f}, Time: {:.4f} ms'.format(dice_pmt, iou_pmt, sensitivity_pmt, time_pmt*1000))

    print("Table 2 Complete")