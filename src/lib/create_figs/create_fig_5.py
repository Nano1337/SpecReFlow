import cv2
import os
import numpy as np
from skimage import io

from tensorflow.keras.models import load_model

from external.tchoulack import get_tchoulack
from inference import infer, fit_image
from models.unet import get_premade_unet


def make_gt(output_dir, imgs, gts):
    """
    Get ground truth for figure 5
    """
    print("Working on ground truth...")
    for i, (img, gt) in enumerate(zip(imgs, gts)):
        img[gt>0] = (0, 255, 0)
        cv2.imwrite(os.path.join(output_dir, f"gt_{chr(ord('a') + i)}.png"), img)


def make_tchoulack(output_dir, img_paths, imgs, gts):
    """
    Get tchoulack output for figure 5
    """
    print("Working on tchoulack...")
    for i, (img_path, img, gt) in enumerate(zip(img_paths, imgs, gts)):
        spec_mask = get_tchoulack(img_path, img)
        intersection = np.bitwise_and(spec_mask, gt)
        img[spec_mask>0] = (255, 0, 0)
        img[intersection>0] = (0, 255, 0)
        cv2.imwrite(os.path.join(output_dir, f"tchoulack_{chr(ord('a') + i)}.png"), img)


def make_adaptive_rpca(output_dir, imgs, gts):
    """
    Get adaptive rpca output for figure 5
    Note: masks were previously generated from MATLAB code the adaptive RPCA method is run
          and saved to disk. This function just visualizes the outputs. For the full code, please see
          https://github.com/fu123456/SHDNet/tree/main/src/highlight_detection_tmi2019
    """
    print("Working on adaptive rpca...")
    generated_masks_paths = ["../data/generatedMasks184/36.bmp", "../data/generatedMasks184/20.bmp", "../data/generatedMasks184/28.bmp"]
    generated_masks = [cv2.imread(path)[:, :, 0] for path in generated_masks_paths]
    for i, (img, gt, generated_mask) in enumerate(zip(imgs, gts, generated_masks)):
        generated_mask[generated_mask>0] = 1
        gt[gt>0] = 1
        intersection = np.bitwise_and(generated_mask, gt)
        img[generated_mask>0] = (255, 0, 0)
        img[intersection>0] = (0, 255, 0)
        cv2.imwrite(os.path.join(output_dir, f"adaptive_rpca_{chr(ord('a') + i)}.png"), img)

def make_modified_dcnn(output_dir, img_paths, gts):
    """
    Get modified dcnn output for figure 5
    Note: there may be TensortFlow TensorRT environment errors - please ignore
          the expected output is still saved
    """
    print("Working on modified dcnn...")
    model = load_model('../data/FCN_baseline.h5')
    for i, (img_path, gt) in enumerate(zip(img_paths, gts)):
        img = io.imread(img_path)
        img = img.astype('float') / 255.0
        img = np.expand_dims(img, axis=0)
        gt[gt>0] = 1

        specular_mask = model.predict(img)
        specular_mask[specular_mask > 0.6] = 1
        specular_mask[specular_mask <= 0.6] = 0
        specular_mask = specular_mask[0, :, :, 0]
        specular_mask = specular_mask.astype(np.uint8)
        intersection = np.bitwise_and(specular_mask, gt)
        
        img = cv2.imread(img_path)
        img[specular_mask>0] = (255, 0, 0)
        img[intersection>0] = (0, 255, 0)

        cv2.imwrite(os.path.join(output_dir, f"modified_dcnn_{chr(ord('a') + i)}.png"), img)

def make_ours(output_dir, imgs, gts):
    """
    Get our output for figure 5
    """
    print("Working on ours...")
    model = get_premade_unet()
    for i, (img, gt) in enumerate(zip(imgs, gts)):
        gt[gt>0] = 1
        specular_mask = infer(img, model, use_enhanced=True, use_threshold=True, use_model=True)
        gt = fit_image(gt)
        intersection = np.bitwise_and(specular_mask, gt)
        img = fit_image(img)
        img[specular_mask>0] = (255, 0, 0)
        img[intersection>0] = (0, 255, 0)
        cv2.imwrite(os.path.join(output_dir, f"ours_{chr(ord('a') + i)}.png"), img)

def make_fig_5(): 
    output_dir = "../../figs/fig_5/"

    img_path_a = "../data/CVC-all/img/36.bmp"
    img_path_b = "../data/CVC-all/img/20.bmp"
    img_path_c = "../data/CVC-all/img/28.bmp"

    img_a = cv2.imread(img_path_a)
    img_b = cv2.imread(img_path_b)
    img_c = cv2.imread(img_path_c)

    gt_a = cv2.imread("../data/CVC-all/mask/36.bmp")[:, :, 0]
    gt_b = cv2.imread("../data/CVC-all/mask/20.bmp")[:, :, 0]
    gt_c = cv2.imread("../data/CVC-all/mask/28.bmp")[:, :, 0]

    img_paths = [img_path_a, img_path_b, img_path_c]
    imgs = [img_a, img_b, img_c]
    gts = [gt_a, gt_b, gt_c]

    # Create detection image data for figure 5
    make_gt(output_dir, [img.copy() for img in imgs], [gt.copy() for gt in gts])
    make_tchoulack(output_dir, img_paths, [img.copy() for img in imgs], [gt.copy() for gt in gts])
    make_adaptive_rpca(output_dir, [img.copy() for img in imgs], [gt.copy() for gt in gts])
    make_modified_dcnn(output_dir, img_paths, [gt.copy() for gt in gts])
    make_ours(output_dir, [img.copy() for img in imgs], [gt.copy() for gt in gts])