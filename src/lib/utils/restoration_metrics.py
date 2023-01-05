from skimage.metrics import structural_similarity as origSSIM
from skimage.measure import label, regionprops
import numpy as np
import os 
import cv2
from math import log10, sqrt

def origPSNR(original, compressed):
    """
    Calculate PSNR between two images
    :param: original: original image
    :param: compressed: restored image
    :return: PSNR score
    """
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR has no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def mPSNR(img1, img2, mask):
    """
    Calculate modified PSNR between two images
    :param: img1: original image
    :param: img2: restored image
    :param: mask: mask of the region of interest
    :return: mPSNR score
    """
    # make copy bc python is by pass by reference
    img1_ = img1.copy()
    img2_ = img2.copy()
    mask_ = mask.copy()

    # binarize mask
    mask_[mask>0] = 1
    
    # get total number of "ON" pixels
    num_mask_pix = np.sum(mask_)

    # mask out irrelevant regions
    img1_[mask==0] = 0
    img2_[mask==0] = 0

    # turn to float
    img1_ = img1_.astype(np.float64)/255.
    img2_ = img2_.astype(np.float64)/255.
    mask_ = mask_.astype(np.float64)/1.

    # get squared error for each channel and then average errors
    channel_errors = []
    for i in range(img1_.shape[2]):
        subtract = img1_[:, :, i] - img2_[:, :, i]
        channel_errors.append(np.sum(np.square(subtract)))
    squared_error = np.mean(np.asarray(channel_errors))

    # get mask-wise MSE
    mse = squared_error/num_mask_pix

    # get PSNR
    return 10*log10(1./mse)

def masks_as_image(mask):
    """
    Takes a mask and returns bounding boxes for each mask
    :param: mask: mask of the region of interest
    :return: list of bounding boxes
    """
    output = []
    mask[mask>0] = 1
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props: 
        output.append([prop.bbox[1], prop.bbox[0], prop.bbox[4], prop.bbox[3]]) # TODO: check if this is correct - I think this is wrong lolz
    return np.asarray(output)


def mSSIM(img1_, img2_, mask_):
    """
    Calculate modified SSIM between two images
    :param: img1_: original image
    :param: img2_: restored image
    :param: mask_: mask of the region of interest
    :return: mSSIM score
    """
    # make copy bc python is by pass by reference
    img1 = img1_.copy()
    img2 = img2_.copy()
    mask = mask_.copy()
    mask[mask>0] = 1

    # get bounding boxes from dense pixel-wise segmentation mask
    output = masks_as_image(mask)
    all_ssim, mask_weight = [], []
    imgH, imgW = img1.shape[0], img1.shape[1]

    for bbox in output: 
        # calculate bounding box length
        beforeH = bbox[3]-bbox[1]
        beforeW = bbox[2]-bbox[0]

        # box height adjustment
        if beforeH < 8: 
            amount_left = 8-beforeH
            while amount_left > 0: 

                if bbox[1] > 0: 
                    bbox[1] -= 1
                    amount_left -= 1

                if amount_left == 0: 
                    break

                if bbox[3] < imgH: 
                    bbox[3] += 1
                    amount_left -= 1
        
        # box width adjustment 
        if beforeW < 8: 
            amount_left = 8-beforeW
            while amount_left > 0: 

                if bbox[0] > 0: 
                    bbox[0] -= 1
                    amount_left -= 1

                if amount_left == 0: 
                    break
                
                if bbox[2] < imgW: 
                    bbox[2] += 1
                    amount_left -= 1

        img1_cropped = img1[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        img2_cropped = img2[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        mask_cropped = mask[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        ssim = origSSIM(img1_cropped, img2_cropped, channel_axis=2, gaussian_weights=True, win_size=7, use_sample_covariance=False, data_range=255)
        all_ssim.append(ssim)
        mask_weight.append(np.sum(mask_cropped))

    # apply mask weights to ssim values
    weights = np.asarray(mask_weight, dtype=float)
    total = np.sum(weights)
    weights /= total
    ssims = np.asarray(all_ssim, dtype=float)
    return float(np.sum(weights*ssims))