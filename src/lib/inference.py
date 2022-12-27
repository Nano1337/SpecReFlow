from models.unet import get_premade_unet
from utils.image import postprocess, preprocess, reflection_enhance
import torch
import numpy as np
import cv2
import os 

def fit_image(image):
    """
    Check image dimensions to see if it's divisible by 8, if not then cut either side to make it so
    """
    h, w = image.shape[:2]
    if h % 8 != 0:
        h = h % 8 
        if h % 2 != 0: 
            h1 = h // 2
            h2 = h1 + 1
        else:
            h2 = h1 = h // 2
    if w % 8 != 0:
        w = w % 8
        if w % 2 != 0:
            w1 = w // 2
            w2 = w1 + 1
        else:
            w2 = w1 = w // 2

    fit_img = image[h1:-h2, w1:-w2]
    return fit_img

def predict(img, model, preprocess, postprocess, device):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network
    out_sigmoid = torch.sigmoid(out)  # perform softmax on outputs
    result = postprocess(out_sigmoid)  # postprocess outputs
    return result

def enhance(img):
    return reflection_enhance(img)*255

def unet(img, model): 
    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output = predict(img, model, preprocess, postprocess, device)
    output[output>0] = 1
    return output.astype(np.uint8)

def threshold(img, modality): 
    gray = np.zeros(img.shape[:2])
    if modality == "raw": 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    elif modality == "preprocess": 
        gray = img[:, :, 2]*255

    # threshold
    output = np.zeros(gray.shape)
    output[gray>194] = 1
    
    return output.astype(np.uint8)

def infer(image, model, use_enhanced=False, use_threshold=False, use_model=True, use_dilation=False):
    """
    Run inference on an image using a combination of preprocessed image, model, thresholding, or dilation
    :param: image: raw image to run inference on
    :param: model: model to use for inference (passed in so model is not loaded every time)
    :param: use_enhanced: whether to use image preprocessing
    :param: use_threshold: whether to use thresholding
    :param: use_model: whether to use model
    :param: use_dilation: whether to use dilation
    :return: output: uint8 binary image of predictions
    """

    if use_model==False and use_threshold==False: 
        assert "You must use either the model or thresholding to get an output"

    # check image dimensions to see if it's divisible by 8, if not then cut either side to make it so
    img = fit_image(image)

    output1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    output2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if use_enhanced:
        img = reflection_enhance(img)

    if use_model:
        output1 = unet(img, model)

    if use_threshold:
        if use_enhanced:
            output2 = threshold(img, "preprocess")
        else:
            output2 = threshold(img, "raw")

    output = np.bitwise_or(output1, output2)
    
    if use_dilation: 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        output = cv2.dilate(output,kernel,iterations = 1)
        output[output>0] = 1
        output.astype(np.uint8)
    
    return output


    

    