# FIXME: end to end demo script is still work in progress
# TODO: update SETUP.md with how to run requirements.txt from FGT
# TODO: update SETUP.md with how to clone FGT and alter requirements.txt (easiest fix is to just incorporate FGT's requirements.txt into existing one)

# file processing libraries
import os
import cv2
import numpy as np
import argparse
import subprocess
from lib.utils.image import reflection_enhance
from lib.utils.image import preprocess, postprocess

# deep learning libraries
import torch
from lib.models.unet import UNet

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network
    out_sigmoid = torch.sigmoid(out)  # perform softmax on outputs
    result = postprocess(out_sigmoid)  # postprocess outputs
    return result

def get_model():
    # model
    model = UNet(in_channels=3,
                out_channels=1,
                n_blocks=4,
                start_filters=8,
                activation='relu',
                normalization='batch',
                conv_mode='same',
                dim=2).to(device)

    model_name = 'data/unetv5.pt'
    model_weights = model_name # change path to model weights
    model.load_state_dict(torch.load(model_weights))

    model = torch.jit.script(model)

    return model

def create_detection_frame(args, img=None, model=get_model(), img_path=None, save_frames=False, count=0):
    
    if img_path: 
        img = cv2.imread(img_path)
    
    img = pad_image(img)
    
    # save original image
    if save_frames:
        cv2.imwrite(os.path.join("tempStorage", "frames", str(count).zfill(5) + '.png'), img)

    # create copy of image
    copy = img.copy()

    # calculate enhanced image
    enhanced = reflection_enhance(copy)

    # DL model prediction 
    output1 = predict(enhanced, model, preprocess, postprocess, device)
    output1[output1!=0] = 1

    # get grayscale from enhanced image V channel of HSV
    gray = enhanced[:, :, 2]

    # making mask
    output2 = np.zeros((img.shape[0], img.shape[1]))
    output2[gray>194] = 1

    # combine masks
    output = output1 + output2
    output[output == 2] = 1

    # mask dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    output = cv2.dilate(output,kernel, iterations=1)
    output[output>0] = 255

    # save output
    if save_frames:
        cv2.imwrite(os.path.join("tempStorage", "masks", str(count).zfill(5) + '.png'), output)

def pad_image(img, pad_size=8):

    # get dimensions
    height, width, channels = img.shape

    # calculate padding
    pad_height = pad_size - (height % pad_size)
    pad_width = pad_size - (width % pad_size)

    # pad image
    img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img

def main(args):
    # create temp folder for video frames and detection output frames
    if not os.path.exists(os.path.join('tempStorage', 'frames')):
        os.mkdir(os.path.join('tempStorage', 'frames'))

    if not os.path.exists(os.path.join('tempStorage', 'masks')):
        os.mkdir(os.path.join('tempStorage', 'masks'))

    print("Begin detection on video: " + args.video_path)

    # read in all frames of a video and save to folder
    cap = cv2.VideoCapture(args.video_path)
    success, img = cap.read()
    
    padded = pad_image(img)

    count = 0
    print("Is video read in successfully? " + str(success))
    while success:
        create_detection_frame(args, img, save_frames=True, count=count)
        count += 1
        success, img = cap.read()

    print("Detection complete. Restoring video...")

    # FIXME: figure out how relative paths work with subprocesses
    subprocess.call(['python3', 'FGT/tool/video_inpainting.py', 
                    '--imgH', str(padded.shape[0]), 
                    '--imgW', str(padded.shape[1]),
                    '--path', 'tempStorage/frames',
                    '--path_mask', 'tempStorage/masks',
                    '--outroot', os.path.dirname(args.video_path),
    ])

    # TODO: delete tempStorage subfolders after video restoration




if __name__ == '__main__':

    # take in command line arguments for input video path
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=r'data/demo/demo_vid.mp4', help='path to input video')
    args = parser.parse_args()
    main(args)
        





