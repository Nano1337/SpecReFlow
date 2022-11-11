
# file processing libraries
import os
import cv2
import numpy as np
from lib.utils.image import reflection_enhance
from lib.utils.image import preprocess, postprocess
from lib.utils.video import Singleton, get_frames

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

    model_name = 'unetv5.pt'
    model_weights = os.path.join('D:', 'Unet Weights', model_name)
    model.load_state_dict(model_weights)

    model = torch.jit.script(model)

    return model

def create_detection_video(video_path, output_path): 

    # iterate through all frames of video without saving individual frames
    frames = get_frames(video_path)

    # create a singleton instance
    state = Singleton(frames[0].shape[0], frames[0].shape[1])

    for img in frames: 

        # calculate enhanced image
        enhanced = reflection_enhance(img)*255

        # declare and initialize model
        model = get_model()

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
        output = cv2.dilate(output,kernel,iterations = 1)

        # write to video        
        state.write(output)

    # save video
    state.close_and_get(output_path)

if __name__ == '__main__':
    video_path = 'D:\\Videos\\test.mp4'
    output_path = 'D:\\Videos\\test_output.mp4'
    create_detection_video(video_path, output_path)