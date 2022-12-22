
# file processing libraries
import os
import cv2
import numpy as np
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

    model_name = 'unetv5.pt'
    model_weights = model_name # change path to model weights
    model.load_state_dict(torch.load(model_weights))

    model = torch.jit.script(model)

    return model

def create_detection_frame(img=None, model=get_model(), img_path=None):
    
    if img_path: 
        img = cv2.imread(img_path)
    
    img = pad_image(img)
    copy = img.copy()

    # calculate enhanced image
    enhanced = reflection_enhance(copy) # this is messed up

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
    output[output>0] = 1

    # apply mask to original image
    copy[output==1] = [0, 255, 0]

    return img, output, copy

# write a method that will take an input image and pad it have image dimensions divisible by 16
def pad_image(img, pad_size=8):

    # get dimensions
    height, width, channels = img.shape

    # calculate padding
    pad_height = pad_size - (height % pad_size)
    pad_width = pad_size - (width % pad_size)

    # pad image
    img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img

def create_detection_video(video_path, output_path, save_frames=False): 

    # image and mask output directories
    img_dir = r'C:\Users\haoli\OneDrive\Documents\SpecFLow\output\images'
    mask_dir = r'C:\Users\haoli\OneDrive\Documents\SpecFLow\output\masks'

    # iterate through all frames of video without saving individual frames
    cap = cv2.VideoCapture(video_path)
    success, img = cap.read()
    count = 0

    # declare and initialize model
    print("Warming up model...")
    model = get_model()

    # create video writer for output
    img = pad_image(img)
    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc('m','p','4','v'), 30, (img.shape[1],img.shape[0]))
    print("Beginning video processing...")

    while success:
        print("Processing frame " + str(count))

        # run detection
        img, output, copy = create_detection_frame(img=img, model=model)

        # write to video
        out.write(copy)
        if save_frames:
            cv2.imwrite(os.path.join(img_dir, str(count).zfill(5) + '.png'), img)
            cv2.imwrite(os.path.join(mask_dir, str(count).zfill(5) + '.png'), output)

        count += 1
        success, img = cap.read()

    print("Video processing complete.")
    cap.release()
    out.release()

if __name__ == '__main__':
    # video_path = 'D:\\Videos\\vid_raw.mp4'
    # output_path = 'D:\\Videos\\vid_detected.mp4'
    # create_detection_video(video_path, output_path, save_frames=True)

    # test_image = '53.bmp'
    # test_image = cv2.imread(test_image)
    # output = create_detection_frame(img=test_image)
    # cv2.imwrite('output.png', output)

    # read in all frames of a video and store in numpy array
    cap = cv2.VideoCapture('D:\\Videos\\vid_detected.mp4')
    success, img = cap.read()
    count = 0
    frames = []
    while success:
        print("Working on frame " + str(count))
        img = cv2.resize(img, (640, 496), interpolation=cv2.INTER_AREA)
        frames.append(img)
        count += 1
        success, img = cap.read()

    # convert to numpy array
    frames = np.array(frames)

    # save to file
    np.save('vid_detected.npy', frames)

    # read in numpy array and use cv2 to convert from BGR to RGB
    frames = np.load('vid_detected.npy')
    for i in range(frames.shape[0]):
        frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        




