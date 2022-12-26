import cv2
import os
import numpy as np

def make_rgb(output_dir, read):
    """
    Make separate RGB images from the original image and save to output_dir
    :param: output_dir: directory to save the images
    :param: read: original image
    """
    
    # blue 
    read_b = read.copy()
    read_b[:, :, 1] = 0
    read_b[:, :, 2] = 0
    cv2.imwrite(os.path.join(output_dir, "blue.png"), read_b)

    # green 
    read_g = read.copy()
    read_g[:, :, 0] = 0
    read_g[:, :, 2] = 0
    cv2.imwrite(os.path.join(output_dir, "green.png"), read_g)

    # red
    read_r = read.copy()
    read_r[:, :, 0] = 0
    read_r[:, :, 1] = 0
    cv2.imwrite(os.path.join(output_dir, "red.png"), read_r)

    return read_b, read_g, read_r

def make_hsv(output_dir, read):
    """
    Make separate HSV images from the original image and save to output_dir
    :param: output_dir: directory to save the images
    :param: read: original image
    """

    read = cv2.cvtColor(read, cv2.COLOR_BGR2HSV)

    # hue 
    hue = read.copy()
    hue = hue[:, :, 0]
    cv2.imwrite(os.path.join(output_dir, "hue.png"), hue)

    # saturation 
    saturation = read.copy()
    saturation = saturation[:, :, 1]
    cv2.imwrite(os.path.join(output_dir, "saturation.png"), saturation)

    # value
    value = read.copy()
    value = value[:, :, 2]
    cv2.imwrite(os.path.join(output_dir, "value.png"), value)

    return hue, saturation, value

def norm_and_return(img):
    return np.float32(cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX))
    
def make_inverse_figs(output_dir, b, g, r, s):
    """
    Make inverse and enhanced images from the original image and save to output_dir
    :param: output_dir: directory to save the images
    :param: b: blue channel
    :param: g: green channel
    :param: r: red channel
    :param: s: saturation channel
    """

    norm_saturation = norm_and_return(s)
    inverse_norm_saturation = np.expand_dims((1-norm_saturation), axis=2)
    cv2.imwrite(os.path.join(output_dir, "inverse_saturation.png"), inverse_norm_saturation*255)

    norm_b = norm_and_return(b)
    norm_g = norm_and_return(g)
    norm_r = norm_and_return(r)

    inverse_norm_b = inverse_norm_saturation*norm_b
    cv2.imwrite(os.path.join(output_dir, "enhanced_blue.png"), inverse_norm_b*255) 

    inverse_norm_g = inverse_norm_saturation*norm_g
    cv2.imwrite(os.path.join(output_dir, "enhanced_green.png"), inverse_norm_g*255) 

    inverse_norm_r = inverse_norm_saturation*norm_r
    cv2.imwrite(os.path.join(output_dir, "enhanced_red.png"), inverse_norm_r*255) 

    # merge enhanced rgb channels to create enhanced rgb image
    enhanced_rgb = cv2.merge((inverse_norm_b[:, :, 0], inverse_norm_g[:, :, 1], inverse_norm_r[:, :, 2]))
    cv2.imwrite(os.path.join(output_dir, "enhanced_img.png"), enhanced_rgb*255)

def make_fig_1_2_4():
    """
    Make images for figures 1, 2, and 4
    """
    output_dir = "../../figs/fig_1_2_4/"
    img = cv2.imread(os.path.join(output_dir, "original_img.png"))
    img = cv2.resize(img, (600, 450), interpolation=cv2.INTER_AREA)

    b, g, r = make_rgb(output_dir, img)
    _, s, _ = make_hsv(output_dir, img)
    make_inverse_figs(output_dir, b, g, r, s)


if __name__ == "__main__": 

    make_fig_1_2_4()
