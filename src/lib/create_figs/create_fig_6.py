import cv2
import numpy as np
from inference import enhance, unet, threshold
from models.unet import get_premade_unet

def make_fig_6():
    img_path = "../../figs/fig_6/original_img.png"
    img = cv2.imread(img_path)
    model = get_premade_unet()

    # (a)(1) raw + Light U-net
    a1 = img.copy()
    a1_output = unet(a1, model)
    a1[a1_output>0] = (0, 255, 0)
    cv2.imwrite("../../figs/fig_6/a1.png", a1)

    # (a)(2) preprocessed + Light U-net
    a2 = img.copy()
    a2_output = unet(enhance(a2.copy()), model)
    a2[a2_output>0] = (0, 255, 0)
    cv2.imwrite("../../figs/fig_6/a2.png", a2)

    # (b)(1) raw + thresholding
    b1 = img.copy()
    b1_output = threshold(b1.copy(), "raw")
    b1[b1_output>0] = (255, 0, 0)
    cv2.imwrite("../../figs/fig_6/b1.png", b1)

    # (b)(2) preprocessed + thresholding
    b2 = img.copy()
    b2_output = threshold(enhance(b2.copy())/255.0, "preprocess")
    b2[b2_output>0] = (255, 0, 0)
    cv2.imwrite("../../figs/fig_6/b2.png", b2)

    # (c)(1) raw + Light U-net + thresholding
    c1 = img.copy()
    c1_intersect = np.bitwise_and(a1_output, b1_output)
    c1[a1_output>0] = (0, 255, 0)
    c1[b1_output>0] = (255, 0, 0)
    c1[c1_intersect>0] = (255, 0, 255)
    cv2.imwrite("../../figs/fig_6/c1.png", c1)

    # (c)(2) preprocessed + Light U-net + thresholding
    c2 = img.copy()
    c2_intersect = np.bitwise_and(a2_output, b2_output)
    c2[a2_output>0] = (0, 255, 0)
    c2[b2_output>0] = (255, 0, 0)
    c2[c2_intersect>0] = (255, 0, 255)
    cv2.imwrite("../../figs/fig_6/c2.png", c2)



