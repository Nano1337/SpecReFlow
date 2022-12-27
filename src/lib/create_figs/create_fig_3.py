import torchsummary 
import cv2
import os
from models.unet import get_premade_unet
import contextlib
from inference import infer

def make_torch_summary(output_dir, model):
    """
    Get Unet architecture information for figure 3
    """
    
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        with contextlib.redirect_stdout(f):
            torchsummary.summary(model, (3, 288, 384))

def make_example_mask(output_dir, img, model):
    """
    Get example mask for figure 3
    """
    cv2.imwrite(os.path.join(output_dir, "input.png"), img)

    mask = infer(img, model)
    cv2.imwrite(os.path.join(output_dir, "output_mask.png"), mask*255)
    

def make_fig_3():
    output_dir = "../../figs/fig_3/"
    img_path = "../data/CVC-all/img/20.bmp"
    img = cv2.imread(img_path)
    model = get_premade_unet()
    make_torch_summary(output_dir, model)
    make_example_mask(output_dir, img, model)

if __name__ == "__main__":
    make_fig_3()