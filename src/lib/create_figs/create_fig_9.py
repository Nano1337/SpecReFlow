import os 
import cv2

def crop_a(img): 
    return img[130:255, 170:260]

def crop_b(img):
    return img[175:260, 285:355]

def crop_c(img):
    return img[250:355, 85:200]

def crop_d(img): 
    return img[200:-30, 470:-10]

def make_fig_9(): 
    output_path = r"../../figs/fig_9"
    data_root = r'../data/GLENDA_set_all/'
    # make rows of the image
    output_folders = ['a', 'b', 'c', 'd']
    if not os.path.exists(os.path.join(output_path, output_folders[0])):
        for folder in output_folders:
            os.makedirs(os.path.join(output_path, folder))

    sources = [['2', "00064.png"], ['5', "00090.png"], ['13', '00075.png'], ['11', '00051.png']]

    for i in range(4): 
    
        # cropping function
        crop_fn = eval(f"crop_{output_folders[i]}") 

        # cropping image with ground truth 
        GT_img = cv2.imread(os.path.join(data_root, f"GLENDA_set_{sources[i][0]}_final", f"GLENDA_set_{sources[i][0]}_gt", sources[i][1])) 
        cv2.imwrite(os.path.join(output_path, output_folders[i], "GT.png"), crop_fn(GT_img))

        # cropping image with artificial specular mask
        SR_img = cv2.imread(os.path.join(data_root, f"GLENDA_set_{sources[i][0]}_final", f"GLENDA_set_{sources[i][0]}_img", sources[i][1])) 
        cv2.imwrite(os.path.join(output_path, output_folders[i], "SR.png"), crop_fn(SR_img))

        # cropping image with fgt_output
        fgt_img = cv2.imread(os.path.join(data_root, f"GLENDA_set_{sources[i][0]}_final", "fgt_output", sources[i][1])) 
        cv2.imwrite(os.path.join(output_path, output_folders[i], "fgt.png"), crop_fn(fgt_img))

        # cropping image with fgvc_output
        fgvc_img = cv2.imread(os.path.join(data_root, f"GLENDA_set_{sources[i][0]}_final", "fgvc_output", sources[i][1]))
        cv2.imwrite(os.path.join(output_path, output_folders[i], "fgvc.png"), crop_fn(fgvc_img))

        # cropping image with e2fgvc_output
        e2fgvc_img = cv2.imread(os.path.join(data_root, f"GLENDA_set_{sources[i][0]}_final", "e2fgvc_output", sources[i][1]))
        cv2.imwrite(os.path.join(output_path, output_folders[i], "e2fgvc.png"), crop_fn(e2fgvc_img))

        # cropping image with deepfill_output
        deepfill_img = cv2.imread(os.path.join(data_root, f"GLENDA_set_{sources[i][0]}_final", "deepfill_output", sources[i][1]))
        cv2.imwrite(os.path.join(output_path, output_folders[i], "deepfill.png"), crop_fn(deepfill_img))

        # cropping image with LaMa_output
        LaMa_img = cv2.imread(os.path.join(data_root, f"GLENDA_set_{sources[i][0]}_final", "LaMa_output", sources[i][1]))
        cv2.imwrite(os.path.join(output_path, output_folders[i], "LaMa.png"), crop_fn(LaMa_img))

