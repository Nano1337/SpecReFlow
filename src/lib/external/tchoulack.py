'''
code is from Tchoulack et al. as cited in the paper
'''

import cv2
import numpy as np
import warnings

def derive_graym(impath):
    ''' The intensity value m is calculated as (r+g+b)/3, yet 
        grayscalse will do same operation!
        opencv uses default formula Y = 0.299 R + 0.587 G + 0.114 B
    ''' 
    return cv2.imread(impath, cv2.IMREAD_GRAYSCALE)


def derive_m(img, rimg):
    ''' Derive m (intensity) based on paper formula '''
    
    (rw, cl, ch) = img.shape
    for r in range(rw):
        for c in range(cl):
            rimg[r,c] = int(np.sum(img[r,c])/3.0)
            
    return rimg

def derive_saturation(img, rimg):
    ''' Derive staturation value for a pixel based on paper formula '''

    s_img = np.array(rimg)
    (r, c) = s_img.shape
    for ri in range(r):
        for ci in range(c):
            #opencv ==> b,g,r order
            s1 = (img[ri,ci][0]) + (img[ri,ci][2])
            s2 = 2 * (img[ri,ci][1])
            if  s1 >=  s2:
                s_img[ri,ci] = 1.5*((img[ri,ci][2]) - (rimg[ri,ci]))
            else:
                s_img[ri,ci] = 1.5*((rimg[ri,ci]) - (img[ri,ci][0]))

    return s_img

def check_pixel_specularity(mimg, simg):
    ''' Check whether a pixel is part of specular region or not'''

    m_max = np.max(mimg) * 0.5
    s_max = np.max(simg) * 0.33

    (rw, cl) = simg.shape

    spec_mask = np.zeros((rw,cl), dtype=np.uint8)
    for r in range(rw):
        for c in range(cl):
            if mimg[r,c] >= m_max and simg[r,c] <= s_max:
                spec_mask[r,c] = 255
    
    return spec_mask

def suppress_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

@suppress_warnings
def get_tchoulack(impath, img):
    ''' Get specular region mask using Tchoulack et al. method '''

    gray_img = derive_graym(impath)
    r_img = np.array(gray_img)
    rimg = derive_m(img, r_img)
    s_img = derive_saturation(img, rimg)
    spec_mask = check_pixel_specularity(rimg, s_img)

    return spec_mask
