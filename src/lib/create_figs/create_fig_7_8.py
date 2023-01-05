import os
from utils.restoration_metrics import mPSNR, mSSIM, origSSIM, origPSNR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import contextlib

import scipy.stats as stats

def get_img_paths(img_dir, mask_dir): 
    # get all images in folder
    img_names = os.listdir(img_dir)
    img_names = [os.path.join(img_dir, i) for i in img_names]
    img_names.sort()
    img_names = img_names[50:]

    # get all masks in folder
    mask_names = os.listdir(mask_dir)
    mask_names = [os.path.join(mask_dir, i) for i in mask_names]
    mask_names.sort()
    mask_names = mask_names[50:]
    
    return img_names, mask_names

def get_restored_paths(restored_dir):
    # get all restored images in folder
    restored_names = os.listdir(restored_dir)
    restored_names = [os.path.join(restored_dir, i) for i in restored_names]
    restored_names.sort()
    if "img" in restored_dir: 
        restored_names = restored_names[50:]
    return restored_names

def calc_and_add(img_paths, restored_paths, mask_paths, global_df, row_name):
    """
    Calculate metrics and add to dataframe
    """
    # create empty dataframe 
    df = pd.DataFrame(columns=['Method', 'Metric', 'Average', 'All 50 values']) 

    # set first row, first column to row_name
    df.loc[0, 'Method'] = row_name

    ssims, psnrs, mssims, mpsnrs = [], [], [], []

    # calculate metrics
    for i in range(len(img_paths)):

        # read in images
        img1 = cv2.imread(img_paths[i])
        img2 = cv2.imread(restored_paths[i])
        mask = cv2.imread(mask_paths[i])

        # calculate metrics
        ssims.append(origSSIM(img1, img2, channel_axis=2, gaussian_weights=True, sigma=1.5, win_size=7, use_sample_covariance=False, data_range=255))
        psnrs.append(origPSNR(img1, img2))
        mssims.append(mSSIM(img1, img2, mask))
        mpsnrs.append(mPSNR(img1, img2, mask))

    # calculate averages of metrics
    ssim_avg = sum(ssims)/len(ssims)
    psnr_avg = sum(psnrs)/len(psnrs)
    mssim_avg = sum(mssims)/len(mssims)
    mpsnr_avg = sum(mpsnrs)/len(mpsnrs)

    # add metrics to dataframe
    df.loc[1, 'Metric'] = 'SSIM'
    df.loc[1, 'Average'] = ssim_avg
    df.loc[1, 'All 50 values'] = ssims

    df.loc[2, 'Metric'] = 'PSNR'
    df.loc[2, 'Average'] = psnr_avg
    df.loc[2, 'All 50 values'] = psnrs

    df.loc[3, 'Metric'] = 'mSSIM'
    df.loc[3, 'Average'] = mssim_avg
    df.loc[3, 'All 50 values'] = mssims

    df.loc[4, 'Metric'] = 'mPSNR'
    df.loc[4, 'Average'] = mpsnr_avg
    df.loc[4, 'All 50 values'] = mpsnrs
    
    # append to global dataframe
    global_df = pd.concat([global_df, df], ignore_index=True, axis=0)

    return global_df

def make_restoration_metrics():
    # create empty dataframe 
    df = pd.DataFrame(columns=['Method', 'Metric', 'Average', 'All 50 values']) 
    
    row_names = ["SR", "FGT", "FGVC", "E2FGVI", "DeepFillv2", "LaMa"]
    output_path = "../data/restoration_metrics/"

    root = '../data/GLENDA_set_all/'

    for count in range(10, 14): # TODO change first index back to 1
        print("Working on GLENDA_set_{}".format(count))

        trial_dir = os.path.join(root, 'GLENDA_set_' + str(count) + "_final")
        gt_path = os.path.join(trial_dir, "GLENDA_set_" + str(count) + "_gt")
        mask_path = os.path.join(trial_dir, "GLENDA_set_" + str(count) + "_mask")
        gt_paths, mask_paths = get_img_paths(gt_path, mask_path)

        output_names = ["GLENDA_set_" + str(count) + "_img", "fgt_output", "fgvc_output", "e2fgvc_output", "deepfill_output", "LaMa_output"]
        folder_paths = [os.path.join(trial_dir, i) for i in output_names]

        for i in range(len(folder_paths)):
            df = calc_and_add(gt_paths, get_restored_paths(folder_paths[i]), mask_paths, df, row_names[i])

        df.to_csv(os.path.join(output_path, 'GLENDA_set_{}.csv'.format(count)), index=False)

        print("Finished GLENDA_set_{}".format(count))

def sigtest_helper(avgs, metric, algos, f_a, s_a, f_b, s_b):
    a, b = [], []
    num_cat = 4
    for i in range(13):
        a.append(avgs[i][metric[f_a] + algos[s_a]*num_cat])
        b.append(avgs[i][metric[f_b] + algos[s_b]*num_cat])

    t, p = stats.ttest_rel(a, b)
    
    print("Metric: {}, Algo1: {}, Algo2: {}, p-value: {}".format(f_a, s_a, s_b, p))
 
def create_significance_testing(csv_paths, output_path):
    dfs, avgs = [], []

    # read from csv files
    for i in range(len(csv_paths)):
        dfs.append(pd.read_csv(csv_paths[i]))

        # get column of averages for each metric and remove NaN values and convert to numpy
        avgs.append(dfs[i].loc[:, 'Average'])
        avgs[i] = avgs[i].dropna()
        avgs[i] = avgs[i].to_numpy()

    # maps for interpretability reasons
    metric = {
        "SSIM": 0,
        "PSNR": 1,
        "mSSIM": 2,
        "mPSNR": 3
    }

    algos = {
        "SR": 0,
        "FGT": 1,
        "FGVC": 2,
        "E2FGVI": 3,
        "DeepFillv2": 4,
        "LaMa": 5
    }

    metric_names = ['mSSIM', 'mPSNR']
    flow_based = ['FGT', 'FGVC', 'E2FGVI']
    inpaint_based = ['DeepFillv2', 'LaMa']

    print("Begin significance testing")
    with open('../../figs/fig_7_8/restoration_metrics_significance_testing.txt', 'w') as f:
        with contextlib.redirect_stdout(f): # redirect print statements to text file
            for i in metric_names:
                for j in flow_based:
                    for k in inpaint_based:
                        sigtest_helper(avgs, metric, algos, i, j, i, k)
    print("End significance testing")

# function to add value labels
def addlabels(x,y, ax):
    for i in range(len(x)):
      label = "{:.4f}".format(y[i])
      ax.text(x[i], y[i]/2, label, ha = 'center')

def plot_fig(x_pos, means, stds, cat_var, output_path, title):
    #Build the plot
    fig, ax = plt.subplots()
    print(title)
    print(means)
    print(stds)
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(title + ' metric')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_var)
    ax.set_title("Quantitative Restoration Comparison using " + title)
    ax.yaxis.grid(True)
    addlabels(x_pos, means, ax)

    # Save the figure 
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig_7_8_{}.png'.format(title)))

def create_figures(csv_paths, output_path):
    """
    Takes in a list of csv paths and creates figures for each metric
    """
    dfs, avgs = [], []

    # read from csv files
    for i in range(len(csv_paths)):
        dfs.append(pd.read_csv(csv_paths[i]))

        # get column of averages for each metric and remove NaN values and convert to numpy
        avgs.append(dfs[i].loc[:, 'Average'])
        avgs[i] = avgs[i].dropna()
        avgs[i] = avgs[i].to_numpy()

    # maps for interpretability reasons
    metric = {
        "SSIM": 0,
        "PSNR": 1,
        "mSSIM": 2,
        "mPSNR": 3
    }

    algos = {
        "SR": 0,
        "FGT": 1,
        "FGVC": 2,
        "E2FGVI": 3,
        "DeepFillv2": 4,
        "LaMa": 5
    }

    metric_names = ['SSIM', 'PSNR', 'mSSIM', 'mPSNR']

    cat_var = ["SR", "FGT", "FGVC", "E2FGVI", "DeepFillv2", "LaMa"]
    x_pos = np.arange(len(cat_var))

    a = []
    means, stds = [], []
    for f_a in metric_names:
        for s_a in cat_var:
            for i in range(13):
                a.append(avgs[i][metric[f_a] + algos[s_a]*4])
            means.append(np.mean(a))
            stds.append(np.std(a))
            a = []

        plot_fig(x_pos, means, stds, cat_var, output_path, f_a)

        means, stds = [], []



def make_fig_7_8():

    # create csv files - takes a while
    # make_restoration_metrics()

    output_path = "../../figs/fig_7_8/"
    csv_paths = "../data/restoration_metrics/"
    csv_paths = [os.path.join(csv_paths, i) for i in os.listdir(csv_paths)]
    
    # # conduct significance testing
    # create_significance_testing(csv_paths, output_path)

    # create figures from data
    create_figures(csv_paths, output_path)
