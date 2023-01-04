# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation and training, you will need to setup the dataset. 

## CVC-EndoSceneStill dataset
- First, navigate to the `data` folder:
- Download the dataset by running the following commands:
```
pip install gdown
gdown 14DI-64d3PgwHxdt0JAIncI1e5OWEbxKC -O combined_set.zip
unzip combined_set.zip
gdown 1-1Odr6PfR7Zt6uwJZs6b758tH12WR_f0
unzip CVC_184.zip
gdown 1baxWDbRVxAz0eNJUci5YRF5Uyyt-7wEF
unzip generatedMasks184.zip

gdown 12mmojapoxxLI02yaqwZpqvc-0DVkpf6p
gdown 1eXO9k59fFPy9OqWe6X2I9ld5Po_4YGgk
```

Optionally, you may remove the zip files to save space by running:
```
rm combined_set.zip
rm CVC_184.zip
rm generatedMasks184.zip
```

Note: these datasets are downloaded from the author's personal public Google Drive endpoint. If you have any issues downloading the dataset, please contact the author.

## GLENDA dataset
To reproduce restoration experiments, please download the following dataset. Note that this zip file is quite large (1.78GB). There are also .ipynb_checkpoints folders that must be removed from GLENDA_set_2_final/fgt_output and GLENDA_set_10_final/fgvc_output for the code to run.

```
gdown 1-5oNACHxzapHaN-FWIO1uUm_VuOC3yVM
unzip GLENDA_set_all.zip
rm  GLENDA_set_all.zip
cd content
cd GLENDA_set_all/GLENDA_set_2_final/fgt_output
rm -rf .ipynb_checkpoints
cd ..
cd ..
cd GLENDA_set_10_final/fgvc_output
rm -rf .ipynb_checkpoints
cd ..
cd ..
mv GLENDA_set_all ..
cd ..
rmdir content
```
This processed GLENDA dataset folder contains 13 trials of 100 images each as described in the paper. Each trial is stored in a separate folder. Each folder structure is as follows: 

```
.
    ├── ...
    ├── GLENDA_set_1_final               # Trial number
    │   ├── GLENDA_set_1_gt              # Ground Truth Images
    │   ├── GLENDA_set_1_mask            # Random Binary Masks
    │   ├── GLENDA_set_1_img             # Masks imposed onto images
    │   ├── deepfill_output              # Restoration by Deepfillv2
    │   ├── LaMa_output                  # Restoration by LaMa
    │   ├── fgt_output                   # Restoration by FGT
    │   ├── e2fgvc_output                # Restoration by E2FGVI
    │   └── fgvc_output                  # Restoration by FGVC                
    └── ...
```

We chose to only provide restoration outputs for each restoration method in this repository. The code for each restoration method is available online and only requires the input images and masks to produce the restoration outputs. 