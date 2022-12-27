# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation and training, you will need to setup the dataset. 

## CVC-EndoSceneStill
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