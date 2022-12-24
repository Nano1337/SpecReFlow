# ensure compatability wih python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# general libraries
import os
import torch 
import numpy as np
import argparse

# deep learning libraries
import albumentations
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import from other files
from config import cfg, update_config
from logger import Logger
from datasets.unet_sample import SegmentationDataSet
from utils.image import create_dense_target, normalize_01, ComposeDouble, AlbuSeg2d, FunctionWrapperDouble, plot_training
from models.unet import get_unet
from models.losses import DiceLoss
from trains.unet_trainer import Trainer

def parse_args():
    """
    Parse input arguments to read in config file
    return: args
    """
    parser = argparse.ArgumentParser(description='Train SR detection network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configuration file name',
                        required=True,
                        type=str)                     
    args = parser.parse_args()

    return args

def get_filenames_of_path(root, ext):
    """Returns a list of files in a directory/path. Uses pathlib."""
    temp = os.path.join(root, ext)
    filenames = [os.path.join(temp, file) for file in os.listdir(temp)]
    return filenames

def main(cfg):
    torch.manual_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    
    # set up dataset
    print('Setting up dataset...')
    inputs = get_filenames_of_path(cfg.DATASET.DATA_DIR, 'img')
    targets = get_filenames_of_path(cfg.DATASET.DATA_DIR, 'mask')

    # pre-transformations
    pre_transforms = ComposeDouble([
        FunctionWrapperDouble(resize,
                            input=True,
                            target=False,
                            output_shape=(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH, 3)),
        FunctionWrapperDouble(resize,
                            input=False,
                            target=True,
                            output_shape=(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH),
                            order=0,
                            anti_aliasing=False,
                            preserve_range=True),
    ])

    # training transformations and augmentations
    transforms_training = ComposeDouble([
        AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
        AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])

    # validation transformations
    transforms_validation = ComposeDouble([
        AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
        AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])

    # split dataset into training and validation
    train_size = cfg.DATASET.TRAIN_SIZE

    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=cfg.SEED,
        train_size=train_size,
        shuffle=True)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=cfg.SEED,
        train_size=train_size,
        shuffle=True)

    # dataset training
    dataset_train = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms_training,
                                        use_cache=True,
                                        pre_transform=pre_transforms)

    # dataset validation
    dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        transform=transforms_validation,
                                        use_cache=True,
                                        pre_transform=pre_transforms)

    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    shuffle=True)

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    shuffle=True)



    # set up model
    print('Creating model...')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_unet(cfg, device)

    # load pretrained weights
    if cfg.TRAIN.PRETRAINED_WEIGHTS:
        print('Loading pretrained weights...')
        model.load_state_dict(torch.load(cfg.TRAIN.PRETRAINED_WEIGHTS))

    # set up logger
    logger = Logger(cfg)

    # set up optimizer
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    else: 
        NotImplementedError
    
    # set up loss function
    if cfg.TRAIN.CRITERION == 'dice':
        criterion = DiceLoss()
    else:
        NotImplementedError
    
    # set up learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=cfg.TRAIN.PATIENCE,
        factor=cfg.TRAIN.DECAY_FACTOR,
        verbose=True,
    )

    # set up trainer
    print('Starting training...')
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        training_DataLoader=dataloader_training,
        validation_DataLoader=dataloader_validation,
        lr_scheduler=scheduler,
        epochs=cfg.TRAIN.NUM_EPOCHS,
        epoch=cfg.TRAIN.START_EPOCH,
        logger=logger,
    )

    training_losses, validation_losses, lr_rates = trainer.run_trainer()

    # log training visualization
    if cfg.TRAIN.VISUALIZE:
        fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10, 4))
        fig.savefig(os.path.join(logger.log_dir, 'training.png'))

    # save model
    torch.save(model.state_dict(), os.path.join(logger.log_dir, 'model.pth'))
    print('Training complete!')

if __name__ == '__main__':
    """ 
    Read in config file and begin training
    """
    args = parse_args()
    update_config(cfg, args.cfg)
    main(cfg)