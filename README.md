# SpecFlow: A Specular Reflection Restoration Framework using Flow-Guided Video Completion
The official implementation of [SpecFlow, *SPIE Photonics West 2023*](https://spie.org/photonics-west/presentation/SpecFlow--an-end-to-end-framework-for-specular-reflection/12368-22?SSO=1)

Contact: [haoli.yin@vanderbilt.edu](mailto:haoli.yin@vanderbilt.edu). More info about me at my [personal website](https://haoliyin.me/). Feel free to reach out with any questions or discussion!

# Abstract

Specular reflections (SR) are highlight artifacts commonly found in endoscopy videos and can severely disrupt a surgeon’s observation and judgment. Although numerous methods have been proposed to remove SR regions, they are inefficient, require extensive empirical parameter selection, or inpaint the detected SR region using insufficient information from a single view, which can result in false clinical interpretations. Therefore, we propose an end-to-end deep learning pipeline termed SpecFlow to detect and restore SR regions from endoscopy videos. 

Our proposed SpecFlow consists of two phases: (1) a detection phase where the SR region is segmented to indicate where restoration is required and (2) a restoration phase where we implement a novel approach by using optical flow to guide seamless color propagation and information from other frames of the endoscopy video to restore SR in the current frame. The former phase is done using a reduced U-net architecture (termed Light U-net) with additional preprocessing enhancement and postprocessing thresholding to improve detection. The latter phase is achieved through optical flow estimation of the SR region using SPyNet and propagation of local and non-local features positionally encoded through a vision transformer model.  

Comprehensive quantitative and qualitative tests for each phase conducted on several different endoscopy modalities reveal that our SpecFlow solution is competitive with current state-of-the-art models. Our detection pipeline achieves a Dice score of 82.8% while only requiring 14ms of computational time, allowing for real-time processing and our restoration pipeline successfully incorporates temporal information for more accurate restorations. 

# Demo 
Note: This may take a few seconds to load in. 
![teaser](./figs/croppedgif.gif)

## Installation
Note: This is a work in progress, model weights and data will be made available soon. 

Please refer to [INSTALL.md](docs/INSTALL.md) for installation instructions.

