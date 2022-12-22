we used PyTorch 1.12.1+cu113 on Ubuntu 18.04 with Anaconda Python 3.6. 

1. [Optional but highly recommended] Create a new Conda environment
    ~~~
    conda create --name SpecFlow python=3.7.15
    ~~~
    
    and activate the environment. 

    ~~~
    conda activate SpecFlow
    ~~~

2. Install PyTorch (1.12.1) with compute platform CUDA (11.3) following the instructions on the [PyTorch website](https://pytorch.org/).

    ~~~
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ~~~

3. Install other dependencies
    
    ~~~
    pip install -r requirements.txt
    ~~~

