we used PyTorch 1.12.1+cu113 on Ubuntu 18.04 with Anaconda Python 3.6. 

1. [Optional but recommended] Create a new Conda environment
    ~~~
    conda create --name SpecFlow python=3.6
    ~~~
    
    and activate the environment. 

    ~~~
    conda activate SpecFlow
    ~~~

2. Install PyTorch (1.12.1) with compute platform CUDA (11.3) following the instructions on the [PyTorch website](https://pytorch.org/).

    ~~~
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ~~~

3. Install other dependencies
    
    ~~~
    pip install -r requirements.txt
    ~~~

