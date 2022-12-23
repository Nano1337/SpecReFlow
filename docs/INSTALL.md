we used PyTorch 1.13.1+cu117 on Ubuntu 22.04 with Anaconda Python 3.10. 

1. [Optional but highly recommended] Create a new Conda environment
    ~~~
    conda create --name SpecFlow python=3.10
    ~~~
    
    and activate the environment. 

    ~~~
    conda activate SpecFlow
    ~~~

2. Install PyTorch (1.13.1) with compute platform CUDA (11.7) following the instructions on the [PyTorch website](https://pytorch.org/).

    ~~~
    pip install torch torchvision torchaudio
    ~~~

3. Install other dependencies
    
    ~~~
    pip install -r requirements.txt
    ~~~

