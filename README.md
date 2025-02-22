python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
https://dlib.net/


pornstar database: https://www.thumbnailseries.com/pornstars/



# Not tried yet
# Best install dlib with conda: https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781

install dlib with conda:
Installing dlib using conda with CUDA enabled
Prerequisite: conda and/or miniconda are already installed

Create a conda environment.
$ conda create -n dlib python=3.8 cmake ipython
Activate the environment.
$ conda activate dlib
Install CUDA and cuDNN with conda using nvidia channel
$ conda install cuda cudnn -c nvidia

This may help if not working alone:
$conda install -c conda-forge cudatoolkit=11.8 cudnn=8.8

$which nvcc
/path/to/your/miniconda3/envs/dlib/bin/
Install dlib. Clone and build dlib from source
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build
$ cd build
$ cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 
$ cmake --build .
$ cd ..
$ python setup.py install --set DLIB_USE_CUDA=1
Test dlib
(dlib) $ ipython
Python 3.8.12 (default, Oct 12 2021, 13:49:34)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.27.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import dlib

In [2]: dlib.DLIB_USE_CUDA
Out[2]: True

In [3]: print(dlib.cuda.get_num_devices())
1
