# Nonlinear_microscopy_dl

Order to run - initially
1. first find python installed and create a venv - python3.12 -m venv .venv
2. source .venv/bin/activate
3. python -m pip install --upgrade pip setuptools wheel = to upgrade pip
4. pip install -r requirements.txt - to install all packages
5. check nvidia-smi
6. check this - python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))" - should show active gpu
7. run the training file - python src/train_v2_torch.py
8. python src/train_v2_torch.py --eval  to just see the metrics of all 3 seeds
9. results stored in "results" / "v2_torch"


train with src/train_v2_torch.py - 
all results stored in "results" / "v2_torch"
Steps to always run on GPU
1. ensure that memory alloc code is added to each file

2. needs- to be added for cuda packages needed for driver

export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" 

3. python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))" 

4. run before every training and evaluation
cd ~/projects/NL_Project
source .venv_tf/bin/activate

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

If still showing 0 gpu, update the cuda and torch packages so that they are compatible

nvjit link fails always -fix 
python -m pip install --upgrade nvidia-nvjitlink-cu12
python -m pip show nvidia-nvjitlink-cu12

