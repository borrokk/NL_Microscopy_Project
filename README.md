# Nonlinear_microscopy_dl

Steps to always run on GPU
1. ensure that memory alloc code is added to each file

2. needs- to be added for cuda packages needed for driver


export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" 

3. 
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))" 

4. run before every training and evaluation
cd ~/projects/NL_Project
source .venv_tf/bin/activate

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

or do the nano bashrc thing as well

if still showing 0 gpu do this

1. echo $LD_LIBRARY_PATH
2. find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" \
-type f \( -name "libcudnn.so*" -o -name "libcublas.so*" -o -name "libcudart.so*" -o -name "libcusolver.so*" -o -name "libcusparse.so*" \)
3. export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd:)
4. echo $LD_LIBRARY_PATH
5. python -c "import tensorflow as tf; print('TF:', tf.__version__); print('CUDA:', tf.test.is_built_with_cuda()); print('GPU:', tf.config.list_physical_devices('GPU'))" 
should show gpu 

nvjit link fails always -fix 
python -m pip install --upgrade nvidia-nvjitlink-cu12
python -m pip show nvidia-nvjitlink-cu12

order to run - initially
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