export PYTHONIOENCODING=utf8
gpu=6
model=./model
CUDA_LAUNCH_BLOCKING=1 python3 -u ./gen.py $gpu $model test