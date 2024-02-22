# embedded-ai-tutorial


## Build container_mmlab image
Note: container_mmlab is the root folder.
```
docker build -t container_mmlab:build .

docker run --name mmlab_c -it --runtime nvidia container_mmlab:build
```
```
conda activate mmpose

python3 -c "import torch;print(f'CUDA IS FOUND:{torch.cuda.is_available()}')"
mim install -v "mmcv>=2.0.1,<2.1.0"
mim install -v "mmdet>=3.1.0,<4.0.0"

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose && pip install -r requirements.txt && pip install -v -e .
```
```
conda activate mmyolo

python3 -c "import torch;print(f'CUDA IS FOUND:{torch.cuda.is_available()}')"
mim install -v "mmcv==2.0.0rc4"
mim install -v "mmdet>=3.0.0,<4.0.0"

git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo && pip install -r requirements.txt && pip install -v -e .
```
```
conda activate mmdeploy

cd /mmlab/ppl.cv
./build.sh cuda

# ONNX
cd /mmlab
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-aarch64-1.16.0.tgz
tar -zxvf onnxruntime-linux-aarch64-1.16.0.tgz
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

cd /mmlab/mmdeploy
# build TensorRT custom operators
mkdir -p build && cd build
cmake .. -DMMDEPLOY_TARGET_BACKENDS="trt"
make -j$(nproc) && make install

# install model converter
cd /mmlab/mmdeploy
pip install -v -e .

# Build SDK Libraries and its demo
mkdir -p build && cd build
cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="trt;ort" \
    -DMMDEPLOY_CODEBASES=all \
    -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
    -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl
make -j$(nproc) && make install

# For demo/Testing
cd /mmlab/mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```
```
exit

docker commit mmlab_c container_mmlab:latest

docker save container_mmlab:latest -o container_mmlab.tar
```