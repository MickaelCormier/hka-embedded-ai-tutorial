# embedded-ai-tutorial


## Build container_mmlab image
Note: container_mmlab is the root folder.
```
docker build -t container_mmlab:build .

docker run --name mmlab_c -it --runtime nvidia container_mmlab:build


conda activate mmpose

python3 -c "import torch;print(f'CUDA IS FOUND:{torch.cuda.is_available()}')"
mim install -v "mmcv>=2.0.1,<2.1.0"
mim install -v "mmdet>=3.1.0,<4.0.0"

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose && pip install -r requirements.txt && pip install -v -e .


conda activate mmyolo

python3 -c "import torch;print(f'CUDA IS FOUND:{torch.cuda.is_available()}')"
mim install -v "mmcv==2.0.0rc4"
mim install -v "mmdet>=3.0.0,<4.0.0"

git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo && pip install -r requirements.txt && pip install -v -e .


conda activate mmdeploy

... TODO

exit

docker commit mmlab_c container_mmlab:latest

docker save container_mmlab:latest -o container_mmlab.tar
```