#!/bin/bash
#docker run -it --rm --name dlwpt --gpus '"device=7"' -p 8080:8888 -v `pwd`:/workspace nvcr.io/nvidia/pytorch:22.08-py3 /bin/bash
docker run -it --rm --name torch --shm-size=10240m --gpus '"device=6,7"' -p 8080:8888 -p 6006:6006 -v `pwd`:/workspace nvcr.io/nvidia/pytorch:22.08-py3  bash -c 'source /usr/local/nvm/nvm.sh; jupyter labextension install @jupyterlab/toc; pip install torchinf
; jupyter-notebook'