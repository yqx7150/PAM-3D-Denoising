#################### 命令行
###python main.py --config=configs/subvp/cifar10_ncsnpp_continuous.py --workdir=exp --mode=train --eval_folder=result
###python main.py --config=configs/ve/cifar10_ncsnpp_continuous.py --workdir=exp --mode=train --eval_folder=result

CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train --eval_folder=result


++++++++++++
source activate sde

nvidia-smi -l


CUDA_VISIBLE_DEVICES=1 python pc_sampling_lzl_daikuan.py

CUDA_VISIBLE_DEVICES=1 python pc_sampling_lzl_daikuan.py

docker images

docker ps -a

docker run --name ncsnpplz --gpus all -v /home/lqg/桌面/LZH:/home/lzh/:rw -it ed bash

sudo chmod -R 777 /home/lqg/桌面/LZH

### 启动docker
sudo docker start 5f

docker ps

sudo docker exec -it 5f /bin/bash


conda activate ncsnpp

cd /home/lzh/sde-test-hank-aloha-new-patch


[27.798776856225246, 26.160146199287713, 29.03906433772265, 30.298273151389864, 25.41202796062786, 25.870063414980073, 25.58561123286951, 26.26363455854807, 31.912871526470372, 25.23643574180444, 27.2682402193746, 22.48033344148788, 27.599963868751164, 26.43935635490324, 23.991839308047975, 27.794249340469452, 20.454600830511435, 26.06945920074967, 22.428303140401304, 23.296893378940467, 25.626911278312008, 26.502128928038445]
psnr_ave:  26.06950837590516 0.7982944795268587 22 22






###############****************** train ******************#######################
train    (xiugai:datasets.py)
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train --eval_folder=result

CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train --eval_folder=result




test     (xiugai:controllable_generation_lzl_daikuan.py)
python pc_sampling_lzl_daikuan.py
CUDA_VISIBLE_DEVICES=1 python pc_sampling_lzl_daikuan.py
CUDA_VISIBLE_DEVICES=1 python pc_sampling_lzl_daikuan.py



