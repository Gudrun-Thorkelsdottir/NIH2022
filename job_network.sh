#!/bin/bash


#SBATCH --job-name model_hyp_tune
#SBATCH --partition=gpu
#SBATCH --mail-user=gudrun2001@gmail.com
#SBATCH --mail-type=ALL
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:p100:1
##SBATCH --constraint=p40&gmem24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --ntasks-per-core=1
###SBATCH --output slurm_job-${1}-${4}-${7}.out
###SBATCH --error slurm_job-brca_mtl-%j-%N.err
#SBATCH --time=72:00:00


BATCH_SIZE=${1}
SHUFFLE=${2}
EPOCHS=${3}
LR=${4}
MOMENTUM=${5}
MARGIN=${6}
FINE_TUNE=${7}
REDUCTION=${8}


#echo $BATCH_SIZE
#echo $SHUFFLE
#echo $EPOCHS
#echo $LR
#echo $MOMENTUM
#echo $MARGIN
#echo $FINE_TUNE
#echo $REDUCTION


CUDA_VISIBLE_DEVICES=0 python3 network.py \
--batch_size ${BATCH_SIZE} \
--shuffle ${SHUFFLE} \
--epochs ${EPOCHS} \
--lr ${LR} \
--momentum ${MOMENTUM} \
--margin ${MARGIN} \
--fine_tune ${FINE_TUNE} \
--reduction ${REDUCTION}
