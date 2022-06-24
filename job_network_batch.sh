#!/bin/bash

if [ 1 == 1 ]; then

shuffle=True
epochs=2
momentum=0.9
margin=0.001
reduction="mean"
fine_tune=False

#for fine_tune in 0 1; do
#       for batch_size in 16 32 64; do
#               for lr in 0.1 0.01 0.001; do
#                       JOB_NAME=v11
#                       echo $JOB_NAME
#
#                       sbatch --job-name ${JOB_NAME} --gres=gpu:k80:1 \
#                       job_network.sh ${batch_size} ${shuffle} ${epochs} ${lr} ${momentum} ${margin} ${fine_tune} ${reduction}
#                       sleep 10
#               done
#       done
#done


for batch_size in 16 32 64; do
        for lr in 0.1 0.01 0.001; do
                JOB_NAME=v11
                echo $JOB_NAME

                sbatch --job-name ${JOB_NAME} --gres=gpu:k80:1 \
                job_network.sh ${batch_size} ${shuffle} ${epochs} ${lr} ${momentum} ${margin} ${fine_tune} ${reduction}
                sleep 10
        done
done


fi
