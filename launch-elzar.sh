#!/bin/bash
#$ -cwd
#$ -o elzar-logs
#$ -e elzar-logs
#$ -j y
#$ -N biohsic
#$ -l m_mem_free=36G
#$ -pe threads 8
#$ -l gpu=1
# Array job. Each job will run a different parameter set. 
#$ -t 1-1:1
#$ -tc 10
echo "Running job $JOB_ID, task $SGE_TASK_ID"
echo | which python
echo "Python script: $@"
# python process-checkpoints.py "outputs/generate-weights/2023-05-18/*/checkpoints" # run once
# LD_PRELOAD=/lib64/libtcmalloc.so.4 python train-ddpm-weights.py '+data.checkpoints=["multirun/generate-weights/2023-05-24/22-31-13/*/checkpoints-processed"]' logger=wandb-logger model.nfeatures=128 'model.feature_multipliers=[1, 4, 8]' data.batchsize=128
# python weight-forward-diffusion.py +data=mnist '+weights.checkpoints=["multirun/generate-weights/2023-05-24/22-31-13/0/checkpoints-processed"]' weights.batchsize=100 weights.nsamples=100 data.batchsize=2048 diffusion.ntimesteps=1000
python $@
echo "Job complete!"
