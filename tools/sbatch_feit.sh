#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#SBATCH --partition=feit-gpu-a100 
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --job-name="nerf"
#SBATCH --account="punim2198"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=262114 
#SBATCH --time=3-24:0:00
#SBATCH --output=myjob_output_%j.txt
#SBATCH --error=myjob_error_%j.txt
# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

## Use an account that has GPGPU access
 

pip install numpy 
module purge
 
module load cuDNN/8.7.0.84-CUDA-11.8.0

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip3 install torch_geometric
pip3 install einops
   
srun python -m torch.distributed.launch --nproc_per_node=4 --master_port=11332  tools/train.py projects/NeRF-Det/configs/nerfdet_res101_2x_low_res_depth.py --launcher pytorch --work-dir \
/data/gpfs/projects/punim2198/lzc_data/scannet/work_dir/1105/feit_101_depth_0.1
 ##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s