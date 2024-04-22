#!/bin/bash
#$ -l rt_AF=60
#$ -l h_rt=00:30:00
#$ -j y
#$ -o outputs/a100/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

source .env/bin/activate

# hostfile
mkdir -p ./hostfile

NUM_GPU_PER_NODE=8

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line
do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done < "$SGE_JOB_HOSTLIST" > "$HOSTFILE_NAME"

# unhealthy node list
mkdir -p ./unhealthy_node_list

# nccl test clone
nccl_tests_dir=nccl-tests

if [ ! -d ${nccl_tests_dir} ]; then
  git clone https://github.com/NVIDIA/nccl-tests.git
  cd ${nccl_tests_dir}
  make MPI=1 MPI_HOME=${OMPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME=${NCCL_HOME}
  cd ..
else
  echo "nccl-tests already exists"
fi

# run
python node_check.py \
  --hostfile ${HOSTFILE_NAME} \
  --all-reduce-perf nccl-tests/build/all_reduce_perf \
  --unhealthy-node-list-dir unhealthy_node_list
