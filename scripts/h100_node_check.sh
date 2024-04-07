#!/bin/bash
#SBATCH --job-name=nccl-tests
#SBATCH --time=1:00:00
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/nccl-tests/%x-%j.out
#SBATCH --error=outputs/nccl-tests/%x-%j.out

set -e

# module load
module load cuda/12.1
module load cudnn/8.9.7
module load hpcx/2.17.1

# open file limit
ulimit -n 65536 1048576

# nccl2
export NCCL_HOME="/usr/local/nccl2"

# Important TCPX environment variables
UDS_PATH="/run/tcpx-${SLURM_JOB_ID}"

# Only use TCPX for multi-node jobs.
[[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]] && export USE_TCPX=yes || export USE_TCPX=no

# Only use TCPX for multi-node jobs.
if [[ ${USE_TCPX} = "yes" ]]; then
  # Set up NCCL Environment variables
  export NCCL_NET=GPUDirectTCPX_v7
  # These network interfaces use Ubuntu's consistent naming scheme. See
  # https://manpages.ubuntu.com/manpages/focal/man7/systemd.net-naming-scheme.7.html
  export NCCL_SOCKET_IFNAME=enp0s12
  export NCCL_GPUDIRECTTCPX_CTRL_DEV=enp0s12
  export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=enp6s0,enp12s0,enp134s0,enp140s0
  export NCCL_CROSS_NIC=0
  export NCCL_ALGO=Ring
  export NCCL_PROTO=Simple
  export NCCL_NSOCKS_PERTHREAD=4
  export NCCL_SOCKET_NTHREADS=1
  export NCCL_MAX_NCHANNELS=12
  export NCCL_MIN_NCHANNELS=12
  export NCCL_DYNAMIC_CHUNK_SIZE=524288
  export NCCL_P2P_NET_CHUNKSIZE=524288
  export NCCL_P2P_PCI_CHUNKSIZE=524288
  export NCCL_P2P_NVL_CHUNKSIZE=1048576
  export NCCL_BUFFSIZE=4194304
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export NCCL_NET_GDR_LEVEL=PIX
  export NCCL_P2P_PXN_LEVEL=0
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=${UDS_PATH}
  export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000
  export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
  export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=1000

  export LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH}
else
  unset NCCL_NET
fi

# hostfile
mkdir -p ./hostfile

NUM_GPU_PER_NODE=8

HOSTFILE_NAME=./hostfile/hostfile_${SLURM_JOB_ID}
scontrol show hostnames $SLURM_JOB_NODELIST | while read -r line
do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done > "$HOSTFILE_NAME"

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
