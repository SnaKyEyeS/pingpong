# Ping-pong benchmark

## Run on Lucia (using MPICH)

 - Submission script:

```bash
#!/bin/bash
#SBATCH --job-name=pingpong
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --account=your_account
#SBATCH --output=pingpong-%j.out

# DBS-compiled MPICH
export PREFIX=$HOME/soft/lib-MPICH-4.2.3-OFI-1.22.0-CUDA-12.2.0-hand
source env.sh

# Compile
mpic++ -O3 -I${EBROOTCUDA}/include -L${EBROOTCUDA}/lib64 -lcudart src/pingpong.cpp -o src/pingpong

# Display some node information
nvidia-smi
nvidia-smi topo -m

# 1 GPU/node
mpiexec -l -n 2 -ppn 1 -bind-to numa -map-by numa:1 ./split.sh ./src/pingpong

# 2 GPU/node
mpiexec -l -n 4 -ppn 2 -bind-to numa -map-by numa:2 ./split.sh ./src/pingpong

# 4 GPU/node
mpiexec -l -n 8 -ppn 4 -bind-to numa -map-by numa:1 ./split.sh ./src/pingpong
```

 - GPU binding script:

```bash
#!/bin/bash

# Get my rank in the world comm
nrank_world=${PMI_SIZE}
irank_world=${PMI_RANK}

# Get my rank in the node-local comm
nrank_local=${MPI_LOCALNRANKS}
irank_local=${MPI_LOCALRANKID}

# Select the device & nic I see
if   [ "${nrank_local}" == 1 ]; then
    # 1 GPU/node
    dev=3
    nic=0

elif [ "${nrank_local}" == 2 ]; then
    # 2 GPU/node
    if   [ "${irank_local}" == 0 ]; then
        dev=3
        nic=0
    elif [ "${irank_local}" == 1 ]; then
        dev=1
        nic=1
    fi

elif [ "${nrank_local}" == 4 ]; then
    # 4 GPU/node
    if   [ "${irank_local}" == 0 ]; then
        dev=3
        nic=0
    elif [ "${irank_local}" == 1 ]; then
        dev=2
        nic=0
    elif [ "${irank_local}" == 2 ]; then
        dev=1
        nic=1
    elif [ "${irank_local}" == 3 ]; then
        dev=0
        nic=1
    fi

else
    raise error "nrank_local = ${nrank_local}, not supported"
fi

export CUDA_VISIBLE_DEVICES=${dev}
export MPIR_CVAR_CH4_OFI_IFNAME=mlx5_${nic}
echo "Setting my GPU to device n°${CUDA_VISIBLE_DEVICES} with ifname ${MPIR_CVAR_CH4_OFI_IFNAME}"

# Run...
$@
```

 - Environment variables & modules:
```bash
#!/bin/bash

# Modules
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$PREFIX/bin:$PATH
module load EasyBuild/2023a
module load CUDA/12.2.0
module load GCC/12.3.0
module load GDRCopy/2.3.1-GCCcore-12.3.0.lua

# OFI env vars
export FI_PROVIDER="verbs,ofi_rxm,shm"
export FI_HMEM_CUDA_USE_GDRCOPY=1

export FI_OFI_RXM_BUFFER_SIZE=512
export FI_OFI_RXM_SAR_LIMIT=512     # we don't want SAR

# OFI debug stuff
#export FI_LOG_LEVEL=Info

# MPICH env vars
export MPIR_CVAR_NOLOCAL=1
export MPIR_CVAR_ENABLE_GPU=1
export MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1

# MPICH debug stuff
#export MPICH_DBG=yes
#export MPIR_CVAR_DEBUG_SUMMARY=1

# HYDRA env vars
export HYDRA_TOPO_DEBUG=1
```
