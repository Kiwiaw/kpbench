# source this in every sbatch script (Euler)
source ~/startup_3dv.sh
export KPBENCH_DATA=/cluster/scratch/kkaczmarzyk
export KPBENCH_CACHE=/cluster/scratch/kkaczmarzyk/kpbench/cache
export KPBENCH_RESULTS=$HOME/kpbench/results
export RACO_DIR=$HOME/RaCo
export DEV=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd ~/kpbench
