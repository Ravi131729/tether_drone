
#!/bin/bash
#SBATCH --job-name 1.4parallel_sim
#SBATCH --ntasks 150
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 2gb
#SBATCH --time 02:00:00


module load anaconda3
source activate jax-ai-stack
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Launch MPI program
srun python 1.4parallel_sim.py