#!/bin/bash


cp /home/man/Documents/ER_Bernoulli_Robust_MPC/build/Executables/gpu_simulate_from_data /home/man/Documents/ER_Bernoulli_Robust_MPC/build/data/gpu_simulate_from_data
cp /home/man/Documents/ER_Bernoulli_Robust_MPC/build/Executables/validate_from_data /home/man/Documents/ER_Bernoulli_Robust_MPC/build/data/validate_from_data
# Copy data from the source to the destination
scp -r /home/man/Documents/ER_Bernoulli_Robust_MPC/build/data/ jonas.hjulstad@hpc.unitn.it:/home/jonas.hjulstad/MC_Simulations/
