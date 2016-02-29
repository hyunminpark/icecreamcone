#!/bin/bash
#PBS -N Julia
#PBS -l nodes=1:ppn=1
#PBS -j oe

# execute program
cd $PBS_O_WORKDIR
module load julia
julia PS3_JuMP_Eliot_2_HMSS.jl
