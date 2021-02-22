#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N jobthreenotadaptivecond3     
#$ -cwd                  
#$ -l h_rt=400:00:00 
#$ -l h_vmem=4G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh
 
# Load Python
module load anaconda
source activate mypython 
 
# Run the program
python boolean_not_adaptive_pcfg_5_rules_1000_survive_cond_3_3.py