#!/bin/sh

source run_common_code.sh


OUTPUT_FILE=results_${SIMULATION_STEPS}steps_`date +"%Y-%m-%d_%T"`.csv
IN_HDR="../data/tessina_header.txt"
IN_DEM="../data/tessina_dem.txt"
IN_SRC="../data/tessina_source.txt"


EXE="./sciddicaT_naive"
OUT="./tessina_output_naive"
MESSAGE="TESSINA_EXPERIMENTS_naive"
run_experiments

EXE="./sciddicaT_tiled_basic_halo"
OUT="./tessina_output_tiled_basic_halo"
MESSAGE="TESSINA_EXPERIMENTS_tiled_basic_halo"
run_experiments

EXE="./sciddicaT_tiled_basic_no_halo"
OUT="./tessina_output_tiled_basic_no_halo"
MESSAGE="TESSINA_EXPERIMENTS_tiled_no_halo"
run_experiments

EXE="./sciddicaT_tiled_CfAMe"
OUT="./tessina_output_tiled_CfAMe"
MESSAGE="TESSINA_EXPERIMENTS_tiled_CfAMe"
run_experiments

EXEC="./sciddicaT_tiled_CfAMo"
OUT="./tessina_output_tiled_CfAMo"
MESSAGE="TESSINA_EXPERIMENTS_tiled_CfAMo"
run_experiments
