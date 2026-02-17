#!/bin/sh

OUTPUT_FILE=results_tessina_${SIMULATION_STEPS}steps_`date +"%Y-%m-%d_%T"`.csv
IN_HDR="../data/tessina_header.txt"
IN_DEM="../data/tessina_dem.txt"
IN_SRC="../data/tessina_source.txt"

source run_common_code.sh

EXE="./sciddicaT_naive"
OUT="./output_tessina_naive"
MESSAGE="TESSINA_EXPERIMENTS_naive"
print_info
run_experiments

EXE="./sciddicaT_tiled_basic_halo"
OUT="./output_tessina_tiled_basic_halo"
MESSAGE="TESSINA_EXPERIMENTS_tiled_basic_halo"
print_info
run_experiments

EXE="./sciddicaT_tiled_basic_no_halo"
OUT="./output_tessina_tiled_basic_no_halo"
MESSAGE="TESSINA_EXPERIMENTS_tiled_no_halo"
print_info
run_experiments

EXE="./sciddicaT_tiled_CfAMe"
OUT="./output_tessina_tiled_CfAMe"
MESSAGE="TESSINA_EXPERIMENTS_tiled_CfAMe"
print_info
run_experiments

EXEC="./sciddicaT_tiled_CfAMo"
OUT="./output_tessina_tiled_CfAMo"
MESSAGE="TESSINA_EXPERIMENTS_tiled_CfAMo"
print_info
run_experiments
