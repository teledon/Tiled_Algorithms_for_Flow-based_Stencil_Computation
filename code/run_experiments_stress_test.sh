#!/bin/sh

OUTPUT_FILE=results_extended_${SIMULATION_STEPS}steps_`date +"%Y-%m-%d_%T"`.csv
IN_HDR="../data/stress_test_R_header.txt"
IN_DEM="../data/stress_test_R_dem.txt"
IN_SRC="../data/stress_test_R_source.txt"

source run_common_code.sh

EXE="./sciddicaT_naive"
OUT="./output_stress_test_naive"
MESSAGE="STRESS_TEST_EXPERIMENTS_naive"
print_info
run_experiments

EXE="./sciddicaT_tiled_basic_halo"
OUT="./output_stress_test_tiled_basic_halo"
MESSAGE="STRESS_TEST_EXPERIMENTS_tiled_basic_halo"
print_info
run_experiments

EXE="./sciddicaT_tiled_basic_no_halo"
OUT="./output_stress_test_tiled_basic_no_halo"
MESSAGE="STRESS_TEST_EXPERIMENTS_tiled_basic_no_halo"
print_info
run_experiments

EXE="./sciddicaT_tiled_CfAMe"
OUT="./output_stress_test_tiled_CfAMe"
MESSAGE="STRESS_TEST_EXPERIMENTS_tiled_CfAMe"
print_info
run_experiments
_
EXEC="./sciddicaT_tiled_CfAMo"
OUT="./output_stress_test_tiled_CfAMo"
MESSAGE="STRESS_TEST_EXPERIMENTS_tiled_CfAMo"
print_info
run_experiments
