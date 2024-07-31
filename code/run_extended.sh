#!/bin/sh

if [ "$1" != "-steps" ] || [ "$2" == "" ] || [ "$3" != "-gpu" ] || [ "$4" == "" ]
then
  echo ""
  echo "Usage: sh $0 -steps <simulation_steps> -gpu <gpu_id>"
  echo ""
  echo "Example:"
  echo "  4000 steps on gpu 0:"
  echo "    bash $0 -steps 4000 -gpu 0"
  echo ""
  exit 0
fi

GPU_ID="$4"
X_VALUES=(8 16 32)
Y_VALUES=(8 16 32)
REPETITIONS=4
SIMULATION_STEPS="$2"
OUTPUT_FILE=results_extended_${SIMULATION_STEPS}steps_`date +"%Y-%m-%d_%T"`.csv

SYSYTEM_GPUS=`nvidia-smi -L`
SYSYTEM_GPUS="${SYSYTEM_GPUS// /_}" 
echo "HOSTNAME:${HOSTNAME}" | tee -a $OUTPUT_FILE
echo "${SYSYTEM_GPUS}" | tee -a $OUTPUT_FILE
echo "Used_GPU:${GPU_ID}" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE
echo "Input_to_the_simulation" | tee -a $OUTPUT_FILE
echo "simulation_steps_[s]:$SIMULATION_STEPS" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# make clean
make
run_experiments()
{
  echo "" | tee -a ./$OUTPUT_FILE
  echo "${MESSAGE}" | tee -a ./$OUTPUT_FILE
  echo "Current_directory:`pwd`" | tee -a ./$OUTPUT_FILE
  echo "run;BLOCK_SIZE_X;BLOCK_SIZE_Y,elapsed_time_[s]" | tee -a ./$OUTPUT_FILE
  for BLOCK_SIZE_X in ${X_VALUES[@]}; do
    for BLOCK_SIZE_Y in ${Y_VALUES[@]}; do
        for ((N=1; N<=REPETITIONS; N++)); do
         #echo -n "$N;" | tee -a ./$OUTPUT_FILE
          printf '%2d;' $N | tee -a ./$OUTPUT_FILE
          CUDA_VISIBLE_DEVICES=$GPU_ID $EXE ../data/stress_test_R_header.txt ../data/stress_test_R_dem.txt ../data/stress_test_R_source.txt $OUT $SIMULATION_STEPS $BLOCK_SIZE_X $BLOCK_SIZE_Y | tee -a ./$OUTPUT_FILE
        done
      done
    done
}


EXE="./sciddicaTcuda"
OUT="./tessina_output_CUDA"
MESSAGE="CUDA_experiments"
run_experiments

EXE="./sciddicaTcuda_tiled_classic_noh"
OUT="./tessina_output_CUDA_tiled_classic_noh"
MESSAGE="CUDA_TILED_CLASSIC_NO_HALO_experiments"
run_experiments

EXE="./sciddicaTcuda_tiled_classic_h"
OUT="./tessina_output_CUDA_tiled_classic_h"
MESSAGE="CUDA_TILED_CLASSIC_HALO_experiments"
run_experiments

EXE="./sciddicaTcuda_tiled_halo_sync"
OUT="./tessina_output_CUDA_tiled_halo_sync"
MESSAGE="CUDA_TILED_HALO_SYNC_experiments"
run_experiments

EXEC="./sciddicaTcuda_tiled_halo_sync_priv"
OUT="./tessina_output_CUDA_tiled_halo_sync_priv"
MESSAGE="CUDA_TILED_HALO_SYNC_PRIV_experiments"
run_experiments

