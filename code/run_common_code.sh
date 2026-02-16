if [ "$1" != "-steps" ] || [ "$2" == "" ] || [ "$3" != "-rep" ] || [ "$4" == "" ] || [ "$5" != "-gpu" ] || [ "$6" == "" ]
then
  echo ""
  echo "Usage: sh $0 -steps <simulation_steps> -rep <number_of_repetitions> -gpu <gpu_id>"
  echo ""
  echo "Example:"
  echo "  4000 steps, 10 repetitions on gpu 0:"
  echo "    bash $0 -steps 4000 -rep 10 -gpu 0"
  echo ""
  exit 0
fi

X_VALUES=(8 16 32)
Y_VALUES=(8 16 32)
SIMULATION_STEPS="$2"
REPETITIONS="$4"
GPU_ID="$6"
# OUTPUT_FILE=results_${SIMULATION_STEPS}steps_`date +"%Y-%m-%d_%T"`.csv

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
        for ((N=1; N<=$REPETITIONS; N++)); do
         #echo -n "$N;" | tee -a ./$OUTPUT_FILE
          printf '%2d;' $N | tee -a ./$OUTPUT_FILE
          CUDA_VISIBLE_DEVICES=$GPU_ID $EXE $IN_HDR $IN_DEM $IN_SRC $OUT $SIMULATION_STEPS $BLOCK_SIZE_X $BLOCK_SIZE_Y | tee -a ./$OUTPUT_FILE
        done
      done
    done
}