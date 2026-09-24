if [ "$1" != "" ] [ "$2" != "-steps" ] || [ "$3" == "" ] || [ "$4" != "-rep" ] || [ "$5" == "" ] || [ "$6" != "-prec" ] || [ "$7" == "" ] || [ "$8" != "-gpu" ] || [ "$9" == "" ]
then
  echo ""
  echo "Usage: sh $0 program_name -steps <simulation_steps> -rep <number_of_repetitions> -prec <real_precision> -gpu <gpu_id>"
  echo ""
  echo "Example:"
  echo "  4000 steps, 10 repetitions, double precision on gpu 0:"
  echo "    bash $0 sciddicaT_naive -steps 4000 -rep 10 -prec double -gpu 0"
  echo ""
  exit 0
fi

X_VALUES=(8 16 32)
Y_VALUES=(8 16 32)
SIMULATION_STEPS="$3"
REPETITIONS="$5"
REAL_PRECISION="$7"
GPU_ID="$9"
EXE="./$1"
OUT="./output_tessina_$1"
OUTPUT_FILE=results/results_$1_${REAL_PRECISION}_${SIMULATION_STEPS}steps_`date +"%Y-%m-%d_%H-%M-%S"`.csv

IN_HDR="../data/tessina_header.txt"
IN_DEM="../data/tessina_dem.txt"
IN_SRC="../data/tessina_source.txt"

print_info()
{
  SYSYTEM_GPUS=`nvidia-smi -L`
  SYSYTEM_GPUS="${SYSYTEM_GPUS// /_}" 
  echo "HOSTNAME:${HOSTNAME}" | tee -a $OUTPUT_FILE
  echo "${SYSYTEM_GPUS}" | tee -a $OUTPUT_FILE
  echo "Used_GPU:${GPU_ID}" | tee -a $OUTPUT_FILE
  echo "" | tee -a $OUTPUT_FILE

  echo "Input_to_the_simulation" | tee -a $OUTPUT_FILE
  echo "    header:$IN_HDR" | tee -a $OUTPUT_FILE
  echo "    DEM:$IN_DEM" | tee -a $OUTPUT_FILE
  echo "    SOURCES:$IN_SRC" | tee -a $OUTPUT_FILE
  echo "    simulation_steps:$SIMULATION_STEPS" | tee -a $OUTPUT_FILE
  echo "" | tee -a $OUTPUT_FILE
}

run_experiments()
{
  #make clean
  #make
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

MESSAGE="running $1"
print_info
run_experiments