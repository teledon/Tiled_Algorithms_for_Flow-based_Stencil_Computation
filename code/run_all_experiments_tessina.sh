#!/bin/sh


NUM_REPS=10
STEPS=4000

echo job start time is `date`
echo `hostname`

mkdir -p ./results/
rm ./results/*.csv

PREC_FLAGS=("" "-DDOUBLE_PRECISION")

for PREC in "${PREC_FLAGS[@]}"; do
    
    PRECISION=""
    echo ""

    if [ -z "$PREC" ]; then
        echo "Running with single precision (float)."
        PRECISION="float"
    else
        echo "Running with double precision."
        PRECISION="double"
    fi
    
    make clean
    make REAL_PRECISION_FLAG="$PREC"

    bash run_experiment.sh sciddicaT_naive -steps $STEPS -rep $NUM_REPS -prec $PRECISION -gpu 0
    bash run_experiment.sh sciddicaT_tiled_basic_halo -steps $STEPS -rep $NUM_REPS -prec $PRECISION -gpu 0
    bash run_experiment.sh sciddicaT_tiled_basic_no_halo -steps $STEPS -rep $NUM_REPS -prec $PRECISION -gpu 0
    bash run_experiment.sh sciddicaT_tiled_CfAMe -steps $STEPS -rep $NUM_REPS -prec $PRECISION -gpu 0 
    bash run_experiment.sh sciddicaT_tiled_CfAMo -steps $STEPS -rep $NUM_REPS -prec $PRECISION -gpu 0

done

echo job end time is `date`
