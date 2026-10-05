############
# COMPILER #
############
ifndef CPPC
	CPPC=g++
endif


###########
# DATASET #
###########
HDR=../data/tessina_header.txt
DEM=../data/tessina_dem.txt
SRC=../data/tessina_source.txt
OUT_NAIVE=./output_tessina_naive
OUT_TILED_BASIC_HALO=./output_tessina_tiled_basic_halo
OUT_TILED_BASIC_NO_HALO=./output_tessina_tiled_basic_no_halo
OUT_TILED_CFAME =./output_tessina_tiled_CfAMe
OUT_TILED_CFAMO =./output_tessina_tiled_CfAMo
OUT_ROOFLINE=./output_tessina
# OUT_ROOFLINE is only used temporarily during roofline coords computation
STEPS=100


###############
# COMPILATION #
###############
EXEC_NAIVE = sciddicaT_naive
EXEC_TILED_BASIC_HALO = sciddicaT_tiled_basic_halo
EXEC_TILED_BASIC_NO_HALO = sciddicaT_tiled_basic_no_halo
EXEC_TILED_CFAME = sciddicaT_tiled_CfAMe
EXEC_TILED_CFAMO = sciddicaT_tiled_CfAMo

REAL_PRECISION_FLAG=-DDOUBLE_PRECISION
FLOWTIME_ONLY_FLAG=-DFLOWTIME_ONLY

EXECUTABLES = $(EXEC_NAIVE) $(EXEC_TILED_BASIC_HALO) $(EXEC_TILED_BASIC_NO_HALO) $(EXEC_TILED_CFAME) $(EXEC_TILED_CFAMO)

default: all
all: $(EXECUTABLES)

$(EXEC_NAIVE): sciddicaT_naive.cu
	nvcc $^ -o $@ -O3 -w $(REAL_PRECISION_FLAG) $(FLOWTIME_ONLY_FLAG)
$(EXEC_TILED_BASIC_HALO): sciddicaT_tiled_basic_halo.cu
	nvcc $^ -o $@ -O3 -w $(REAL_PRECISION_FLAG) $(FLOWTIME_ONLY_FLAG)
$(EXEC_TILED_BASIC_NO_HALO): sciddicaT_tiled_basic_no_halo.cu
	nvcc $^ -o $@ -O3 -w $(REAL_PRECISION_FLAG) $(FLOWTIME_ONLY_FLAG)
$(EXEC_TILED_CFAME): sciddicaT_tiled_CfAMe.cu
	nvcc $^ -o $@ -O3 -w $(REAL_PRECISION_FLAG) $(FLOWTIME_ONLY_FLAG)
$(EXEC_TILED_CFAMO): sciddicaT_tiled_CfAMo.cu
	nvcc $^ -o $@ -w $(REAL_PRECISION_FLAG) $(FLOWTIME_ONLY_FLAG)


#############
# EXECUTION #
#############
BLOCK_SIZE_0 = 16
BLOCK_SIZE_1 = 16

.PHONY: run_naive run_tiled_basic_halo run_tiled_basic_no_halo run_tiled_CfAMe run_tiled_CfAMo

run_naive: $(EXEC_NAIVE)
	./$(EXEC_NAIVE) $(HDR) $(DEM) $(SRC) $(OUT_NAIVE) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_NAIVE) && cat $(HDR) $(OUT_NAIVE) > $(OUT_NAIVE).dat && rm $(OUT_NAIVE)

run_tiled_basic_halo: $(EXEC_TILED_BASIC_HALO)
	./$(EXEC_TILED_BASIC_HALO) $(HDR) $(DEM) $(SRC) $(OUT_TILED_BASIC_HALO) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_TILED_BASIC_HALO) && cat $(HDR) $(OUT_TILED_BASIC_HALO) > $(OUT_TILED_BASIC_HALO).dat && rm $(OUT_TILED_BASIC_HALO)

run_tiled_basic_no_halo: $(EXEC_TILED_BASIC_NO_HALO)
	./$(EXEC_TILED_BASIC_NO_HALO) $(HDR) $(DEM) $(SRC) $(OUT_TILED_BASIC_NO_HALO) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_TILED_BASIC_NO_HALO) && cat $(HDR) $(OUT_TILED_BASIC_NO_HALO) > $(OUT_TILED_BASIC_NO_HALO).dat && rm $(OUT_TILED_BASIC_NO_HALO)

run_tiled_CfAMe: $(EXEC_TILED_CFAME)
	./$(EXEC_TILED_CFAME) $(HDR) $(DEM) $(SRC) $(OUT_TILED_CFAME) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_TILED_CFAME) && cat $(HDR) $(OUT_TILED_CFAME) > $(OUT_TILED_CFAME).dat && rm $(OUT_TILED_CFAME)

run_tiled_CfAMo: $(EXEC_TILED_CFAMO)
	./$(EXEC_TILED_CFAMO) $(HDR) $(DEM) $(SRC) $(OUT_TILED_CFAMO) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_TILED_CFAMO) && cat $(HDR) $(OUT_TILED_CFAMO) > $(OUT_TILED_CFAMO).dat && rm $(OUT_TILED_CFAMO)


###############################################################################
# ROOFLINE DATA                                                               #
###############################################################################

RL_NCU_FILTER=-k "regex:FlowsComputation|WidthUpdate"
RL_NCU_OPT_METRICS=sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,sm__inst_executed_pipe_tensor.sum,dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum
RL_NCU_RUNTIME_METRICS=sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second
RL_NCU_PARAMS=--section-folder ./sections --print-summary per-kernel --csv
RL_NCU_LAUNCH_OPT=TMP=$(HOME)/tmp ncu $(RL_NCU_FILTER) $(RL_NCU_PARAMS) --metrics $(RL_NCU_OPT_METRICS)
RL_NCU_LAUNCH_RUNTIME=TMP=$(HOME)/tmp ncu $(RL_NCU_FILTER) $(RL_NCU_PARAMS) --metrics $(RL_NCU_RUNTIME_METRICS)
RL_EXE_PARAMS=$(HDR) $(DEM) $(SRC) $(OUT_ROOFLINE) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1)

.PHONY: roofline

roofline: $(RL_EXE)
	$(RL_NCU_LAUNCH_OPT) --log-file roofline/prof_opt_$(RL_EXE).csv ./$(RL_EXE) $(RL_EXE_PARAMS)
	$(RL_NCU_LAUNCH_RUNTIME) --log-file roofline/prof_runtime_$(RL_EXE).csv ./$(RL_EXE) $(RL_EXE_PARAMS)
	python3 roofline/roofline.py roofline/prof_opt_$(RL_EXE).csv roofline/prof_runtime_$(RL_EXE).csv
	rm roofline/prof_opt_$(RL_EXE).csv roofline/prof_runtime_$(RL_EXE).csv $(OUT_ROOFLINE)*

rl_naive: RL_EXE = $(EXEC_NAIVE)
rl_naive: $(EXEC_NAIVE) roofline

rl_tiled_basic_halo: RL_EXE = $(EXEC_TILED_BASIC_HALO)
rl_tiled_basic_halo: $(EXEC_TILED_BASIC_HALO) roofline

rl_tiled_basic_no_halo: RL_EXE = $(EXEC_TILED_BASIC_NO_HALO)
rl_tiled_basic_no_halo: $(EXEC_TILED_BASIC_NO_HALO) roofline

rl_tiled_CfAMe: RL_EXE = $(EXEC_TILED_CFAME)
rl_tiled_CfAMe: $(EXEC_TILED_CFAME) roofline

rl_tiled_CfAMo: RL_EXE = $(EXEC_TILED_CFAMO)
rl_tiled_CfAMo: $(EXEC_TILED_CFAMO) roofline


############
# CLEAN UP #
############

clean:
	rm -f $(EXECUTABLES) *.o *output*

wipe:
	rm -f *.o *output*
