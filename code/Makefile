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
NCU_STEPS=50


###############
# COMPILATION #
###############
EXEC_NAIVE = sciddicaT_naive
EXEC_TILED_BASIC_HALO = sciddicaT_tiled
EXEC_TILED_BASIC_NO_HALO = sciddicaT_no_halo
EXEC_TILED_CFAME = sciddicaT_cfame
EXEC_TILED_CFAMO = sciddicaT_cfamo

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
RL_EXE_PARAMS=$(HDR) $(DEM) $(SRC) $(OUT_ROOFLINE) $(NCU_STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1)
RL_CREATE_CHART_PYSCRIPT=../../Efficient_Execution_of_Flow-Based_Low-Order_Stencil-like_Algorithms_on_GPU_experiments/Roofline/scripts/roofline_coords.py

.PHONY: roofline

roofline: $(RL_EXE)
	mkdir -p roofline
	mkdir -p sections
	mkdir -p $(HOME)/tmp
	PREC_TXTs="float double"; \
	for PREC_TXT in $$PREC_TXTs; do \
		mkdir -p "traces/$$PREC_TXT"; \
		if [ "$$PREC_TXT" = "float" ]; then \
			PREC=""; \
		else \
			PREC="-DDOUBLE_PRECISION"; \
		fi; \
		$(MAKE) clean; \
		$(MAKE) $(RL_EXE) REAL_PRECISION_FLAG="$$PREC"; \
		$(RL_NCU_LAUNCH_OPT) --log-file "roofline/prof_opt_$(RL_EXE)_$$PREC_TXT.csv" ./$(RL_EXE) $(RL_EXE_PARAMS); \
		$(RL_NCU_LAUNCH_RUNTIME) --log-file "roofline/prof_runtime_$(RL_EXE)_$$PREC_TXT.csv" ./$(RL_EXE) $(RL_EXE_PARAMS); \
		python3 $(RL_CREATE_CHART_PYSCRIPT) sciddicaT "$$PREC_TXT" "roofline/prof_opt_$(RL_EXE)_$$PREC_TXT.csv" "roofline/prof_runtime_$(RL_EXE)_$$PREC_TXT.csv"; \
	done
	rm -r roofline

rl_naive: RL_EXE = $(EXEC_NAIVE)
rl_naive: roofline

rl_tiled: RL_EXE = $(EXEC_TILED_BASIC_HALO)
rl_tiled: roofline

rl_no_halo: RL_EXE = $(EXEC_TILED_BASIC_NO_HALO)
rl_no_halo: roofline

rl_cfame: RL_EXE = $(EXEC_TILED_CFAME)
rl_cfame: roofline

rl_cfamo: RL_EXE = $(EXEC_TILED_CFAMO)
rl_cfamo: roofline


###############################################################################
# TRACE DATA                                                                  #
###############################################################################
TRACE_TIME_METRICS="smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio,smsp__warp_issue_stalled_lg_throttle_per_warp_active.ratio,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio,smsp__warp_issue_stalled_mio_throttle_per_warp_active.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.ratio,smsp__warps_active.sum"
TRACE_SCHEDULE_METRICS="launch__block,launch__grid_size,sm__ctas_active.avg,sm__cycles_active.avg,gpu__time_duration.sum,device__attribute_multiprocessor_count"
TRACE_EXE_PARAMS=$(HDR) $(DEM) $(SRC) $(OUT_ROOFLINE) $(NCU_STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1)
TRACE_TIME_CHART_PYSCRIPT=../../Efficient_Execution_of_Flow-Based_Low-Order_Stencil-like_Algorithms_on_GPU_experiments/Trace/scripts/trace.py

.PHONY: trace

trace: $(TRACE_EXE)
	mkdir -p traces
	mkdir -p sections
	mkdir -p $(HOME)/tmp
	PREC_TXTs="float double"; \
	for PREC_TXT in $$PREC_TXTs; do \
		mkdir -p "traces/$$PREC_TXT"; \
		if [ "$$PREC_TXT" = "float" ]; then \
			PREC=""; \
		else \
			PREC="-DDOUBLE_PRECISION"; \
		fi; \
		$(MAKE) clean; \
		$(MAKE) $(TRACE_EXE) REAL_PRECISION_FLAG="$$PREC"; \
		TMP=$(HOME)/tmp ncu --kernel-name "regex:FlowsComputation|WidthUpdate" --section-folder ./sections --metrics $(TRACE_TIME_METRICS) --print-summary per-kernel --csv --log-file "traces/$$PREC_TXT/$(TRACE_EXE).time" $(TRACE_EXE) $(TRACE_EXE_PARAMS); \
		TMP=$(HOME)/tmp ncu --kernel-name "regex:FlowsComputation|WidthUpdate" --section-folder ./sections --metrics $(TRACE_SCHEDULE_METRICS) --print-summary per-kernel --csv --log-file "traces/$$PREC_TXT/$(TRACE_EXE).sched" $(TRACE_EXE) $(TRACE_EXE_PARAMS); \
		$(TRACE_TIME_CHART_PYSCRIPT) sciddicaT "$(BLOCK_SIZE_0)x$(BLOCK_SIZE_1)" "$$PREC_TXT" "./traces/$$PREC_TXT"; \
	done

trace_naive: TRACE_EXE = $(EXEC_NAIVE)
trace_naive: trace

trace_tiled: TRACE_EXE = $(EXEC_TILED_BASIC_HALO)
trace_tiled: trace

trace_no_halo: TRACE_EXE = $(EXEC_TILED_BASIC_NO_HALO)
trace_no_halo: trace

trace_cfame: TRACE_EXE = $(EXEC_TILED_CFAME)
trace_cfame: trace

trace_cfamo: TRACE_EXE = $(EXEC_TILED_CFAMO)
trace_cfamo: trace


############
# CLEAN UP #
############

clean:
	rm -f $(EXECUTABLES) *.o *output*

wipe:
	rm -f *.o *output*
