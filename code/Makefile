ifndef CPPC
	CPPC=g++
endif


HDR=../data/tessina_header.txt
DEM=../data/tessina_dem.txt
SRC=../data/tessina_source.txt
OUT_NAIVE=./tessina_output_naive
OUT_TILED_BASIC_HALO=./tessina_output_tiled_basic_halo
OUT_TILED_BASIC_NO_HALO=./tessina_output_tiled_basic_no_halo
OUT_TILED_CFAME =./tessina_output_tiled_CfAMe
OUT_TILED_CFAMO =./tessina_output_tiled_CfAMo
STEPS=4000

BLOCK_SIZE_0 = 16
BLOCK_SIZE_1 = 16
EXEC_NAIVE = sciddicaT_naive
EXEC_TILED_BASIC_HALO = sciddicaT_tiled_basic_halo
EXEC_TILED_BASIC_NO_HALO = sciddicaT_tiled_basic_no_halo
EXEC_TILED_CFAME = sciddicaT_tiled_CfAMe
EXEC_TILED_CFAMO = sciddicaT_tiled_CfAMo


EXECUTABLES = $(EXEC_NAIVE) $(EXEC_TILED_BASIC_HALO) $(EXEC_TILED_BASIC_NO_HALO) $(EXEC_TILED_CFAME) $(EXEC_TILED_CFAMO)

default: all
all: $(EXECUTABLES)

$(EXEC_NAIVE): sciddicaT_naive.cu
	nvcc $^ -o $@ -O3 -w
$(EXEC_TILED_BASIC_HALO): sciddicaT_tiled_basic_halo.cu
	nvcc $^ -o $@ -O3 -w
$(EXEC_TILED_BASIC_NO_HALO): sciddicaT_tiled_basic_no_halo.cu
	nvcc $^ -o $@ -O3 -w
$(EXEC_TILED_CFAME): sciddicaT_tiled_CfAMe.cu
	nvcc $^ -o $@ -O3 -w
$(EXEC_TILED_CFAMO): sciddicaT_tiled_CfAMo.cu
	nvcc $^ -o $@ -w


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


clean:
	rm -f $(EXECUTABLES) *.o *output*

wipe:
	rm -f *.o *output*
