ifndef CPPC
	CPPC=g++
endif


HDR=../data/tessina_header.txt
DEM=../data/tessina_dem.txt
SRC=../data/tessina_source.txt
OUT_CUDA=./tessina_output_CUDA
OUT_CUDA_TILED_CLASSIC_HALO=./tessina_output_CUDA_tiled_classic_h
OUT_CUDA_TILED_CLASSIC_NO_HALO=./tessina_output_CUDA_tiled_classic_noh
OUT_CUDA_TILED_HALO_SYNC =./tessina_output_CUDA_tiled_halo_sync
OUT_CUDA_TILED_HALO_SYNC_PRIV =./tessina_output_CUDA_tiled_halo_sync_priv
STEPS=4000

BLOCK_SIZE_0 = 32
BLOCK_SIZE_1 = 32
EXEC_CUDA = sciddicaTcuda
EXEC_CUDA_TILED_CLASSIC_HALO = sciddicaTcuda_tiled_classic_h
EXEC_CUDA_TILED_CLASSIC_NO_HALO = sciddicaTcuda_tiled_classic_noh
EXEC_CUDA_TILED_HALO_SYNC = sciddicaTcuda_tiled_halo_sync
EXEC_CUDA_TILED_HALO_SYNC_PRIV = sciddicaTcuda_tiled_halo_sync_priv

EXECUTABLES = $(EXEC_CUDA_TILED_CLASSIC_HALO) $(EXEC_CUDA_TILED_CLASSIC_NO_HALO) $(EXEC_CUDA_TILED_HALO_SYNC) $(EXEC_CUDA_TILED_HALO_SYNC_PRIV) $(EXEC_CUDA)

default: all
all: $(EXECUTABLES)

$(EXEC_CUDA_TILED_HALO_SYNC): sciddicaT_tiled_halo_sync.cu
	nvcc $^ -o $@ -w
$(EXEC_CUDA_TILED_HALO_SYNC_PRIV): sciddicaT_tiled_halo_sync_priv.cu
	nvcc $^ -o $@ -O3 -w
$(EXEC_CUDA_TILED_CLASSIC_HALO): sciddicaT_tiled_classic_h.cu
	nvcc $^ -o $@ -O3 -w
$(EXEC_CUDA_TILED_CLASSIC_NO_HALO): sciddicaT_tiled_classic_noh.cu
	nvcc $^ -o $@ -O3 -w
$(EXEC_CUDA): sciddicaT.cu
	nvcc $^ -o $@ -O3 -w

run_cuda_tiled_halo_sync: $(EXEC_CUDA_TILED_HALO_SYNC)
	./$(EXEC_CUDA_TILED_HALO_SYNC) $(HDR) $(DEM) $(SRC) $(OUT_CUDA_TILED_HALO_SYNC) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_CUDA_TILED_HALO_SYNC) && cat $(HDR) $(OUT_CUDA_TILED_HALO_SYNC) > $(OUT_CUDA_TILED_HALO_SYNC).dat && rm $(OUT_CUDA_TILED_HALO_SYNC)

run_cuda_tiled_halo_sync_priv: $(EXEC_CUDA_TILED_HALO_SYNC_PRIV)
	./$(EXEC_CUDA_TILED_HALO_SYNC_PRIV) $(HDR) $(DEM) $(SRC) $(OUT_CUDA_TILED_HALO_SYNC_PRIV) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_CUDA_TILED_HALO_SYNC_PRIV) && cat $(HDR) $(OUT_CUDA_TILED_HALO_SYNC_PRIV) > $(OUT_CUDA_TILED_HALO_SYNC_PRIV).dat && rm $(OUT_CUDA_TILED_HALO_SYNC_PRIV)

run_cuda_tiled_classic_h: $(EXEC_CUDA_TILED_CLASSIC_HALO)
	./$(EXEC_CUDA_TILED_CLASSIC_HALO) $(HDR) $(DEM) $(SRC) $(OUT_CUDA_TILED_CLASSIC_HALO) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_CUDA_TILED_CLASSIC_HALO) && cat $(HDR) $(OUT_CUDA_TILED_CLASSIC_HALO) > $(OUT_CUDA_TILED_CLASSIC_HALO).dat && rm $(OUT_CUDA_TILED_CLASSIC_HALO)

run_cuda_tiled_classic_noh: $(EXEC_CUDA_TILED_CLASSIC_NO_HALO)
	./$(EXEC_CUDA_TILED_CLASSIC_NO_HALO) $(HDR) $(DEM) $(SRC) $(OUT_CUDA_TILED_CLASSIC_NO_HALO) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_CUDA_TILED_CLASSIC_NO_HALO) && cat $(HDR) $(OUT_CUDA_TILED_CLASSIC_NO_HALO) > $(OUT_CUDA_TILED_CLASSIC_NO_HALO).dat && rm $(OUT_CUDA_TILED_CLASSIC_NO_HALO)

run_cuda: $(EXEC_CUDA)
	./$(EXEC_CUDA) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) && md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).dat && rm $(OUT_CUDA)

clean:
	rm -f $(EXECUTABLES) *.o *output*

wipe:
	rm -f *.o *output*


