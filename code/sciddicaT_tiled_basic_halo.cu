#include <iostream>
using namespace std;
#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "util.hpp"
#include "sciddicaT.hpp"


__global__
void sciddicaTFlowsComputation(const integer_t i_start, const integer_t i_end, const integer_t j_start, const integer_t j_end,
                               const integer_t r, const integer_t c,
                               const real_t* const __restrict__ Sz, const real_t* const __restrict__ Sh, real_t* const __restrict__ Sf)
{
    integer_t i = blockIdx.y * (blockDim.y-2) + threadIdx.y - 1;
    integer_t j = blockIdx.x * (blockDim.x-2) + threadIdx.x - 1;

    integer_t i_out = i + 1;
    integer_t j_out = j + 1;
    if (i_out < i_start || i_out >= i_end || j_out < j_start || j_out >= j_end)
        return;

    extern __shared__ real_t  Sz_shared[];
                      real_t* Sh_shared = Sz_shared + blockDim.x * blockDim.y;

    SET(Sz_shared, blockDim.x, threadIdx.y, threadIdx.x, GET(Sz, c, i, j));
    SET(Sh_shared, blockDim.x, threadIdx.y, threadIdx.x, GET(Sh, c, i, j));

    __syncthreads();

    if (threadIdx.y < 1 || threadIdx.y >= blockDim.y - 1 ||
        threadIdx.x < 1 || threadIdx.x >= blockDim.x - 1)
        return;


    bool eliminated_cells[NEIGBORHOOD_SIZE] = {false, false, false, false, false};
    bool again;
    uint8_t cells_count;
    real_t average;
    real_t m;
    real_t u[NEIGBORHOOD_SIZE];
    real_t z, h;

    m = GET(Sh_shared, blockDim.x, threadIdx.y, threadIdx.x) - P_EPSILON;
    u[0] = GET(Sz_shared, blockDim.x, threadIdx.y, threadIdx.x) + P_EPSILON;

    for (uint8_t n = 1; n <= ADJACENT_CELLS; n++)
    {
        z = GET(Sz_shared, blockDim.x, threadIdx.y + Xi[n], threadIdx.x + Xj[n]);
        h = GET(Sh_shared, blockDim.x, threadIdx.y + Xi[n], threadIdx.x + Xj[n]);
        u[n] = z + h;
    }

    do
    {
        again = false;
        average = m;
        cells_count = 0;

        for (uint8_t n = 0; n < NEIGBORHOOD_SIZE; n++)
            if (!eliminated_cells[n])
            {
                average += u[n];
                cells_count++;
            }

        if (cells_count != 0)
            average /= cells_count;

        for (uint8_t n = 0; n < NEIGBORHOOD_SIZE; n++)
            if ((average <= u[n]) && (!eliminated_cells[n]))
            {
                eliminated_cells[n] = true;
                again = true;
            }
    } while (again);

    if (!eliminated_cells[1]) BUF_SET(Sf, r, c, 0, i, j, (average - u[1]) * P_R);
    if (!eliminated_cells[2]) BUF_SET(Sf, r, c, 1, i, j, (average - u[2]) * P_R);
    if (!eliminated_cells[3]) BUF_SET(Sf, r, c, 2, i, j, (average - u[3]) * P_R);
    if (!eliminated_cells[4]) BUF_SET(Sf, r, c, 3, i, j, (average - u[4]) * P_R);
}

__global__
void sciddicaTWidthUpdate(const integer_t i_start, const integer_t i_end, const integer_t j_start, const integer_t j_end,
                          const integer_t r, const integer_t c,
                          real_t* const __restrict__ Sh, const real_t* const __restrict__ Sf)
{
    integer_t i =  blockIdx.y * (blockDim.y-2) + threadIdx.y - 1;
    integer_t j =  blockIdx.x * (blockDim.x-2) + threadIdx.x - 1;
    integer_t i_out = i + 1;
    integer_t j_out = j + 1;

    extern __shared__ real_t Sf_shared[];

    if (i_out < i_start || i_out >= i_end || j_out < j_start || j_out >= j_end)
    {
        for (uint8_t n = 1; n <= ADJACENT_CELLS; n++)
            BUF_SET(Sf_shared, blockDim.y, blockDim.x, n-1, threadIdx.y, threadIdx.x, 0.0);
        return;
    }

    for (uint8_t n = 1; n <= ADJACENT_CELLS; n++)
        BUF_SET(Sf_shared, blockDim.y, blockDim.x, n-1, threadIdx.y, threadIdx.x, BUF_GET(Sf, r, c, n-1, i, j));

    __syncthreads();

    if (threadIdx.y < 1 || threadIdx.y >= blockDim.y - 1 ||
        threadIdx.x < 1 || threadIdx.x >= blockDim.x - 1)
        return;

    real_t h_next = GET(Sh, c, i, j);

    for (uint8_t n = 1; n <= ADJACENT_CELLS; n++)
        h_next += BUF_GET(Sf_shared, blockDim.y, blockDim.x, (ADJACENT_CELLS-n), threadIdx.y+Xi[n], threadIdx.x+Xj[n]) \
                - BUF_GET(Sf_shared, blockDim.y, blockDim.x, (n-1), threadIdx.y, threadIdx.x);

    SET(Sh, c, i, j, h_next);
}


class SciddicaTCudaHalo: public SciddicaTCuda<SciddicaTCudaHalo>
{
    friend class SciddicaTCuda<SciddicaTCudaHalo>;

public:
    SciddicaTCudaHalo(char **argv): SciddicaTCuda(argv) {}

protected:
    void init_extras()
    {
        Sf = addLayer2D(ADJACENT_CELLS* r, c); // Allocates the Sf substates grid, 
                                               //   having one layer for each adjacent cell
        cudaMemPrefetchAsync(Sf, sizeof(real_t)*r*c*ADJACENT_CELLS, 0 , NULL);
        
        util::init_dim3( grid_size_tiled, ceil(c/(float)(block_size.x-2)), ceil(r/(float)(block_size.y-2)), 1 );
        shmem_size_fc = 2 * block_size.x * block_size.y * sizeof(real_t);
        shmem_size_wu = ADJACENT_CELLS * block_size.x * block_size.y * sizeof(real_t);

        if (!util::checkShmemSizePerBlock(shmem_size_fc) || !util::checkShmemSizePerBlock(shmem_size_wu))
            exit(EXIT_FAILURE);
    }

    void free_extras()
        { cudaFree(Sf); }

    inline void on_step_start()
        { sciddicaTResetFlows<<<grid_size,block_size>>>(i_start, i_end, j_start, j_end, r, c, Sf); }
    inline void launch_flows_computation_kernel()
        { sciddicaTFlowsComputation<<<grid_size_tiled,block_size,shmem_size_fc>>>(i_start, i_end, j_start, j_end, r, c, Sz, Sh, Sf); }
    inline void launch_width_update_kernel()
        { sciddicaTWidthUpdate<<<grid_size_tiled,block_size,shmem_size_wu>>>(i_start, i_end, j_start, j_end, r, c, Sh, Sf); }

private:
    real_t *Sf;  // Sf: 4 substates containing the flows towards the 4 neighs
    dim3 grid_size_tiled;
    size_t shmem_size_fc;
    size_t shmem_size_wu;
};


int main(int argc, char **argv)
{
    SciddicaTCudaHalo sciddicaT(argv);
    return sciddicaT.execute();
}
