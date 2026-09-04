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
    integer_t i = blockIdx.y * blockDim.y + threadIdx.y;
    integer_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < i_start or i >= i_end or j < j_start or j >= j_end)
        return;

    bool eliminated_cells[NEIGBORHOOD_SIZE] = {false, false, false, false, false};
    bool again;
    uint8_t cells_count;
    real_t average;
    real_t m;
    real_t u[NEIGBORHOOD_SIZE];
    real_t z, h;

    extern __shared__ real_t  Sz_shared[];
                      real_t* Sh_shared = Sz_shared + blockDim.x * blockDim.y;

    SET(Sz_shared, blockDim.x, threadIdx.y, threadIdx.x, GET(Sz, c, i, j));
    SET(Sh_shared, blockDim.x, threadIdx.y, threadIdx.x, GET(Sh, c, i, j));

    __syncthreads();

    integer_t This_tile_start_point_x = blockIdx.x * blockDim.x;
    integer_t Next_tile_start_point_x = (blockIdx.x + 1) * blockDim.x;
    integer_t This_tile_start_point_y = blockIdx.y * blockDim.y;
    integer_t Next_tile_start_point_y = (blockIdx.y + 1) * blockDim.y;

    m = GET(Sh_shared, blockDim.x, threadIdx.y, threadIdx.x) - P_EPSILON;
    u[0] = GET(Sz_shared, blockDim.x, threadIdx.y, threadIdx.x) + P_EPSILON;

    for (uint8_t n = 1; n <= ADJACENT_CELLS; n++)
    {
        if (i + Xi[n] >= This_tile_start_point_y && i + Xi[n] < Next_tile_start_point_y  &&
            j + Xj[n] >= This_tile_start_point_x && j + Xj[n] < Next_tile_start_point_x) 
        {
            z = GET(Sz_shared, blockDim.x, threadIdx.y + Xi[n], threadIdx.x + Xj[n]);
            h = GET(Sh_shared, blockDim.x, threadIdx.y + Xi[n], threadIdx.x + Xj[n]);
        }
        else
        {
            z = GET(Sz, c, i + Xi[n], j + Xj[n]);
            h = GET(Sh, c, i + Xi[n], j + Xj[n]);
        }
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
    integer_t i = blockIdx.y * blockDim.y + threadIdx.y;
    integer_t j = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ real_t Sf_shared[];

    if (i < i_start or i >= i_end or j < j_start or j >= j_end)
    {
        for (uint8_t n = 1; n <= ADJACENT_CELLS; n++)
            BUF_SET(Sf_shared, blockDim.y, blockDim.x, n-1, threadIdx.y, threadIdx.x, 0.0);
        return;
    }

    for (uint8_t n = 1; n <= ADJACENT_CELLS; n++)
        BUF_SET(Sf_shared, blockDim.y, blockDim.x, n-1, threadIdx.y, threadIdx.x, BUF_GET(Sf, r, c, n-1, i, j));

    __syncthreads();

    integer_t This_tile_start_point_x = blockIdx.x * blockDim.x;
    integer_t Next_tile_start_point_x = (blockIdx.x + 1) * blockDim.x;
    integer_t This_tile_start_point_y = blockIdx.y * blockDim.y;
    integer_t Next_tile_start_point_y = (blockIdx.y + 1) * blockDim.y;

    real_t h_next = GET(Sh, c, i, j);

    for (uint8_t n = 1; n <= ADJACENT_CELLS; n++)
        if (   (i+Xi[n]) >= This_tile_start_point_y && (i+Xi[n]) < Next_tile_start_point_y \
            && (j+Xj[n]) >= This_tile_start_point_x && (j+Xj[n]) < Next_tile_start_point_x )
            h_next += BUF_GET(Sf_shared, blockDim.y, blockDim.x, (ADJACENT_CELLS-n), threadIdx.y+Xi[n], threadIdx.x+Xj[n]) \
                    - BUF_GET(Sf_shared, blockDim.y, blockDim.x, (n-1), threadIdx.y, threadIdx.x);
        else
            h_next += BUF_GET(Sf,r, c, (ADJACENT_CELLS-n), i+Xi[n], j+Xj[n]) \
                    - BUF_GET(Sf_shared, blockDim.y, blockDim.x, (n-1), threadIdx.y, threadIdx.x);

    SET(Sh, c, i, j, h_next);
}


class SciddicaTCudaNoHalo: public SciddicaTCuda<SciddicaTCudaNoHalo>
{
    friend class SciddicaTCuda<SciddicaTCudaNoHalo>;

public:
    SciddicaTCudaNoHalo(char **argv): SciddicaTCuda(argv) {}

protected:
    void init_extras()
    {
        Sf = addLayer2D(ADJACENT_CELLS* r, c); // Allocates the Sf substates grid, 
                                               //   having one layer for each adjacent cell
        cudaMemPrefetchAsync(Sf, sizeof(real_t)*r*c*ADJACENT_CELLS, 0 , NULL);

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
        { sciddicaTFlowsComputation<<<grid_size,block_size,shmem_size_fc>>>(i_start, i_end, j_start, j_end, r, c, Sz, Sh, Sf); }
    inline void launch_width_update_kernel()
        { sciddicaTWidthUpdate<<<grid_size,block_size,shmem_size_wu>>>(i_start, i_end, j_start, j_end, r, c, Sh, Sf); }

private:
    real_t *Sf;  // Sf: 4 substates containing the flows towards the 4 neighs
    dim3 grid_size_tiled;
    size_t shmem_size_fc;
    size_t shmem_size_wu;
};


int main(int argc, char **argv)
{
    SciddicaTCudaNoHalo sciddicaT(argv);
    return sciddicaT.execute();
}
