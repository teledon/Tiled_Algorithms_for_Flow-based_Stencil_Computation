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
void sciddicaTFlowsComputation(const integer_t i_start, const integer_t i_end, const integer_t j_start, const integer_t j_end, const integer_t r, const integer_t c,
                               const real_t* const __restrict__ Sz, const real_t* const __restrict__ Sh, real_t* const __restrict__ Sf)
{
    integer_t i = blockIdx.y * blockDim.y + threadIdx.y;
    integer_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < i_start or i >= i_end or j < j_start or j >= j_end)
        return;

    bool eliminated_cells[NEIGBORHOOD_SIZE] = {false, false, false, false, false};
    bool again;
    real_t average;
    real_t m;
    real_t u[5];
    real_t z, h;

    m = GET(Sh, c, i, j) - P_EPSILON;
    if (m <= 0.0)
        return;

    u[0] = GET(Sz, c, i, j) + P_EPSILON;
    z = GET(Sz, c, i + Xi[1], j + Xj[1]);
    h = GET(Sh, c, i + Xi[1], j + Xj[1]);
    u[1] = z + h;                                         
    z = GET(Sz, c, i + Xi[2], j + Xj[2]);
    h = GET(Sh, c, i + Xi[2], j + Xj[2]);
    u[2] = z + h;                                         
    z = GET(Sz, c, i + Xi[3], j + Xj[3]);
    h = GET(Sh, c, i + Xi[3], j + Xj[3]);
    u[3] = z + h;                                         
    z = GET(Sz, c, i + Xi[4], j + Xj[4]);
    h = GET(Sh, c, i + Xi[4], j + Xj[4]);
    u[4] = z + h;

    do
    {
        again = false;
        average = m;
        uint8_t cells_count = 0;

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
void sciddicaTWidthUpdate(const integer_t i_start, const integer_t i_end, const integer_t j_start, const integer_t j_end, const integer_t r, const integer_t c,
                          real_t* const __restrict__ Sh, const real_t* const __restrict__ Sf)
{
    integer_t i = blockIdx.y * blockDim.y + threadIdx.y;
    integer_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < i_start or i >= i_end or j < j_start or j >= j_end)
        return;

    real_t h_next;
    h_next = GET(Sh, c, i, j);
    h_next += BUF_GET(Sf, r, c, 3, i+Xi[1], j+Xj[1]) - BUF_GET(Sf, r, c, 0, i, j);
    h_next += BUF_GET(Sf, r, c, 2, i+Xi[2], j+Xj[2]) - BUF_GET(Sf, r, c, 1, i, j);
    h_next += BUF_GET(Sf, r, c, 1, i+Xi[3], j+Xj[3]) - BUF_GET(Sf, r, c, 2, i, j);
    h_next += BUF_GET(Sf, r, c, 0, i+Xi[4], j+Xj[4]) - BUF_GET(Sf, r, c, 3, i, j);

    SET(Sh, c, i, j, h_next);
}


class SciddicaTCudaNaive: public SciddicaTCuda<SciddicaTCudaNaive>
{
    friend class SciddicaTCuda<SciddicaTCudaNaive>;

public:
    SciddicaTCudaNaive(char **argv): SciddicaTCuda(argv) {}

protected:
    void init_extras()
    {
        Sf = addLayer2D(ADJACENT_CELLS* r, c); // Allocates the Sf substates grid, 
                                               //   having one layer for each adjacent cell
        cudaMemPrefetchAsync(Sf, sizeof(real_t)*r*c*ADJACENT_CELLS, 0 , NULL);
    }
    
    void free_extras()
        { cudaFree(Sf); }

    inline void on_step_start()
        { sciddicaTResetFlows<<<grid_size,block_size>>>(i_start, i_end, j_start, j_end, r, c, Sf); }
    inline void launch_flows_computation_kernel()
        { sciddicaTFlowsComputation<<<grid_size,block_size>>>(i_start, i_end, j_start, j_end, r, c, Sz, Sh, Sf); }
    inline void launch_width_update_kernel()
        { sciddicaTWidthUpdate<<<grid_size,block_size>>>(i_start, i_end, j_start, j_end, r, c, Sh, Sf); }

private:
    real_t *Sf;  // Sf: 4 substates containing the flows towards the 4 neighs
};


int main(int argc, char **argv)
{
    SciddicaTCudaNaive sciddicaT(argv);
    return sciddicaT.execute();
}
