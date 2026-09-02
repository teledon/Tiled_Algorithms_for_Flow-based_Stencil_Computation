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
                               const real_t* const __restrict__ Sz, const real_t* const __restrict__ Sh, real_t* const __restrict__ Sh_next)
{
    dim3 tileDim(blockDim.x - 2, blockDim.y - 2, 1);  // I remove the 3 halo cells from blockDim to obtain tileDim, i.e., the size of the domain partitions.
    integer_t i = blockIdx.y * tileDim.y + threadIdx.y - 1; // thread 0 maps the first halo cell, thread 1 the first domain partition cell, ...
    integer_t j = blockIdx.x * tileDim.x + threadIdx.x - 1;
    #define ti threadIdx.y // local indexes
    #define tj threadIdx.x //
                                                        // domain_size = 16 |++++------------| ("-" = domain cells; "+" = cells to be updated)
                                                        // blk_0           |.****.|            ("*" = threads updating domain cells; "." = threads computing average)
                                                        // blk_0 index i    012345
                                                        //  
                                                        // domain_size = 16 |----++++--------| ("-" = domain cells; "+" = cells to be updated)
                                                        // blk_1               |.****.|        ("*" = threads updating domain cells; "." = threads computing average)
                                                        // blk_1 index i        012345
                                                        //

    extern __shared__ real_t sh_next_shared[];

    SET(sh_next_shared, blockDim.x, threadIdx.y, threadIdx.x, 0.0);

    if (i < i_start or i >= i_end or j < j_start or j >= j_end)
        return;

    bool eliminated_cells[NEIGBORHOOD_SIZE] = {false, false, false, false, false};
    bool again;
    uint8_t cells_count;
    real_t average;
    real_t m;
    real_t u[5];
    real_t z, h;

    m = GET(Sh, c, i, j) - P_EPSILON;
    if (m <= 0.0)
    {
        bool at_least_one = false;
        for (uint8_t n = 1; n<NEIGBORHOOD_SIZE;++n)
        {
            integer_t i_n = i+Xi[n];
            integer_t j_n = j+Xj[n];
            if (not (i_n < 0 or i_n >= r or j_n < 0 or j_n >= c))
                if (GET(Sh, c, i_n, j_n) - P_EPSILON > 0)
                    at_least_one=true;
        }
        if(not at_least_one)
            return;
    }

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

    real_t f_out = 0.0;
    real_t f_out_sum = 0.0;
    for (uint8_t n = 1; n < NEIGBORHOOD_SIZE; n++)
    {
        // Shifting local indexes
        integer_t ti_n =  threadIdx.y + Xi[n] ;
        integer_t tj_n =  threadIdx.x + Xj[n] ;

        if (!(ti_n < 0 or ti_n >= blockDim.y or tj_n < 0 or tj_n >= blockDim.x))
        {
            if (!eliminated_cells[n])
            {
                f_out = (average - u[n]) * P_R;
                f_out_sum += f_out ;
                SET(sh_next_shared, blockDim.x, ti_n, tj_n, (GET(sh_next_shared, blockDim.x, ti_n, tj_n) + f_out));
            }
        }
        __syncthreads();
    }

    // Only in tile threads
    if (ti > 0 and ti < tileDim.y + 1 and  tj > 0 and tj < tileDim.x + 1)
    {
        real_t f_in_sum = GET(sh_next_shared, blockDim.x, ti, tj);
        SET(Sh_next, c, i, j,GET(Sh_next, c, i, j) + f_in_sum - f_out_sum );
    }

    #undef ti
    #undef tj
}


class SciddicaTCudaCfAMo: public SciddicaTCuda<SciddicaTCudaCfAMo>
{
    friend class SciddicaTCuda<SciddicaTCudaCfAMo>;

public:
    SciddicaTCudaCfAMo(char **argv): SciddicaTCuda(argv) {}

protected:
    void init_extras()
    {
        Sh_next = addLayer2D(r, c);
        memcpy(Sh_next, Sh, sizeof(real_t) * r * c);
        cudaMemPrefetchAsync(Sh_next, sizeof(real_t)*r*c, 0 , NULL);
        
        util::init_dim3( grid_size_tiled, ceil(c/(float)(block_size.x - 2)), ceil(r/(float)(block_size.y - 2)), 1 );
        shmem_size = block_size.x * block_size.y * sizeof(real_t);
    }

    void free_extras()
        { cudaFree(Sh_next); }

    inline void launch_flows_computation_kernel()
        { sciddicaTFlowsComputation<<<grid_size_tiled, block_size, shmem_size>>>(i_start, i_end, j_start, j_end, r, c, Sz, Sh, Sh_next); }
    
    inline void on_step_end()
        { cudaMemcpy(Sh, Sh_next, sizeof(double)*r*c, cudaMemcpyDeviceToDevice); }

private:
    real_t *Sh_next;
    dim3 grid_size_tiled;
    size_t shmem_size;
};


int main(int argc, char **argv)
{
    SciddicaTCudaCfAMo sciddicaT(argv);
    return sciddicaT.execute();
}
