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
#include "bitwise.hpp"


__global__
void sciddicaTFlowsComputation(const integer_t i_start, const integer_t i_end, const integer_t j_start, const integer_t j_end,
                               const integer_t r, const integer_t c,
                               const real_t* const __restrict__ Sz, const real_t* const __restrict__ Sh, real_t* const __restrict__ Sh_next)
{
    #define tileDim_x (blockDim.x - 2)  // I remove the 3 halo cells from blockDim to obtain tileDim, i.e., the size of the domain partitions.
    #define tileDim_y (blockDim.y - 2)
    #define ti threadIdx.y // local indexes
    #define tj threadIdx.x //

    integer_t i = blockIdx.y * tileDim_y + threadIdx.y - 1; // thread 0 maps the first halo cell, thread 1 the first domain partition cell, ...
    integer_t j = blockIdx.x * tileDim_x + threadIdx.x - 1;
    
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

    bool in_domain = i >= i_start && i < i_end && j >= j_start && j < j_end;
    uint8_t eliminated_cells = 0;  // all neighbours NOT eliminated by default
    real_t h_next;
    real_t m;
    real_t average;
    real_t u[5];

    if ( in_domain )
    {
        
        bool again;
        uint8_t cells_count;
        real_t z, h;

        h_next = GET(Sh, c, i, j);
        m = h_next - P_EPSILON;
        if (m > 0.0)
        {
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
                    if (!BIT_GET(eliminated_cells, n))
                    {
                        average += u[n];
                        cells_count++;
                    }

                if (cells_count != 0)
                    average /= cells_count;

                for (uint8_t n = 0; n < NEIGBORHOOD_SIZE; n++)
                    if ((average <= u[n]) && (!BIT_GET(eliminated_cells, n)))
                    {
                        BIT_SET(eliminated_cells, n); 
                        again = true;
                    }
            } while (again);

        }
    }

    for (uint8_t n = 1; n < NEIGBORHOOD_SIZE; n++)
    {
        if ( in_domain && m > 0.0 )
        {
            // Shifting local indexes
            integer_t ti_n =  threadIdx.y + Xi[n] ;
            integer_t tj_n =  threadIdx.x + Xj[n] ;

            if (!(ti_n < 0 or ti_n >= blockDim.y or tj_n < 0 or tj_n >= blockDim.x))
            {
                if (!BIT_GET(eliminated_cells, n))
                {
                    real_t f_out = (average - u[n]) * P_R;
                    h_next -= f_out;
                    SET(sh_next_shared, blockDim.x, ti_n, tj_n, (GET(sh_next_shared, blockDim.x, ti_n, tj_n) + f_out));
                }
            }   
        }
        __syncthreads();
    }

    // Only in tile threads
    if (in_domain && ti > 0 and ti < tileDim_y + 1 and  tj > 0 and tj < tileDim_x + 1)
    {
        h_next += GET(sh_next_shared, blockDim.x, ti, tj);
        SET(Sh_next, c, i, j, h_next);
    }

    #undef tileDim_x
    #undef tileDim_y
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
        memPrefetch(Sh_next, sizeof(real_t)*r*c);
        
        util::init_dim3( grid_size_tiled, ceil(c/(float)(block_size.x - 2)), ceil(r/(float)(block_size.y - 2)), 1 );
        shmem_size = block_size.x * block_size.y * sizeof(real_t);

        if (!util::checkShmemSizePerBlock(shmem_size))
            exit(EXIT_FAILURE);
    }

    void free_extras()
        { cudaFree(Sh_next); }

    inline void launch_flows_computation_kernel()
        { sciddicaTFlowsComputation<<<grid_size_tiled, block_size, shmem_size>>>(i_start, i_end, j_start, j_end, r, c, Sz, Sh, Sh_next); }
    
    inline void on_step_end()
        { swap(Sh, Sh_next); }

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
