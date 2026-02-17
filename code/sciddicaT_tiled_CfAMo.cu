#include <iostream>
using namespace std;
#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "util.hpp"


// ----------------------------------------------------------------------------
// The adopted von Neuman neighborhood
// Format: flow_index:cell_label:(row_index,col_index)
//
//   cell_label in [0,1,2,3,4]: label assigned to each cell in the neighborhood
//       (row_index,col_index): 2D relative indices of the cells
//
//             |1:(-1, 0)|
//   |2:( 0,-1)|0:( 0, 0)|3:( 0, 1)|
//             |4:( 1, 0)|
//
//
int h_Xi[] = {0, -1,  0,  0,  1};// Xj: von Neuman neighborhood row coordinates (see below)
int h_Xj[] = {0,  0, -1,  1,  0};// Xj: von Neuman neighborhood col coordinates (see below)
__constant__ int Xi[5]; // Xj: von Neuman neighborhood row coordinates (see below)
__constant__ int Xj[5]; // Xj: von Neuman neighborhood col coordinates (see below)
// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
//#define DOMAIN_CHUNK_SIZE_X_ID 6
//#define DOMAIN_CHUNK_SIZE_Y_ID 7
#define BLOCK_SIZE_D0_ID 6
#define BLOCK_SIZE_D1_ID 7
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char* path, int &nrows, int &ncols, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
  FILE* f;
  
  if ( (f = fopen(path,"r") ) == 0){
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  //Reading the header
  char str[STRLEN];
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncols = atoi(str);      //ncols
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrows = atoi(str);      //nrows
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str);  //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str);  //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);   //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);     //NODATA_value 
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

bool saveBinaryGrid2Dr(double *M, int rows, int columns, const char *path)
{
  FILE *f = fopen(path, "w");

  if (!f)
    return false;

  fwrite(M, sizeof(double), rows*columns, f);

  fclose(f);

  return true;
}

double* addLayer2D(int rows, int columns)
{
  //double *tmp = (double *)malloc(sizeof(double) * rows * columns);
  double *tmp;
  cudaMallocManaged(&tmp, sizeof(double) * rows * columns);  
  if (!tmp)
    return NULL;
  return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
void sciddicaTSimulationInit(int i, int j, int r, int c, double* Sz, double* Sh)
{
  double z, h;
  h = GET(Sh, c, i, j);

  if (h > 0.0)
  {
    z = GET(Sz, c, i, j);
    SET(Sz, c, i, j, z - h);
  }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__
void sciddicaTFlowsComputation(int i_start, int i_end, int j_start, int j_end, int r, int c, double nodata, /*int* Xi, int* Xj,*/ const double* __restrict__ Sz, const double* __restrict__ Sh, double* __restrict__ Sh_next, const double p_r, const double p_epsilon)
{
  dim3 tileDim(blockDim.x - 2, blockDim.y - 2, 1);  // I remove the 3 halo cells from blockDim to obtain tileDim, i.e., the size of the domain partitions.
  int i = blockIdx.y * tileDim.y + threadIdx.y - 1; // thread 0 maps the first halo cell, thread 1 the first domain partition cell, ...
  int j = blockIdx.x * tileDim.x + threadIdx.x - 1;
  int ti = threadIdx.y ; // local indexes
  int tj = threadIdx.x ; //
                                                     // domain_size = 16 |++++------------| ("-" = domain cells; "+" = cells to be updated)
                                                     // blk_0           |.****.|            ("*" = threads updating domain cells; "." = threads computing average)
                                                     // blk_0 index i    012345
                                                     //  
                                                     // domain_size = 16 |----++++--------| ("-" = domain cells; "+" = cells to be updated)
                                                     // blk_1               |.****.|        ("*" = threads updating domain cells; "." = threads computing average)
                                                     // blk_1 index i        012345
                                                     //

  extern __shared__ uint8_t S[];
  double* sh_next_shared = (double*) S;

  SET(sh_next_shared, blockDim.x, threadIdx.y, threadIdx.x, 0.0);

  if (i < i_start or i >= i_end or j < j_start or j >= j_end)
    return;

  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  m = GET(Sh, c, i, j) - p_epsilon;
  if (m <= 0.0)
  {
    bool at_least_one = false;
    for(int n = 1; n<5;++n){
      int i_n = i+Xi[n];
      int j_n = j+Xj[n];
      if(not (i_n < 0 or i_n >= r or j_n < 0 or j_n >= c))
        if(GET(Sh, c, i_n, j_n) - p_epsilon > 0)
          at_least_one=true;
    }
    if(not at_least_one)
      return;
  }

  u[0] = GET(Sz, c, i, j) + p_epsilon;
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

    for (n = 0; n < 5; n++)
      if (!eliminated_cells[n])
      // if ( !(elim_cells & (1 << n)) ) 
      {
        average += u[n];
        cells_count++;
      }

    if (cells_count != 0)
      average /= cells_count;

    for (n = 0; n < 5; n++)
      if ((average <= u[n]) && (!eliminated_cells[n]))
      {
        eliminated_cells[n] = true; 
        again = true;
      }
  } while (again);

  double f_out = 0.0;
  double f_out_sum = 0.0;
  for (int n = 1; n < 5; n++)
  {
    // Shifting local indexes
    int ti_n =  threadIdx.y + Xi[n] ;
    int tj_n =  threadIdx.x + Xj[n] ;
  
    if(!(ti_n < 0 or ti_n >= blockDim.y or tj_n < 0 or tj_n >= blockDim.x)){
      if (!eliminated_cells[n])
      {
        f_out = (average - u[n]) * p_r;
        f_out_sum += f_out ;
        SET(sh_next_shared, blockDim.x, ti_n, tj_n, (GET(sh_next_shared, blockDim.x, ti_n, tj_n) + f_out));
      }
    }
    __syncthreads();
  }

  // Only in tile threads
  if(ti > 0 and ti < tileDim.y + 1 and  tj > 0 and tj < tileDim.x + 1){
    double f_in_sum = GET(sh_next_shared, blockDim.x, ti, tj);
    SET(Sh_next, c, i, j,GET(Sh_next, c, i, j) + f_in_sum - f_out_sum );
  }

}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows;                  // r: grid rows
  int c = cols;                  // c: grid columns
  int i_start = 2, i_end = r-2;  // [i_start,i_end[: kernels application range along the rows
  int j_start = 2, j_end = c-2;  // [i_start,i_end[: kernels application range along the rows
  double *Sz;                    // Sz: substate (grid) containing the cells' altitude a.s.l.
  double *Sh;                    // Sh: substate (grid) containing the cells' flow thickness
  double *Sh_next;               // Sh_next: substate (grid) containing the updated cells' flow thickness
  double p_r = P_R;              // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;  // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); //steps: simulation steps
  //dim3 domain_chunk_size(atoi(argv[DOMAIN_CHUNK_SIZE_X_ID]), atoi(argv[DOMAIN_CHUNK_SIZE_Y_ID]), 1);
  int block_size_d0 = atoi(argv[BLOCK_SIZE_D0_ID]);
  int block_size_d1 = atoi(argv[BLOCK_SIZE_D1_ID]);


  Sz      = addLayer2D(r, c); // Allocates the Sz substate grid
  Sh      = addLayer2D(r, c); // Allocates the Sh substate grid
  Sh_next = addLayer2D(r, c); // Allocates the Sh_next substate grid

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);   // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);// Load Sh from file

  // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        sciddicaTSimulationInit(i, j, r, c, Sz, Sh);
  memcpy(Sh_next, Sh, sizeof(double) * r * c);

  cudaMemcpyToSymbol(Xi, h_Xi, 5*sizeof(int)); // Copy Xi to DEVICE in CONSTANT memory
  cudaMemcpyToSymbol(Xj, h_Xj, 5*sizeof(int)); // Copy Xj to DEVICE in CONSTANT memory

  cudaMemPrefetchAsync(Sz,      sizeof(double)*r*c, 0 , NULL);
  cudaMemPrefetchAsync(Sh,      sizeof(double)*r*c, 0 , NULL);
  cudaMemPrefetchAsync(Sh_next, sizeof(double)*r*c, 0 , NULL);
  cudaDeviceSynchronize();

  dim3 block_size(block_size_d0, block_size_d1, 1);
  dim3 grid_size(ceil(c/(float)(block_size.x - 2)), ceil(r/(float)(block_size.y - 2)), 1);
  int shmem_size = block_size.x * block_size.y * sizeof(double);

  util::Timer cl_timer;
  // simulation loop
  for (int s = 0; s < steps; ++s)
  {
    sciddicaTFlowsComputation<<<grid_size, block_size, shmem_size>>>(i_start, i_end, j_start, j_end, r, c, nodata, /*Xi, Xj,*/ Sz, Sh, Sh_next, p_r, p_epsilon);
    fflush (stdout);
    cudaMemcpy(Sh, Sh_next, sizeof(double)*r*c, cudaMemcpyDeviceToDevice);
  }
  cudaDeviceSynchronize();
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf(" %2d; %2d; %7.3f\n", block_size_d0, block_size_d1, cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);// Save Sh to file
  const std::string binPath = std::string(argv[OUTPUT_PATH_ID]) + ".bin";
  saveBinaryGrid2Dr(Sh, r, c, binPath.c_str());// Save Sh to file in binary format

  //printf("Releasing memory...\n");
  cudaFree(Sz);
  cudaFree(Sh);
  cudaFree(Sh_next);

  return 0;
}
