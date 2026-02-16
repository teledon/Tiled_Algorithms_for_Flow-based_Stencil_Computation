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
//   flow_index in   [0,1,2,3]: outgoing flow indices in Sf from cell 0 to the others
//       (row_index,col_index): 2D relative indices of the cells
//
//               |0:1:(-1, 0)|
//   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
//               |3:4:( 1, 0)|
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
void sciddicaTResetFlows(int i_start, int i_end, int j_start, int j_end, int r, int c, double nodata, double* __restrict__ Sf)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i_start or i >= i_end or j < j_start or j >= j_end)
    return;

  BUF_SET(Sf, r, c, 0, i, j, 0.0);
  BUF_SET(Sf, r, c, 1, i, j, 0.0);
  BUF_SET(Sf, r, c, 2, i, j, 0.0);
  BUF_SET(Sf, r, c, 3, i, j, 0.0);
}

__global__
void sciddicaTFlowsComputation(int i_start, int i_end, int j_start, int j_end, int r, int c, double nodata, /*int* Xi, int* Xj,*/ const double* __restrict__ Sz, const double* __restrict__ Sh, double* __restrict__ Sf, const double p_r, const double p_epsilon)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i_start or i >= i_end or j < j_start or j >= j_end)
    return;

  //int Xi[] = {0, -1,  0,  0,  1};// Xj: von Neuman neighborhood row coordinates (see below)
  //int Xj[] = {0,  0, -1,  1,  0};// Xj: von Neuman neighborhood col coordinates (see below)

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
    return;

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

  if (!eliminated_cells[1]) BUF_SET(Sf, r, c, 0, i, j, (average - u[1]) * p_r);
  if (!eliminated_cells[2]) BUF_SET(Sf, r, c, 1, i, j, (average - u[2]) * p_r);
  if (!eliminated_cells[3]) BUF_SET(Sf, r, c, 2, i, j, (average - u[3]) * p_r);
  if (!eliminated_cells[4]) BUF_SET(Sf, r, c, 3, i, j, (average - u[4]) * p_r);
}

__global__
void sciddicaTWidthUpdate(int i_start, int i_end, int j_start, int j_end, int r, int c, double nodata, /*int* Xi, int* Xj,*/ const double* __restrict__ Sz, double* __restrict__ Sh, const double* __restrict__ Sf)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  //int Xi[] = {0, -1,  0,  0,  1};// Xj: von Neuman neighborhood row coordinates (see below)
  //int Xj[] = {0,  0, -1,  1,  0};// Xj: von Neuman neighborhood col coordinates (see below)

  if (i < i_start or i >= i_end or j < j_start or j >= j_end)
    return;

  double h_next;
  h_next = GET(Sh, c, i, j);
  h_next += BUF_GET(Sf, r, c, 3, i+Xi[1], j+Xj[1]) - BUF_GET(Sf, r, c, 0, i, j);
  h_next += BUF_GET(Sf, r, c, 2, i+Xi[2], j+Xj[2]) - BUF_GET(Sf, r, c, 1, i, j);
  h_next += BUF_GET(Sf, r, c, 1, i+Xi[3], j+Xj[3]) - BUF_GET(Sf, r, c, 2, i, j);
  h_next += BUF_GET(Sf, r, c, 0, i+Xi[4], j+Xj[4]) - BUF_GET(Sf, r, c, 3, i, j);

  SET(Sh, c, i, j, h_next);
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
  int i_start = 1, i_end = r-1;  // [i_start,i_end[: kernels application range along the rows
  int j_start = 1, j_end = c-1;  // [i_start,i_end[: kernels application range along the rows
  double *Sz;                    // Sz: substate (grid) containing the cells' altitude a.s.l.
  double *Sh;                    // Sh: substate (grid) containing the cells' flow thickness
  double *Sf;                    // Sf: 4 substates containing the flows towards the 4 neighs
  double p_r = P_R;              // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;  // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); //steps: simulation steps
  int block_size_d0 = atoi(argv[BLOCK_SIZE_D0_ID]);
  int block_size_d1 = atoi(argv[BLOCK_SIZE_D1_ID]);

  Sz = addLayer2D(r, c);                 // Allocates the Sz substate grid
  Sh = addLayer2D(r, c);                 // Allocates the Sh substate grid
  Sf = addLayer2D(ADJACENT_CELLS* r, c); // Allocates the Sf substates grid, 
                                         //   having one layer for each adjacent cell

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);   // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);// Load Sh from file

  // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        sciddicaTSimulationInit(i, j, r, c, Sz, Sh);

  cudaMemcpyToSymbol(Xi, h_Xi, 5*sizeof(int)); // Copy Xi to DEVICE in CONSTANT memory
  cudaMemcpyToSymbol(Xj, h_Xj, 5*sizeof(int)); // Copy Xj to DEVICE in CONSTANT memory

  cudaMemPrefetchAsync(Sz, sizeof(double)*r*c, 0 , NULL);
  cudaMemPrefetchAsync(Sh, sizeof(double)*r*c, 0 , NULL);
  cudaMemPrefetchAsync(Sf, sizeof(double)*r*c*ADJACENT_CELLS, 0 , NULL);
  cudaDeviceSynchronize();

  util::Timer cl_timer;
  // simulation loop
  for (int s = 0; s < steps; ++s)
  {
    dim3 block_size_rf(block_size_d0, block_size_d1, 1);
    dim3 grid_size_rf(ceil(c/(float)block_size_rf.x), ceil(r/(float)block_size_rf.y), 1);
    sciddicaTResetFlows<<<grid_size_rf,block_size_rf>>>(i_start, i_end, j_start, j_end, r, c, nodata, Sf);

    dim3 block_size_fc(block_size_d0, block_size_d1, 1);
    dim3 grid_size_fc(ceil(c/(float)block_size_fc.x), ceil(r/(float)block_size_fc.y), 1);
    sciddicaTFlowsComputation<<<grid_size_fc,block_size_fc>>>(i_start, i_end, j_start, j_end, r, c, nodata, /*Xi, Xj,*/ Sz, Sh, Sf, p_r, p_epsilon);

    dim3 block_size_wu(block_size_d0, block_size_d1, 1);
    dim3 grid_size_wu(ceil(c/(float)block_size_wu.x), ceil(r/(float)block_size_wu.y), 1);
    sciddicaTWidthUpdate<<<grid_size_wu,block_size_wu>>>(i_start, i_end, j_start, j_end, r, c, nodata, /*Xi, Xj,*/ Sz, Sh, Sf);
  }
  cudaDeviceSynchronize();
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf(" %2d; %2d; %7.3f\n", block_size_d0, block_size_d1, cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);// Save Sh to file
  const std::string binPath = std::string(argv[OUTPUT_PATH_ID]) + ".bin";
  saveBinaryGrid2Dr(Sh, r, c, binPath.c_str());// Save Sh to file in binary format

  //printf("Releasing memory...\n");
  // delete[] Sz;
  // delete[] Sh;
  // delete[] Sf;
  cudaFree(Sz);
  cudaFree(Sh);
  cudaFree(Sf);

  return 0;
}
