#ifndef SCIDDICA_T_H
#define SCIDDICA_T_H

#include "types.h"


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
int8_t h_Xi[] = {0, -1,  0,  0,  1};// Xj: von Neuman neighborhood row coordinates (see below)
int8_t h_Xj[] = {0,  0, -1,  1,  0};// Xj: von Neuman neighborhood col coordinates (see below)
__constant__ int8_t Xi[5]; // Xj: von Neuman neighborhood row coordinates (see below)
__constant__ int8_t Xj[5]; // Xj: von Neuman neighborhood col coordinates (see below)


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
#ifdef DOUBLE_PRECISION
    #define P_R 0.5                 // minimization algorithm outflows dumping factor
    #define P_EPSILON 0.001         // frictional parameter threshold
#else
    #define P_R 0.5f
    #define P_EPSILON 0.001f
#endif
#define ADJACENT_CELLS 4
#define NEIGBORHOOD_SIZE 5
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
void readHeaderInfo(char* path, integer_t &nrows, integer_t &ncols, real_t &nodata)
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

bool loadGrid2D(real_t *M, integer_t rows, integer_t columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (integer_t i = 0; i < rows; i++)
    for (integer_t j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(real_t *M, integer_t rows, integer_t columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (integer_t i = 0; i < rows; i++)
  {
    for (integer_t j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

bool saveBinaryGrid2Dr(real_t *M, integer_t rows, integer_t columns, const char *path)
{
  FILE *f = fopen(path, "w");

  if (!f)
    return false;

  fwrite(M, sizeof(real_t), rows*columns, f);

  fclose(f);

  return true;
}

real_t* addLayer2D(integer_t rows, integer_t columns)
{
  //double *tmp = (double *)malloc(sizeof(double) * rows * columns);
  real_t *tmp;
  cudaMallocManaged(&tmp, sizeof(real_t) * rows * columns);  
  if (!tmp)
    return NULL;
  return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
void sciddicaTSimulationInit(const integer_t i, const integer_t j, const integer_t c,
                             real_t* const Sz, real_t* const Sh)
{
  real_t z, h;
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
void sciddicaTResetFlows(const integer_t i_start, const integer_t i_end, const integer_t j_start, const integer_t j_end, const integer_t r, const integer_t c,
                         real_t* const __restrict__ Sf)
{
  integer_t i = blockIdx.y * blockDim.y + threadIdx.y;
  integer_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i_start or i >= i_end or j < j_start or j >= j_end)
    return;

  BUF_SET(Sf, r, c, 0, i, j, 0.0);
  BUF_SET(Sf, r, c, 1, i, j, 0.0);
  BUF_SET(Sf, r, c, 2, i, j, 0.0);
  BUF_SET(Sf, r, c, 3, i, j, 0.0);
}


// ----------------------------------------------------------------------------
// Super class for running a generic (variant independent) sciddicaT cuda application:
//    * common code to all variants is factorized
//    * variant-specific operations and kernel launches are implemnted by
//      subclasses (one for each variant)
// ----------------------------------------------------------------------------
template< class DerivedSciddicaTCuda >
class SciddicaTCuda
{
    #define derivedSciddicaTCuda static_cast<DerivedSciddicaTCuda&>(*this)

public:
    SciddicaTCuda(char **argv):
        steps ( atoi(argv[STEPS_ID]) ),
        block_size_d0 ( atoi(argv[BLOCK_SIZE_D0_ID]) ),
        block_size_d1 ( atoi(argv[BLOCK_SIZE_D1_ID]) ),
        output_path ( argv[OUTPUT_PATH_ID] )
        {
            readHeaderInfo(argv[HEADER_PATH_ID], r, c, nodata);
            i_start = 1;
            i_end = r-1;
            j_start = 1;
            j_end = c-1;

            Sz = addLayer2D(r, c);                 // Allocates the Sz substate grid
            Sh = addLayer2D(r, c);                 // Allocates the Sh substate grid
            
            loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);   // Load Sz from file
            loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);// Load Sh from file
        }
    
    int execute()
    {
        // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
        #pragma omp parallel for
        for (integer_t i = i_start; i < i_end; i++)
            for (integer_t j = j_start; j < j_end; j++)
                sciddicaTSimulationInit(i, j, c, Sz, Sh);

        cudaMemcpyToSymbol(Xi, h_Xi, 5*sizeof(int8_t)); // Copy Xi to DEVICE in CONSTANT memory
        cudaMemcpyToSymbol(Xj, h_Xj, 5*sizeof(int8_t)); // Copy Xj to DEVICE in CONSTANT memory

        util::init_dim3( block_size, block_size_d0, block_size_d1, 1 );
        util::init_dim3( grid_size, ceil(c/(float)block_size.x), ceil(r/(float)block_size.y), 1 );

        derivedSciddicaTCuda.init_extras();

        cudaMemPrefetchAsync(Sz, sizeof(real_t)*r*c, 0 , NULL);
        cudaMemPrefetchAsync(Sh, sizeof(real_t)*r*c, 0 , NULL);
        cudaDeviceSynchronize();
        
        util::Timer cl_timer;
        // simulation loop
        for (int s = 0; s < steps; ++s)
        {
            derivedSciddicaTCuda.on_step_start();
            derivedSciddicaTCuda.launch_flows_computation_kernel();
            derivedSciddicaTCuda.launch_width_update_kernel();
            derivedSciddicaTCuda.on_step_end();
        }
        cudaDeviceSynchronize();
        double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
        printf(" %2d; %2d; %7.3f\n", block_size_d0, block_size_d1, cl_time);

        saveGrid2Dr(Sh, r, c, output_path);// Save Sh to file
        const std::string binPath = std::string(output_path) + ".bin";
        saveBinaryGrid2Dr(Sh, r, c, binPath.c_str());// Save Sh to file in binary format

        //printf("Releasing memory...\n");
        cudaFree(Sz);
        cudaFree(Sh);
        derivedSciddicaTCuda.free_extras();

        return 0;
    }

protected:
    void init_extras() {}
    void free_extras() {}
    inline void on_step_start() {}
    inline void launch_flows_computation_kernel() {}
    inline void launch_width_update_kernel() {}
    inline void on_step_end() {}

    integer_t  r, c;  // grid rows and columns
    integer_t i_start, i_end;  // [i_start,i_end[: kernels application range along the rows
    integer_t j_start, j_end;  // [i_start,i_end[: kernels application range along the rows
    real_t     nodata;
    
    real_t *Sz;  // Sz: substate (grid) containing the cells' altitude a.s.l.
    real_t *Sh;  // Sh: substate (grid) containing the cells' flow thickness
    
    int steps;  //steps: simulation steps
    
    integer_t  block_size_d0, block_size_d1;
    
    dim3 block_size;
    dim3 grid_size;

    char *output_path;
    
    #undef derivedSciddicaTCuda
};


#endif