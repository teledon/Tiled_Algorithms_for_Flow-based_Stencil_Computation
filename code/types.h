#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

//
// data type for real-valued variables
//
#ifdef DOUBLE_PRECISION
    typedef double real_t;
#else
    typedef float real_t;
#endif

//
// integer type with values within the domain dimensional range (for sizes and indices)
//
typedef  int16_t integer_t;
typedef uint16_t natural_t;

#endif