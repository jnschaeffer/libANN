#ifndef __ANN_MAIN_H__
#define __ANN_MAIN_H__

#ifdef __cplusplus
# define BEGIN_C_DECLS extern "C" {
# define END_C_DECLS }
#else
# define BEGIN_C_DECLS
# define END_C_DECLS
#endif

#include "ANN_config.h"
#
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#
#ifdef HAVE_STDIO_H 
# include <stdio.h>
#endif
#
#ifdef HAVE_TIME_H
# include <time.h>
#endif
#
#ifdef HAVE_MATH_H
# include <math.h>
#endif
#
#ifndef TRUE
# define TRUE 1
#endif
#
#ifndef FALSE
# define FALSE 0
#endif

//Uncomment these to debug the appropriate areas
//#define _DEBUG_FFNET
//#define _DEBUG_LAYER

#endif
