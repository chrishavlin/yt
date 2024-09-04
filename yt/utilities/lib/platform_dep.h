#include <math.h>
#ifdef MS_WIN32
#include "malloc.h"
#include <float.h>
/* minimum _MSC_VER >= 1928 (VS2019) */
#include <stdint.h>
#elif defined(__FreeBSD__)
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#else
#include <stdint.h>
#include "alloca.h"
#include <math.h>
#endif
