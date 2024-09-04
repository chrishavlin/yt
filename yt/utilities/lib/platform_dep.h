#include <math.h>
#ifdef MS_WIN32
#include "malloc.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#elif defined(__FreeBSD__)
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#else
#include <stdint.h>
#include "alloca.h"
#include <math.h>
#endif
