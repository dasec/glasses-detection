#ifndef BSIF_H
#define BSIF_H

/*
 * bsif.h
 *
 * Author: F.Struck (florian.struck@cased.de), C. Rathgeb (crathgeb@cosy.sbg.ac.at), P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Generates an iris code from iris texture using the BSIF algorithm
 * BSIF filters can be obtained from http://www.ee.oulu.fi/~jkannala/bsif/
 */
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <fstream>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stddef.h>
#include <matio.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iomanip>
#include <time.h>

using namespace std;
using namespace cv;

/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 */

/*=======================================================================*
 * Target hardware information
 *   Device type: Generic->MATLAB Host Computer
 *   Number of bits:     char:   8    short:   16    int:  32
 *                       long:  64    long long:  64
 *                       native word size:  64
 *   Byte ordering: LittleEndian
 *   Signed integer division rounds to: Undefined
 *   Shift right on a signed integer as arithmetic shift: on
 *=======================================================================*/

/*=======================================================================*
 * Fixed width word size data types:                                     *
 *   int8_T, int16_T, int32_T     - signed 8, 16, or 32 bit integers     *
 *   uint8_T, uint16_T, uint32_T  - unsigned 8, 16, or 32 bit integers   *
 *   real32_T, real64_T           - 32 and 64 bit floating point numbers *
 *=======================================================================*/
typedef signed char int8_T;
typedef unsigned char uint8_T;
typedef short int16_T;
typedef unsigned short uint16_T;
typedef int int32_T;
typedef unsigned int uint32_T;
typedef long int64_T;
typedef unsigned long uint64_T;
typedef float real32_T;
typedef double real64_T;

/*===========================================================================*
 * Generic type definitions: real_T, time_T, boolean_T, int_T, uint_T,       *
 *                           ulong_T, ulonglong_T, char_T and byte_T.        *
 *===========================================================================*/
typedef double real_T;
typedef double time_T;
typedef unsigned char boolean_T;
typedef int int_T;
typedef unsigned int uint_T;
typedef unsigned long ulong_T;
typedef unsigned long long ulonglong_T;
typedef char char_T;
typedef char_T byte_T;

/*===========================================================================*
 * Complex number type definitions                                           *
 *===========================================================================*/

struct BsifCell {
    int startX, startY;
    int endX, endY;
    int width, height;
};

typedef struct {
    real32_T re;
    real32_T im;
} creal32_T;

typedef struct {
    real64_T re;
    real64_T im;
} creal64_T;

typedef struct {
    real_T re;
    real_T im;
} creal_T;

typedef struct {
    int8_T re;
    int8_T im;
} cint8_T;

typedef struct {
    uint8_T re;
    uint8_T im;
} cuint8_T;

typedef struct {
    int16_T re;
    int16_T im;
} cint16_T;

typedef struct {
    uint16_T re;
    uint16_T im;
} cuint16_T;

typedef struct {
    int32_T re;
    int32_T im;
} cint32_T;

typedef struct {
    uint32_T re;
    uint32_T im;
} cuint32_T;

typedef struct {
    int64_T re;
    int64_T im;
} cint64_T;

typedef struct {
    uint64_T re;
    uint64_T im;
} cuint64_T;

/*=======================================================================*
 * Min and Max:                                                          *
 *   int8_T, int16_T, int32_T     - signed 8, 16, or 32 bit integers     *
 *   uint8_T, uint16_T, uint32_T  - unsigned 8, 16, or 32 bit integers   *
 *=======================================================================*/
#define MAX_int8_T                     ((int8_T)(127))
#define MIN_int8_T                     ((int8_T)(-128))
#define MAX_uint8_T                    ((uint8_T)(255))
#define MIN_uint8_T                    ((uint8_T)(0))
#define MAX_int16_T                    ((int16_T)(32767))
#define MIN_int16_T                    ((int16_T)(-32768))
#define MAX_uint16_T                   ((uint16_T)(65535))
#define MIN_uint16_T                   ((uint16_T)(0))
#define MAX_int32_T                    ((int32_T)(2147483647))
#define MIN_int32_T                    ((int32_T)(-2147483647-1))
#define MAX_uint32_T                   ((uint32_T)(0xFFFFFFFFU))
#define MIN_uint32_T                   ((uint32_T)(0))
#define MAX_int64_T                    ((int64_T)(9223372036854775807L))
#define MIN_int64_T                    ((int64_T)(-9223372036854775807L-1L))
#define MAX_uint64_T                   ((uint64_T)(0xFFFFFFFFFFFFFFFFUL))
#define MIN_uint64_T                   ((uint64_T)(0UL))

struct emxArray__common {
    void *data;
    int *size;
    int allocatedSize;
    int numDimensions;
    boolean_T canFreeData;
};

struct emxArray_real_T {
    double *data;
    int *size;
    int allocatedSize;
    int numDimensions;
    boolean_T canFreeData;
};

typedef struct {

    struct {
        uint32_T wordH;
        uint32_T wordL;
    } words;
} BigEndianIEEEDouble;

typedef struct {

    struct {
        uint32_T wordL;
        uint32_T wordH;
    } words;
} LittleEndianIEEEDouble;

typedef struct {

    union {
        real32_T wordLreal;
        uint32_T wordLuint;
    } wordL;
} IEEESingle;

extern emxArray_real_T *emxCreateND_real_T(int numDimensions, int *size);
extern emxArray_real_T *emxCreateWrapperND_real_T(double *data, int
        numDimensions, int *size);
extern emxArray_real_T *emxCreateWrapper_real_T(double *data, int rows, int cols);
extern emxArray_real_T *emxCreate_real_T(int rows, int cols);
extern void emxDestroyArray_real_T(emxArray_real_T *emxArray);
extern void emxInitArray_real_T(emxArray_real_T **pEmxArray, int numDimensions);
extern void emxEnsureCapacity(emxArray__common *emxArray, int oldNumel, int
        elementSize);
extern void emxFree_real_T(emxArray_real_T **pEmxArray);
extern void emxInit_real_T(emxArray_real_T **pEmxArray, int numDimensions);
extern void bsif(const emxArray_real_T *img, const double texturefilters_data[],
        const int texturefilters_size[3], double bsifdescription_data[],
        int bsifdescription_size[2]);
extern void bsif_initialize();
extern void bsif_terminate();
extern void float_colon_length(double b, int *n, double *anew, double *bnew,
        boolean_T *n_too_large);

class combine_vector_elements {
public:
    combine_vector_elements();
    combine_vector_elements(const combine_vector_elements& orig);
    virtual ~combine_vector_elements();
};
extern void b_conv2(const double arg1_data[], const int arg1_size[2], const
        emxArray_real_T *arg3, emxArray_real_T *c);
extern void c_conv2(const emxArray_real_T *arg1, const double arg2_data[], const
        int arg2_size[2], emxArray_real_T *c);
extern void conv2(const double arg2_data[], const int arg2_size[2], const
        emxArray_real_T *arg3, emxArray_real_T *c);
extern double eps(double x);
extern void filter2(const double b_data[], const int b_size[2], const
        emxArray_real_T *x, emxArray_real_T *y);
extern void histc(const double X[32768], const double edges_data[], const int
        edges_size[2], double N_data[], int N_size[1]);
extern void hist(const emxArray_real_T *Y, const double X_data[], const int
        X_size[2], double no_data[], int no_size[2]);
extern boolean_T b_isfinite(double x);

class mrdivide {
public:
    mrdivide();
    mrdivide(const mrdivide& orig);
    virtual ~mrdivide();
};
extern void rot90(const double A_data[], const int A_size[2], double B_data[],
        int B_size[2]);
extern void b_round(emxArray_real_T *x);
extern real_T rtGetInf(void);
extern real32_T rtGetInfF(void);
extern real_T rtGetMinusInf(void);
extern real32_T rtGetMinusInfF(void);
extern real_T rtGetNaN(void);
extern real32_T rtGetNaNF(void);
extern real_T rtInf;
extern real_T rtMinusInf;
extern real_T rtNaN;
extern real32_T rtInfF;
extern real32_T rtMinusInfF;
extern real32_T rtNaNF;
extern void rt_InitInfAndNaN(size_t realSize);
extern boolean_T rtIsInf(real_T value);
extern boolean_T rtIsInfF(real32_T value);
extern boolean_T rtIsNaN(real_T value);
extern boolean_T rtIsNaNF(real32_T value);

class sum {
public:
    sum();
    sum(const sum& orig);
    virtual ~sum();
};
extern void svd(const double A_data[], const int A_size[2], double U_data[], int
        U_size[2], double S_data[], int S_size[2], double V_data[], int
        V_size[2]);
//Declaration--------------------------------------------------------------------------------------
static double rt_powd_snf(double u0, double u1);

static double rt_powd_snf(double u0, double u1) {
    double y;
    double d0;
    double d1;
    if (rtIsNaN(u0) || rtIsNaN(u1)) {
        y = rtNaN;
    } else {
        d0 = fabs(u0);
        d1 = fabs(u1);
        if (rtIsInf(u1)) {
            if (d0 == 1.0) {
                y = rtNaN;
            } else if (d0 > 1.0) {
                if (u1 > 0.0) {
                    y = rtInf;
                } else {
                    y = 0.0;
                }
            } else if (u1 > 0.0) {
                y = 0.0;
            } else {
                y = rtInf;
            }
        } else if (d1 == 0.0) {
            y = 1.0;
        } else if (d1 == 1.0) {
            if (u1 > 0.0) {
                y = u0;
            } else {
                y = 1.0 / u0;
            }
        } else if (u1 == 2.0) {
            y = u0 * u0;
        } else if ((u1 == 0.5) && (u0 >= 0.0)) {
            y = sqrt(u0);
        } else if ((u0 < 0.0) && (u1 > floor(u1))) {
            y = rtNaN;
        } else {
            y = pow(u0, u1);
        }
    }

    return y;
}

void bsif(const emxArray_real_T *img, const double texturefilters_data[], const
        int texturefilters_size[3], double bsifdescription_data[], int
        bsifdescription_size[2]) {
    int numScl;
    unsigned int uv0[2];
    int i0;
    emxArray_real_T *codeImg;
    int loop_ub;
    int r;
    double anew;
    int i1;
    int b_loop_ub;
    int i2;
    int i3;
    int c_loop_ub;
    int d_loop_ub;
    int e_loop_ub;
    int i4;
    int i5;
    int i6;
    int i7;
    int f_loop_ub;
    int i8;
    int i9;
    int nm1d2;
    int k;
    emxArray_real_T *imgWrap;
    int n;
    int g_loop_ub;
    int h_loop_ub;
    int i_loop_ub;
    int i10;
    int j_loop_ub;
    int k_loop_ub;
    int i11;
    emxArray_real_T *ci;
    int tmp_size[2];
    double tmp_data[289];
    boolean_T n_too_large;
    double bnew;
    double y_data[4096];
    int y_size[2];
    int b_codeImg[1];
    emxArray_real_T c_codeImg;

    /* % Default parameters */
    /* sigmaBase=1; */
    /* scl=[1 2 4 8]; %sigma=scl(i)*sigmaBase; */
    /* % Check that input is gray scale */
    /* % Initialize */
    /*  Convert image to double */
    numScl = texturefilters_size[2] - 1;

    /* length(scl); */
    for (i0 = 0; i0 < 2; i0++) {
        uv0[i0] = (unsigned int) img->size[i0];
    }

    emxInit_real_T(&codeImg, 2);
    i0 = codeImg->size[0] * codeImg->size[1];
    codeImg->size[0] = (int) uv0[0];
    emxEnsureCapacity((emxArray__common *) codeImg, i0, (int) sizeof (double));
    i0 = codeImg->size[0] * codeImg->size[1];
    codeImg->size[1] = (int) uv0[1];
    emxEnsureCapacity((emxArray__common *) codeImg, i0, (int) sizeof (double));
    loop_ub = (int) uv0[0] * (int) uv0[1];
    for (i0 = 0; i0 < loop_ub; i0++) {
        codeImg->data[i0] = 1.0;
    }

    /*  Make spatial coordinates for sliding window */
    r = (int) floor((double) texturefilters_size[0] / 2.0);

    /* 3*max(scl)*sigmaBase; */
    /*  Wrap image (increase image size according to maximum filter radius by wrapping around) */
    if (1 > r) {
        loop_ub = -1;
    } else {
        loop_ub = r - 1;
    }

    anew = (double) (img->size[0] - r) + 1.0;
    if (anew > img->size[0]) {
        i0 = 1;
        i1 = 0;
    } else {
        i0 = (int) anew;
        i1 = img->size[0];
    }

    if (1 > r) {
        b_loop_ub = -1;
    } else {
        b_loop_ub = r - 1;
    }

    anew = (double) (img->size[1] - r) + 1.0;
    if (anew > img->size[1]) {
        i2 = 0;
        i3 = 0;
    } else {
        i2 = (int) anew - 1;
        i3 = img->size[1];
    }

    if (1 > r) {
        c_loop_ub = -1;
    } else {
        c_loop_ub = r - 1;
    }

    if (1 > r) {
        d_loop_ub = -1;
    } else {
        d_loop_ub = r - 1;
    }

    if (1 > r) {
        e_loop_ub = 0;
    } else {
        e_loop_ub = r;
    }

    anew = (double) (img->size[1] - r) + 1.0;
    if (anew > img->size[1]) {
        i4 = 0;
        i5 = 0;
    } else {
        i4 = (int) anew - 1;
        i5 = img->size[1];
    }

    anew = (double) (img->size[0] - r) + 1.0;
    if (anew > img->size[0]) {
        i6 = 1;
        i7 = 0;
    } else {
        i6 = (int) anew;
        i7 = img->size[0];
    }

    if (1 > r) {
        f_loop_ub = 0;
    } else {
        f_loop_ub = r;
    }

    anew = (double) (img->size[0] - r) + 1.0;
    if (anew > img->size[0]) {
        i8 = 0;
        i9 = 0;
    } else {
        i8 = (int) anew - 1;
        i9 = img->size[0];
    }

    anew = (double) (img->size[1] - r) + 1.0;
    if (anew > img->size[1]) {
        nm1d2 = 0;
        k = 0;
    } else {
        nm1d2 = (int) anew - 1;
        k = img->size[1];
    }

    emxInit_real_T(&imgWrap, 2);
    n = img->size[1];
    g_loop_ub = img->size[0];
    h_loop_ub = img->size[0] - 1;
    i_loop_ub = img->size[1];
    i10 = imgWrap->size[0] * imgWrap->size[1];
    imgWrap->size[0] = ((i9 - i8) + g_loop_ub) + e_loop_ub;
    imgWrap->size[1] = ((k - nm1d2) + n) + f_loop_ub;
    emxEnsureCapacity((emxArray__common *) imgWrap, i10, (int) sizeof (double));
    j_loop_ub = k - nm1d2;


    for (i10 = 0; i10 < j_loop_ub; i10++) {
        k_loop_ub = i9 - i8;
        for (i11 = 0; i11 < k_loop_ub; i11++) {
            imgWrap->data[i11 + imgWrap->size[0] * i10] = img->data[(i8 + i11) +
                    img->size[0] * (nm1d2 + i10)];
        }
    }


    for (i10 = 0; i10 < n; i10++) {
        j_loop_ub = i1 - i0;
        for (i11 = 0; i11 <= j_loop_ub; i11++) {
            imgWrap->data[i11 + imgWrap->size[0] * ((i10 + k) - nm1d2)] = img->data
                    [((i0 + i11) + img->size[0] * i10) - 1];
        }
    }


    for (i0 = 0; i0 < f_loop_ub; i0++) {
        j_loop_ub = i7 - i6;
        for (i1 = 0; i1 <= j_loop_ub; i1++) {
            imgWrap->data[i1 + imgWrap->size[0] * (((i0 + k) - nm1d2) + n)] =
                    img->data[((i6 + i1) + img->size[0] * i0) - 1];
        }
    }

    f_loop_ub = i3 - i2;

    for (i0 = 0; i0 < f_loop_ub; i0++) {
        for (i1 = 0; i1 < g_loop_ub; i1++) {
            imgWrap->data[((i1 + i9) - i8) + imgWrap->size[0] * i0] = img->data[i1 +
                    img->size[0] * (i2 + i0)];
        }
    }

    f_loop_ub = img->size[1];

    for (i0 = 0; i0 < f_loop_ub; i0++) {
        n = img->size[0];
        for (i1 = 0; i1 < n; i1++) {
            imgWrap->data[((i1 + i9) - i8) + imgWrap->size[0] * ((i0 + i3) - i2)] =
                    img->data[i1 + img->size[0] * i0];
        }
    }


    for (i0 = 0; i0 <= b_loop_ub; i0++) {
        for (i1 = 0; i1 <= h_loop_ub; i1++) {
            imgWrap->data[((i1 + i9) - i8) + imgWrap->size[0] * (((i0 + i3) - i2) +
                    img->size[1])] = img->data[i1 + img->size[0] * i0];
        }
    }

    b_loop_ub = i5 - i4;

    for (i0 = 0; i0 < b_loop_ub; i0++) {
        for (i1 = 0; i1 < e_loop_ub; i1++) {
            imgWrap->data[(((i1 + i9) - i8) + g_loop_ub) + imgWrap->size[0] * i0] =
                    img->data[i1 + img->size[0] * (i4 + i0)];
        }
    }


    for (i0 = 0; i0 < i_loop_ub; i0++) {
        for (i1 = 0; i1 <= loop_ub; i1++) {
            imgWrap->data[(((i1 + i9) - i8) + g_loop_ub) + imgWrap->size[0] * ((i0 +
                    i5) - i4)] = img->data[i1 + img->size[0] * i0];
        }
    }


    for (i0 = 0; i0 <= d_loop_ub; i0++) {
        for (i1 = 0; i1 <= c_loop_ub; i1++) {
            imgWrap->data[(((i1 + i9) - i8) + g_loop_ub) + imgWrap->size[0] * (((i0 +
                    i5) - i4) + i_loop_ub)] = img->data[i1 + img->size[0] * i0];
        }
    }

    /* % Loop over scales */
    /* figf=figure;subplot(numScl/2,2,1); */
    /* counter=1; */
    g_loop_ub = 0;
    emxInit_real_T(&ci, 2);
    while (g_loop_ub <= numScl) {
        loop_ub = texturefilters_size[0];
        b_loop_ub = texturefilters_size[1];
        nm1d2 = numScl - g_loop_ub;
        tmp_size[0] = texturefilters_size[0];
        tmp_size[1] = texturefilters_size[1];


        for (i0 = 0; i0 < b_loop_ub; i0++) {
            for (i1 = 0; i1 < loop_ub; i1++) {
                tmp_data[i1 + loop_ub * i0] = texturefilters_data[(i1 +
                        texturefilters_size[0] * i0) + texturefilters_size[0] *
                        texturefilters_size[1] * nm1d2];
            }
        }

        /* figure;imagesc(tmp);axis image;axis off; colormap('gray'); */
        filter2(tmp_data, tmp_size, imgWrap, ci);

        /* figure(figf);subplot(numScl/2,2,i); */
        /* imagesc(ci);axis image;axis off; */
        n = (int) rt_powd_snf(2.0, (1.0 + (double) g_loop_ub) - 1.0);
        i0 = codeImg->size[0] * codeImg->size[1];
        emxEnsureCapacity((emxArray__common *) codeImg, i0, (int) sizeof (double));
        nm1d2 = codeImg->size[0];
        k = codeImg->size[1];
        loop_ub = nm1d2 * k;
        for (i0 = 0; i0 < loop_ub; i0++) {
            codeImg->data[i0] += (double) (ci->data[i0] > 0.0) * (double) n;
        }

        g_loop_ub++;
    }

    emxFree_real_T(&ci);
    emxFree_real_T(&imgWrap);
    float_colon_length(rt_powd_snf(2.0, (double) texturefilters_size[2]), &n, &anew,
            &bnew, &n_too_large);
    y_size[0] = 1;
    y_size[1] = n;
    if (n > 0) {
        y_data[0] = anew;
        if (n > 1) {
            y_data[n - 1] = bnew;
            i0 = n - 1;
            nm1d2 = i0 >> 1;
            for (k = 1; k < nm1d2; k++) {
                y_data[k] = anew + (double) k;
                y_data[(n - k) - 1] = bnew - (double) k;
            }

            if (nm1d2 << 1 == n - 1) {
                y_data[nm1d2] = (anew + bnew) / 2.0;
            } else {
                y_data[nm1d2] = anew + (double) nm1d2;
                y_data[nm1d2 + 1] = bnew - (double) nm1d2;
            }
        }
    }

    b_codeImg[0] = codeImg->size[0] * codeImg->size[1];
    c_codeImg = *codeImg;
    c_codeImg.size = (int *) &b_codeImg;
    c_codeImg.numDimensions = 1;
    hist(&c_codeImg, y_data, y_size, bsifdescription_data, bsifdescription_size);
    emxFree_real_T(&codeImg);
}

emxArray_real_T *emxCreateND_real_T(int numDimensions, int *size) {
    emxArray_real_T *emx;
    int numEl;
    int i;
    emxInit_real_T(&emx, numDimensions);
    numEl = 1;
    for (i = 0; i < numDimensions; i++) {
        numEl *= size[i];
        emx->size[i] = size[i];
    }

    emx->data = (double *) calloc((unsigned int) numEl, sizeof (double));
    emx->numDimensions = numDimensions;
    emx->allocatedSize = numEl;
    return emx;
}

emxArray_real_T *emxCreateWrapperND_real_T(double *data, int numDimensions, int *
        size) {
    emxArray_real_T *emx;
    int numEl;
    int i;
    emxInit_real_T(&emx, numDimensions);
    numEl = 1;
    for (i = 0; i < numDimensions; i++) {
        numEl *= size[i];
        emx->size[i] = size[i];
    }

    emx->data = data;
    emx->numDimensions = numDimensions;
    emx->allocatedSize = numEl;
    emx->canFreeData = false;
    return emx;
}

emxArray_real_T *emxCreateWrapper_real_T(double *data, int rows, int cols) {
    emxArray_real_T *emx;
    int size[2];
    int numEl;
    int i;
    size[0] = rows;
    size[1] = cols;
    emxInit_real_T(&emx, 2);
    numEl = 1;
    for (i = 0; i < 2; i++) {
        numEl *= size[i];
        emx->size[i] = size[i];
    }

    emx->data = data;
    emx->numDimensions = 2;
    emx->allocatedSize = numEl;
    emx->canFreeData = false;
    return emx;
}

emxArray_real_T *emxCreate_real_T(int rows, int cols) {
    emxArray_real_T *emx;
    int size[2];
    int numEl;
    int i;
    size[0] = rows;
    size[1] = cols;
    emxInit_real_T(&emx, 2);
    numEl = 1;
    for (i = 0; i < 2; i++) {
        numEl *= size[i];
        emx->size[i] = size[i];
    }

    emx->data = (double *) calloc((unsigned int) numEl, sizeof (double));
    emx->numDimensions = 2;
    emx->allocatedSize = numEl;
    return emx;
}

void emxDestroyArray_real_T(emxArray_real_T *emxArray) {
    emxFree_real_T(&emxArray);
}

void emxInitArray_real_T(emxArray_real_T **pEmxArray, int numDimensions) {
    emxInit_real_T(pEmxArray, numDimensions);
}

/* Function Definitions */
void emxEnsureCapacity(emxArray__common *emxArray, int oldNumel, int elementSize) {
    int newNumel;
    int i;
    void *newData;
    newNumel = 1;
    for (i = 0; i < emxArray->numDimensions; i++) {
        newNumel *= emxArray->size[i];
    }

    if (newNumel > emxArray->allocatedSize) {
        i = emxArray->allocatedSize;
        if (i < 16) {
            i = 16;
        }

        while (i < newNumel) {
            i <<= 1;
        }

        newData = calloc((unsigned int) i, (unsigned int) elementSize);
        if (emxArray->data != NULL) {
            memcpy(newData, emxArray->data, (unsigned int) (elementSize * oldNumel));
            if (emxArray->canFreeData) {
                free(emxArray->data);
            }
        }

        emxArray->data = newData;
        emxArray->allocatedSize = i;
        emxArray->canFreeData = true;
    }
}

void emxFree_real_T(emxArray_real_T **pEmxArray) {
    if (*pEmxArray != (emxArray_real_T *) NULL) {
        if (((*pEmxArray)->data != (double *) NULL) && (*pEmxArray)->canFreeData) {
            free((void *) (*pEmxArray)->data);
        }

        free((void *) (*pEmxArray)->size);
        free((void *) *pEmxArray);
        *pEmxArray = (emxArray_real_T *) NULL;
    }
}

void emxInit_real_T(emxArray_real_T **pEmxArray, int numDimensions) {
    emxArray_real_T *emxArray;
    int i;
    *pEmxArray = (emxArray_real_T *) malloc(sizeof (emxArray_real_T));
    emxArray = *pEmxArray;
    emxArray->data = (double *) NULL;
    emxArray->numDimensions = numDimensions;
    emxArray->size = (int *) malloc((unsigned int) (sizeof (int) * numDimensions));
    emxArray->allocatedSize = 0;
    emxArray->canFreeData = true;
    for (i = 0; i < numDimensions; i++) {
        emxArray->size[i] = 0;
    }
}

/* Function Definitions */
void bsif_initialize() {
    rt_InitInfAndNaN(8U);
}

void bsif_terminate() {
    /* (no terminate code required) */
}

void float_colon_length(double b, int *n, double *anew, double *bnew, boolean_T *
        n_too_large) {
    double ndbl;
    double cdiff;
    double absb;
    if (rtIsNaN(b)) {
        *n = 1;
        *anew = rtNaN;
        *bnew = b;
        *n_too_large = false;
    } else if (rtIsInf(b)) {
        *n = 1;
        *anew = rtNaN;
        *bnew = b;
        *n_too_large = !(1.0 == b);
    } else {
        *anew = 1.0;
        ndbl = floor((b - 1.0) + 0.5);
        *bnew = 1.0 + ndbl;
        cdiff = (1.0 + ndbl) - b;
        absb = fabs(b);
        if ((1.0 >= absb) || rtIsNaN(absb)) {
            absb = 1.0;
        }

        if (fabs(cdiff) < 4.4408920985006262E-16 * absb) {
            ndbl++;
            *bnew = b;
        } else if (cdiff > 0.0) {
            *bnew = 1.0 + (ndbl - 1.0);
        } else {
            ndbl++;
        }

        *n_too_large = false;
        *n = (int) ndbl;
    }
}

combine_vector_elements::combine_vector_elements() {
}

combine_vector_elements::combine_vector_elements(const combine_vector_elements& orig) {
}

combine_vector_elements::~combine_vector_elements() {
}

void b_conv2(const double arg1_data[], const int arg1_size[2], const
        emxArray_real_T *arg3, emxArray_real_T *c) {
    int ma;
    int nb;
    int mc;
    emxArray_real_T *work;
    int b_ma;
    int b_nb;
    int k;
    int i14;
    int j;
    int ko;
    int ilo;
    int ihi;
    ma = arg1_size[0] * arg1_size[1] - 1;
    nb = arg3->size[1];
    if (arg3->size[0] < ma) {
        mc = 0;
    } else {
        mc = arg3->size[0] - ma;
    }

    emxInit_real_T(&work, 2);
    b_ma = arg1_size[0] * arg1_size[1];
    b_nb = arg3->size[1];
    k = arg3->size[1];
    i14 = work->size[0] * work->size[1];
    work->size[0] = mc;
    emxEnsureCapacity((emxArray__common *) work, i14, (int) sizeof (double));
    i14 = work->size[0] * work->size[1];
    work->size[1] = k;
    emxEnsureCapacity((emxArray__common *) work, i14, (int) sizeof (double));
    j = mc * k;
    for (i14 = 0; i14 < j; i14++) {
        work->data[i14] = 0.0;
    }

    k = arg3->size[1];
    i14 = c->size[0] * c->size[1];
    c->size[0] = mc;
    emxEnsureCapacity((emxArray__common *) c, i14, (int) sizeof (double));
    i14 = c->size[0] * c->size[1];
    c->size[1] = k;
    emxEnsureCapacity((emxArray__common *) c, i14, (int) sizeof (double));
    j = mc * k;
    for (i14 = 0; i14 < j; i14++) {
        c->data[i14] = 0.0;
    }

    if ((arg1_size[0] == 0) || (arg1_size[1] == 0) || ((arg3->size[0] == 0) ||
            (arg3->size[1] == 0)) || ((mc == 0) || (k == 0))) {
    } else {
        j = (ma - arg3->size[0]) + 2;
        k = mc + ma;
        if (b_ma <= k) {
            i14 = b_ma;
        } else {
            i14 = k;
        }

        if (j > 1) {
            k = j - 1;
        } else {
            k = 0;
        }

        while (k + 1 <= i14) {
            ko = (k - ma) + 1;
            if (ko > 0) {
                ilo = ko;
            } else {
                ilo = 1;
            }

            ihi = (arg3->size[0] + ko) - 1;
            if (ihi > mc) {
                ihi = mc;
            }

            if (arg1_data[k] != 0.0) {
                for (j = 0; j + 1 <= b_nb; j++) {
                    for (b_ma = ilo; b_ma <= ihi; b_ma++) {
                        work->data[(b_ma + work->size[0] * j) - 1] += arg3->data[(b_ma - ko)
                                + arg3->size[0] * j] * arg1_data[k];
                    }
                }
            }

            k++;
        }

        if (b_nb > nb) {
            b_nb = nb;
        }

        for (j = 0; j + 1 <= b_nb; j++) {
            for (b_ma = 0; b_ma + 1 <= mc; b_ma++) {
                c->data[b_ma + c->size[0] * j] += work->data[b_ma + work->size[0] * j];
            }
        }
    }

    emxFree_real_T(&work);
}

void c_conv2(const emxArray_real_T *arg1, const double arg2_data[], const int
        arg2_size[2], emxArray_real_T *c) {
    int mc;
    int nc;
    int r;
    int loop_ub;
    boolean_T b1;
    int a;
    int b_a;
    int c_a;
    int ma;
    int na;
    int b_c;
    int firstColB;
    int lastColB;
    int firstRowB;
    int lastRowB;
    int lastColA;
    int k;
    int b_firstColB;
    int iC;
    int c_c;
    int iB;
    int i;
    int b_i;
    int a_length;
    if (arg1->size[0] < arg2_size[0] - 1) {
        mc = 0;
    } else {
        mc = (arg1->size[0] - arg2_size[0]) + 1;
    }

    if (arg1->size[1] < arg2_size[1] - 1) {
        nc = 0;
    } else {
        nc = (arg1->size[1] - arg2_size[1]) + 1;
    }

    r = c->size[0] * c->size[1];
    c->size[0] = mc;
    emxEnsureCapacity((emxArray__common *) c, r, (int) sizeof (double));
    r = c->size[0] * c->size[1];
    c->size[1] = nc;
    emxEnsureCapacity((emxArray__common *) c, r, (int) sizeof (double));
    loop_ub = mc * nc;
    for (r = 0; r < loop_ub; r++) {
        c->data[r] = 0.0;
    }

    if ((arg1->size[0] == 0) || (arg1->size[1] == 0) || ((arg2_size[0] == 0) ||
            (arg2_size[1] == 0)) || ((mc == 0) || (nc == 0))) {
        b1 = true;
    } else {
        b1 = false;
    }

    if (!b1) {
        loop_ub = arg2_size[1] - 1;
        a = arg1->size[1];
        b_a = arg2_size[0] - 1;
        c_a = arg1->size[0];
        ma = arg1->size[0];
        na = arg1->size[1];
        b_c = arg1->size[0] - arg2_size[0];
        if (arg1->size[1] < arg2_size[1] - 1) {
            firstColB = arg2_size[1] - arg1->size[1];
        } else {
            firstColB = 0;
        }

        if (arg2_size[1] <= arg1->size[1] - 1) {
            lastColB = arg2_size[1];
        } else {
            lastColB = arg1->size[1];
        }

        if (arg1->size[0] < arg2_size[0] - 1) {
            firstRowB = arg2_size[0] - arg1->size[0];
        } else {
            firstRowB = 0;
        }

        if (arg2_size[0] <= arg1->size[0] - 1) {
            lastRowB = arg2_size[0];
        } else {
            lastRowB = arg1->size[0];
        }

        while (firstColB <= lastColB - 1) {
            if ((firstColB + na) - 1 < a - 1) {
                lastColA = na;
            } else {
                lastColA = a - firstColB;
            }

            if (firstColB < loop_ub) {
                k = loop_ub - firstColB;
            } else {
                k = 0;
            }

            while (k <= lastColA - 1) {
                if (firstColB + k > loop_ub) {
                    b_firstColB = (firstColB + k) - loop_ub;
                } else {
                    b_firstColB = 0;
                }

                iC = b_firstColB * (b_c + 1);
                c_c = k * ma;
                iB = firstRowB + firstColB * arg2_size[0];
                for (i = firstRowB; i < lastRowB; i++) {
                    if (i < b_a) {
                        mc = b_a - i;
                    } else {
                        mc = 0;
                    }

                    if (i + ma <= c_a - 1) {
                        b_i = ma;
                    } else {
                        b_i = c_a - i;
                    }

                    a_length = b_i - mc;
                    mc += c_c;
                    nc = iC;
                    for (r = 1; r <= a_length; r++) {
                        c->data[nc] += arg2_data[iB] * arg1->data[mc];
                        mc++;
                        nc++;
                    }

                    iB++;
                    if (i >= b_a) {
                        iC++;
                    }
                }

                k++;
            }

            firstColB++;
        }
    }
}

void conv2(const double arg2_data[], const int arg2_size[2], const
        emxArray_real_T *arg3, emxArray_real_T *c) {
    int na;
    int mb;
    int nc;
    emxArray_real_T *work;
    int b_na;
    int nb;
    int ko;
    int ihi;
    int i13;
    int j;
    int i;
    int jmkom1;
    na = arg2_size[0] * arg2_size[1] - 1;
    mb = arg3->size[0];
    if (arg3->size[1] < na) {
        nc = 0;
    } else {
        nc = arg3->size[1] - na;
    }

    emxInit_real_T(&work, 2);
    b_na = arg2_size[0] * arg2_size[1];
    nb = arg3->size[1];
    ko = arg3->size[0];
    ihi = arg3->size[1];
    i13 = work->size[0] * work->size[1];
    work->size[0] = ko;
    emxEnsureCapacity((emxArray__common *) work, i13, (int) sizeof (double));
    i13 = work->size[0] * work->size[1];
    work->size[1] = ihi;
    emxEnsureCapacity((emxArray__common *) work, i13, (int) sizeof (double));
    ihi *= ko;
    for (i13 = 0; i13 < ihi; i13++) {
        work->data[i13] = 0.0;
    }

    ko = arg3->size[0];
    i13 = c->size[0] * c->size[1];
    c->size[0] = ko;
    emxEnsureCapacity((emxArray__common *) c, i13, (int) sizeof (double));
    i13 = c->size[0] * c->size[1];
    c->size[1] = nc;
    emxEnsureCapacity((emxArray__common *) c, i13, (int) sizeof (double));
    ihi = ko * nc;
    for (i13 = 0; i13 < ihi; i13++) {
        c->data[i13] = 0.0;
    }

    if ((arg2_size[0] == 0) || (arg2_size[1] == 0) || ((arg3->size[0] == 0) ||
            (arg3->size[1] == 0)) || ((ko == 0) || (nc == 0))) {
    } else {
        ihi = arg3->size[0];
        if (ihi > mb) {
            ihi = mb;
        }

        for (j = 0; j + 1 <= nb; j++) {
            for (i = 0; i + 1 <= ihi; i++) {
                work->data[i + work->size[0] * j] += arg3->data[i + arg3->size[0] * j];
            }
        }

        ko = (na - arg3->size[1]) + 2;
        ihi = nc + na;
        if (b_na <= ihi) {
            i13 = b_na;
        } else {
            i13 = ihi;
        }

        if (ko > 1) {
            ihi = ko - 1;
        } else {
            ihi = 0;
        }

        while (ihi + 1 <= i13) {
            ko = ihi - na;
            b_na = nb + ko;
            if (b_na > nc) {
                b_na = nc;
            }

            if (arg2_data[ihi] != 0.0) {
                if (ko + 1 > 0) {
                    j = ko;
                } else {
                    j = 0;
                }

                while (j + 1 <= b_na) {
                    jmkom1 = j - ko;
                    for (i = 0; i + 1 <= mb; i++) {
                        c->data[i + c->size[0] * j] += work->data[i + work->size[0] * jmkom1]
                                * arg2_data[ihi];
                    }

                    j++;
                }
            }

            ihi++;
        }
    }

    emxFree_real_T(&work);
}

double eps(double x) {
    double r;
    double absxk;
    int exponent;
    absxk = fabs(x);
    if ((!rtIsInf(absxk)) && (!rtIsNaN(absxk))) {
        if (absxk <= 2.2250738585072014E-308) {
            r = 4.94065645841247E-324;
        } else {
            frexp(absxk, &exponent);
            r = ldexp(1.0, exponent - 53);
        }
    } else {
        r = rtNaN;
    }

    return r;
}

static void b_eml_xaxpy(int n, double a, const double x_data[], int ix0, double
        y_data[], int iy0);
static double b_eml_xnrm2(int n, const double x_data[], int ix0);
static void c_eml_xaxpy(int n, double a, const double x_data[], int ix0, double
        y_data[], int iy0);
static void eml_xaxpy(int n, double a, int ix0, double y_data[], int iy0);
static double eml_xdotc(int n, const double x_data[], int ix0, const double
        y_data[], int iy0);
static void eml_xgesvd(const double A_data[], const int A_size[2], double
        U_data[], int U_size[2], double S_data[], int S_size[1], double V_data[], int
        V_size[2]);
static double eml_xnrm2(int n, const double x_data[], int ix0);
static void eml_xrot(int n, double x_data[], int ix0, int iy0, double c, double
        s);
static void eml_xrotg(double *a, double *b, double *c, double *s);
static void eml_xscal(int n, double a, double x_data[], int ix0);
static void eml_xswap(int n, double x_data[], int ix0, int iy0);

/* Function Definitions */
static void b_eml_xaxpy(int n, double a, const double x_data[], int ix0, double
        y_data[], int iy0) {
    int ix;
    int iy;
    int k;
    if ((n < 1) || (a == 0.0)) {
    } else {
        ix = ix0 - 1;
        iy = iy0 - 1;
        for (k = 0; k < n; k++) {
            y_data[iy] += a * x_data[ix];
            ix++;
            iy++;
        }
    }
}

static double b_eml_xnrm2(int n, const double x_data[], int ix0) {
    double y;
    double scale;
    int kend;
    int k;
    double absxk;
    double t;
    y = 0.0;
    if (n < 1) {
    } else if (n == 1) {
        y = fabs(x_data[ix0 - 1]);
    } else {
        scale = 2.2250738585072014E-308;
        kend = (ix0 + n) - 1;
        for (k = ix0; k <= kend; k++) {
            absxk = fabs(x_data[k - 1]);
            if (absxk > scale) {
                t = scale / absxk;
                y = 1.0 + y * t * t;
                scale = absxk;
            } else {
                t = absxk / scale;
                y += t * t;
            }
        }

        y = scale * sqrt(y);
    }

    return y;
}

static void c_eml_xaxpy(int n, double a, const double x_data[], int ix0, double
        y_data[], int iy0) {
    int ix;
    int iy;
    int k;
    if ((n < 1) || (a == 0.0)) {
    } else {
        ix = ix0 - 1;
        iy = iy0 - 1;
        for (k = 0; k < n; k++) {
            y_data[iy] += a * x_data[ix];
            ix++;
            iy++;
        }
    }
}

static void eml_xaxpy(int n, double a, int ix0, double y_data[], int iy0) {
    int ix;
    int iy;
    int k;
    if ((n < 1) || (a == 0.0)) {
    } else {
        ix = ix0 - 1;
        iy = iy0 - 1;
        for (k = 0; k < n; k++) {
            y_data[iy] += a * y_data[ix];
            ix++;
            iy++;
        }
    }
}

static double eml_xdotc(int n, const double x_data[], int ix0, const double
        y_data[], int iy0) {
    double d;
    int ix;
    int iy;
    int k;
    d = 0.0;
    if (n < 1) {
    } else {
        ix = ix0;
        iy = iy0;
        for (k = 1; k <= n; k++) {
            d += x_data[ix - 1] * y_data[iy - 1];
            ix++;
            iy++;
        }
    }

    return d;
}

static void eml_xgesvd(const double A_data[], const int A_size[2], double
        U_data[], int U_size[2], double S_data[], int S_size[1], double V_data[], int
        V_size[2]) {
    int m;
    int kase;
    int mm;
    double b_A_data[289];
    int n;
    int p;
    int minnp;
    double s_data[17];
    double e_data[17];
    double work_data[17];
    int nrt;
    int nct;
    int q;
    int iter;
    int nmq;
    boolean_T apply_transform;
    double ztest0;
    int qs;
    int jj;
    double ztest;
    double snorm;
    boolean_T exitg3;
    boolean_T exitg2;
    double f;
    double b;
    double varargin_1[5];
    double mtmp;
    boolean_T exitg1;
    double sqds;
    m = A_size[0];
    kase = A_size[0] * A_size[1];
    for (mm = 0; mm < kase; mm++) {
        b_A_data[mm] = A_data[mm];
    }

    n = A_size[0];
    p = A_size[1];
    if (A_size[0] <= A_size[1]) {
        minnp = A_size[0];
    } else {
        minnp = A_size[1];
    }

    if (A_size[0] + 1 <= A_size[1]) {
        kase = (signed char) (A_size[0] + 1);
    } else {
        kase = (signed char) A_size[1];
    }

    for (mm = 0; mm < kase; mm++) {
        s_data[mm] = 0.0;
    }

    kase = (signed char) A_size[1];
    for (mm = 0; mm < kase; mm++) {
        e_data[mm] = 0.0;
    }

    kase = (signed char) A_size[0];
    for (mm = 0; mm < kase; mm++) {
        work_data[mm] = 0.0;
    }

    U_size[0] = (signed char) A_size[0];
    U_size[1] = (signed char) A_size[0];
    kase = (signed char) A_size[0] * (signed char) A_size[0];
    for (mm = 0; mm < kase; mm++) {
        U_data[mm] = 0.0;
    }

    V_size[0] = (signed char) A_size[1];
    V_size[1] = (signed char) A_size[1];
    kase = (signed char) A_size[1] * (signed char) A_size[1];
    for (mm = 0; mm < kase; mm++) {
        V_data[mm] = 0.0;
    }

    if ((A_size[0] == 0) || (A_size[1] == 0)) {
        for (kase = 0; kase + 1 <= A_size[0]; kase++) {
            U_data[kase + U_size[0] * kase] = 1.0;
        }

        for (kase = 0; kase + 1 <= A_size[1]; kase++) {
            V_data[kase + V_size[0] * kase] = 1.0;
        }
    } else {
        if (A_size[1] < 2) {
            kase = 0;
        } else {
            kase = A_size[1] - 2;
        }

        if (kase <= A_size[0]) {
            nrt = kase;
        } else {
            nrt = A_size[0];
        }

        if (A_size[0] - 1 <= A_size[1]) {
            nct = A_size[0] - 1;
        } else {
            nct = A_size[1];
        }

        if (nct >= nrt) {
            mm = nct;
        } else {
            mm = nrt;
        }

        for (q = 0; q + 1 <= mm; q++) {
            iter = (q + n * q) + 1;
            nmq = n - q;
            apply_transform = false;
            if (q + 1 <= nct) {
                ztest0 = eml_xnrm2(nmq, b_A_data, iter);
                if (ztest0 > 0.0) {
                    apply_transform = true;
                    if (b_A_data[iter - 1] < 0.0) {
                        s_data[q] = -ztest0;
                    } else {
                        s_data[q] = ztest0;
                    }

                    if (fabs(s_data[q]) >= 1.0020841800044864E-292) {
                        ztest0 = 1.0 / s_data[q];
                        kase = (iter + nmq) - 1;
                        for (qs = iter; qs <= kase; qs++) {
                            b_A_data[qs - 1] *= ztest0;
                        }
                    } else {
                        kase = (iter + nmq) - 1;
                        for (qs = iter; qs <= kase; qs++) {
                            b_A_data[qs - 1] /= s_data[q];
                        }
                    }

                    b_A_data[iter - 1]++;
                    s_data[q] = -s_data[q];
                } else {
                    s_data[q] = 0.0;
                }
            }

            for (jj = q + 1; jj + 1 <= p; jj++) {
                kase = q + n * jj;
                if (apply_transform) {
                    ztest0 = eml_xdotc(nmq, b_A_data, iter, b_A_data, kase + 1);
                    eml_xaxpy(nmq, -(ztest0 / b_A_data[q + m * q]), iter, b_A_data, kase +
                            1);
                }

                e_data[jj] = b_A_data[kase];
            }

            if (q + 1 <= nct) {
                for (kase = q; kase + 1 <= n; kase++) {
                    U_data[kase + U_size[0] * q] = b_A_data[kase + m * q];
                }
            }

            if (q + 1 <= nrt) {
                iter = (p - q) - 1;
                ztest0 = b_eml_xnrm2(iter, e_data, q + 2);
                if (ztest0 == 0.0) {
                    e_data[q] = 0.0;
                } else {
                    if (e_data[q + 1] < 0.0) {
                        e_data[q] = -ztest0;
                    } else {
                        e_data[q] = ztest0;
                    }

                    ztest0 = e_data[q];
                    if (fabs(e_data[q]) >= 1.0020841800044864E-292) {
                        ztest0 = 1.0 / e_data[q];
                        kase = q + iter;
                        for (qs = q + 1; qs + 1 <= kase + 1; qs++) {
                            e_data[qs] *= ztest0;
                        }
                    } else {
                        kase = q + iter;
                        for (qs = q + 1; qs + 1 <= kase + 1; qs++) {
                            e_data[qs] /= ztest0;
                        }
                    }

                    e_data[q + 1]++;
                    e_data[q] = -e_data[q];
                    if (q + 2 <= n) {
                        for (kase = q + 1; kase + 1 <= n; kase++) {
                            work_data[kase] = 0.0;
                        }

                        for (jj = q + 1; jj + 1 <= p; jj++) {
                            b_eml_xaxpy(nmq - 1, e_data[jj], b_A_data, (q + n * jj) + 2,
                                    work_data, q + 2);
                        }

                        for (jj = q + 1; jj + 1 <= p; jj++) {
                            c_eml_xaxpy(nmq - 1, -e_data[jj] / e_data[q + 1], work_data, q + 2,
                                    b_A_data, (q + n * jj) + 2);
                        }
                    }
                }

                for (kase = q + 1; kase + 1 <= p; kase++) {
                    V_data[kase + V_size[0] * q] = e_data[kase];
                }
            }
        }

        if (A_size[1] <= A_size[0] + 1) {
            m = A_size[1];
        } else {
            m = A_size[0] + 1;
        }

        if (nct < A_size[1]) {
            s_data[nct] = b_A_data[nct + A_size[0] * nct];
        }

        if (A_size[0] < m) {
            s_data[m - 1] = 0.0;
        }

        if (nrt + 1 < m) {
            e_data[nrt] = b_A_data[nrt + A_size[0] * (m - 1)];
        }

        e_data[m - 1] = 0.0;
        if (nct + 1 <= A_size[0]) {
            for (jj = nct; jj + 1 <= n; jj++) {
                for (kase = 1; kase <= n; kase++) {
                    U_data[(kase + U_size[0] * jj) - 1] = 0.0;
                }

                U_data[jj + U_size[0] * jj] = 1.0;
            }
        }

        for (q = nct - 1; q + 1 > 0; q--) {
            nmq = n - q;
            iter = q + n * q;
            if (s_data[q] != 0.0) {
                for (jj = q; jj + 2 <= n; jj++) {
                    kase = (q + n * (jj + 1)) + 1;
                    ztest0 = eml_xdotc(nmq, U_data, iter + 1, U_data, kase);
                    eml_xaxpy(nmq, -(ztest0 / U_data[iter]), iter + 1, U_data, kase);
                }

                for (kase = q; kase + 1 <= n; kase++) {
                    U_data[kase + U_size[0] * q] = -U_data[kase + U_size[0] * q];
                }

                U_data[iter]++;
                for (kase = 1; kase <= q; kase++) {
                    U_data[(kase + U_size[0] * q) - 1] = 0.0;
                }
            } else {
                for (kase = 1; kase <= n; kase++) {
                    U_data[(kase + U_size[0] * q) - 1] = 0.0;
                }

                U_data[iter] = 1.0;
            }
        }

        for (q = A_size[1] - 1; q + 1 > 0; q--) {
            if ((q + 1 <= nrt) && (e_data[q] != 0.0)) {
                iter = (p - q) - 1;
                kase = (q + p * q) + 2;
                for (jj = q; jj + 2 <= p; jj++) {
                    qs = (q + p * (jj + 1)) + 2;
                    ztest0 = eml_xdotc(iter, V_data, kase, V_data, qs);
                    eml_xaxpy(iter, -(ztest0 / V_data[kase - 1]), kase, V_data, qs);
                }
            }

            for (kase = 1; kase <= p; kase++) {
                V_data[(kase + V_size[0] * q) - 1] = 0.0;
            }

            V_data[q + V_size[0] * q] = 1.0;
        }

        for (q = 0; q + 1 <= m; q++) {
            if (s_data[q] != 0.0) {
                ztest = fabs(s_data[q]);
                ztest0 = s_data[q] / ztest;
                s_data[q] = ztest;
                if (q + 1 < m) {
                    e_data[q] /= ztest0;
                }

                if (q + 1 <= n) {
                    eml_xscal(n, ztest0, U_data, 1 + n * q);
                }
            }

            if ((q + 1 < m) && (e_data[q] != 0.0)) {
                ztest = fabs(e_data[q]);
                ztest0 = ztest / e_data[q];
                e_data[q] = ztest;
                s_data[q + 1] *= ztest0;
                eml_xscal(p, ztest0, V_data, 1 + p * (q + 1));
            }
        }

        mm = m;
        iter = 0;
        snorm = 0.0;
        for (kase = 0; kase + 1 <= m; kase++) {
            ztest0 = fabs(s_data[kase]);
            ztest = fabs(e_data[kase]);
            if ((ztest0 >= ztest) || rtIsNaN(ztest)) {
            } else {
                ztest0 = ztest;
            }

            if ((snorm >= ztest0) || rtIsNaN(ztest0)) {
            } else {
                snorm = ztest0;
            }
        }

        while ((m > 0) && (!(iter >= 75))) {
            q = m - 1;
            exitg3 = false;
            while (!(exitg3 || (q == 0))) {
                ztest0 = fabs(e_data[q - 1]);
                if ((ztest0 <= 2.2204460492503131E-16 * (fabs(s_data[q - 1]) + fabs
                        (s_data[q]))) || (ztest0 <= 1.0020841800044864E-292) || ((iter >
                        20) && (ztest0 <= 2.2204460492503131E-16 * snorm))) {
                    e_data[q - 1] = 0.0;
                    exitg3 = true;
                } else {
                    q--;
                }
            }

            if (q == m - 1) {
                kase = 4;
            } else {
                qs = m;
                kase = m;
                exitg2 = false;
                while ((!exitg2) && (kase >= q)) {
                    qs = kase;
                    if (kase == q) {
                        exitg2 = true;
                    } else {
                        ztest0 = 0.0;
                        if (kase < m) {
                            ztest0 = fabs(e_data[kase - 1]);
                        }

                        if (kase > q + 1) {
                            ztest0 += fabs(e_data[kase - 2]);
                        }

                        ztest = fabs(s_data[kase - 1]);
                        if ((ztest <= 2.2204460492503131E-16 * ztest0) || (ztest <=
                                1.0020841800044864E-292)) {
                            s_data[kase - 1] = 0.0;
                            exitg2 = true;
                        } else {
                            kase--;
                        }
                    }
                }

                if (qs == q) {
                    kase = 3;
                } else if (qs == m) {
                    kase = 1;
                } else {
                    kase = 2;
                    q = qs;
                }
            }

            switch (kase) {
                case 1:
                    f = e_data[m - 2];
                    e_data[m - 2] = 0.0;
                    for (qs = m - 2; qs + 1 >= q + 1; qs--) {
                        ztest0 = s_data[qs];
                        eml_xrotg(&ztest0, &f, &ztest, &b);
                        s_data[qs] = ztest0;
                        if (qs + 1 > q + 1) {
                            f = -b * e_data[qs - 1];
                            e_data[qs - 1] *= ztest;
                        }

                        eml_xrot(p, V_data, 1 + p * qs, 1 + p * (m - 1), ztest, b);
                    }
                    break;

                case 2:
                    f = e_data[q - 1];
                    e_data[q - 1] = 0.0;
                    for (qs = q; qs + 1 <= m; qs++) {
                        eml_xrotg(&s_data[qs], &f, &ztest, &b);
                        f = -b * e_data[qs];
                        e_data[qs] *= ztest;
                        eml_xrot(n, U_data, 1 + n * qs, 1 + n * (q - 1), ztest, b);
                    }
                    break;

                case 3:
                    varargin_1[0] = fabs(s_data[m - 1]);
                    varargin_1[1] = fabs(s_data[m - 2]);
                    varargin_1[2] = fabs(e_data[m - 2]);
                    varargin_1[3] = fabs(s_data[q]);
                    varargin_1[4] = fabs(e_data[q]);
                    kase = 1;
                    mtmp = varargin_1[0];
                    if (rtIsNaN(varargin_1[0])) {
                        qs = 2;
                        exitg1 = false;
                        while ((!exitg1) && (qs < 6)) {
                            kase = qs;
                            if (!rtIsNaN(varargin_1[qs - 1])) {
                                mtmp = varargin_1[qs - 1];
                                exitg1 = true;
                            } else {
                                qs++;
                            }
                        }
                    }

                    if (kase < 5) {
                        while (kase + 1 < 6) {
                            if (varargin_1[kase] > mtmp) {
                                mtmp = varargin_1[kase];
                            }

                            kase++;
                        }
                    }

                    f = s_data[m - 1] / mtmp;
                    ztest0 = s_data[m - 2] / mtmp;
                    ztest = e_data[m - 2] / mtmp;
                    sqds = s_data[q] / mtmp;
                    b = ((ztest0 + f) * (ztest0 - f) + ztest * ztest) / 2.0;
                    ztest0 = f * ztest;
                    ztest0 *= ztest0;
                    if ((b != 0.0) || (ztest0 != 0.0)) {
                        ztest = sqrt(b * b + ztest0);
                        if (b < 0.0) {
                            ztest = -ztest;
                        }

                        ztest = ztest0 / (b + ztest);
                    } else {
                        ztest = 0.0;
                    }

                    f = (sqds + f) * (sqds - f) + ztest;
                    ztest0 = sqds * (e_data[q] / mtmp);
                    for (qs = q + 1; qs < m; qs++) {
                        eml_xrotg(&f, &ztest0, &ztest, &b);
                        if (qs > q + 1) {
                            e_data[qs - 2] = f;
                        }

                        f = ztest * s_data[qs - 1] + b * e_data[qs - 1];
                        e_data[qs - 1] = ztest * e_data[qs - 1] - b * s_data[qs - 1];
                        ztest0 = b * s_data[qs];
                        s_data[qs] *= ztest;
                        eml_xrot(p, V_data, 1 + p * (qs - 1), 1 + p * qs, ztest, b);
                        s_data[qs - 1] = f;
                        eml_xrotg(&s_data[qs - 1], &ztest0, &ztest, &b);
                        f = ztest * e_data[qs - 1] + b * s_data[qs];
                        s_data[qs] = -b * e_data[qs - 1] + ztest * s_data[qs];
                        ztest0 = b * e_data[qs];
                        e_data[qs] *= ztest;
                        if (qs < n) {
                            eml_xrot(n, U_data, 1 + n * (qs - 1), 1 + n * qs, ztest, b);
                        }
                    }

                    e_data[m - 2] = f;
                    iter++;
                    break;

                default:
                    if (s_data[q] < 0.0) {
                        s_data[q] = -s_data[q];
                        eml_xscal(p, -1.0, V_data, 1 + p * q);
                    }

                    kase = q + 1;
                    while ((q + 1 < mm) && (s_data[q] < s_data[kase])) {
                        ztest = s_data[q];
                        s_data[q] = s_data[kase];
                        s_data[kase] = ztest;
                        if (q + 1 < p) {
                            eml_xswap(p, V_data, 1 + p * q, 1 + p * (q + 1));
                        }

                        if (q + 1 < n) {
                            eml_xswap(n, U_data, 1 + n * q, 1 + n * (q + 1));
                        }

                        q = kase;
                        kase++;
                    }

                    iter = 0;
                    m--;
                    break;
            }
        }
    }

    S_size[0] = minnp;
    for (qs = 0; qs + 1 <= minnp; qs++) {
        S_data[qs] = s_data[qs];
    }
}

static double eml_xnrm2(int n, const double x_data[], int ix0) {
    double y;
    double scale;
    int kend;
    int k;
    double absxk;
    double t;
    y = 0.0;
    if (n < 1) {
    } else if (n == 1) {
        y = fabs(x_data[ix0 - 1]);
    } else {
        scale = 2.2250738585072014E-308;
        kend = (ix0 + n) - 1;
        for (k = ix0; k <= kend; k++) {
            absxk = fabs(x_data[k - 1]);
            if (absxk > scale) {
                t = scale / absxk;
                y = 1.0 + y * t * t;
                scale = absxk;
            } else {
                t = absxk / scale;
                y += t * t;
            }
        }

        y = scale * sqrt(y);
    }

    return y;
}

static void eml_xrot(int n, double x_data[], int ix0, int iy0, double c, double
        s) {
    int ix;
    int iy;
    int k;
    double temp;
    if (n < 1) {
    } else {
        ix = ix0 - 1;
        iy = iy0 - 1;
        for (k = 1; k <= n; k++) {
            temp = c * x_data[ix] + s * x_data[iy];
            x_data[iy] = c * x_data[iy] - s * x_data[ix];
            x_data[ix] = temp;
            iy++;
            ix++;
        }
    }
}

static void eml_xrotg(double *a, double *b, double *c, double *s) {
    double roe;
    double absa;
    double absb;
    double scale;
    double ads;
    double bds;
    roe = *b;
    absa = fabs(*a);
    absb = fabs(*b);
    if (absa > absb) {
        roe = *a;
    }

    scale = absa + absb;
    if (scale == 0.0) {
        *s = 0.0;
        *c = 1.0;
        scale = 0.0;
        *b = 0.0;
    } else {
        ads = absa / scale;
        bds = absb / scale;
        scale *= sqrt(ads * ads + bds * bds);
        if (roe < 0.0) {
            scale = -scale;
        }

        *c = *a / scale;
        *s = *b / scale;
        if (absa > absb) {
            *b = *s;
        } else if (*c != 0.0) {
            *b = 1.0 / *c;
        } else {
            *b = 1.0;
        }
    }

    *a = scale;
}

static void eml_xscal(int n, double a, double x_data[], int ix0) {
    int i15;
    int k;
    i15 = (ix0 + n) - 1;
    for (k = ix0; k <= i15; k++) {
        x_data[k - 1] *= a;
    }
}

static void eml_xswap(int n, double x_data[], int ix0, int iy0) {
    int ix;
    int iy;
    int k;
    double temp;
    ix = ix0 - 1;
    iy = iy0 - 1;
    for (k = 1; k <= n; k++) {
        temp = x_data[ix];
        x_data[ix] = x_data[iy];
        x_data[iy] = temp;
        ix++;
        iy++;
    }
}

void filter2(const double b_data[], const int b_size[2], const emxArray_real_T
        *x, emxArray_real_T *y) {
    int m;
    int n;
    double stencil_data[289];
    int stencil_size[2];
    int i12;
    int j;
    int i;
    boolean_T trysepp;
    boolean_T b0;
    emxArray_real_T *work;
    boolean_T guard1 = false;
    int k;
    boolean_T exitg3;
    int v_size[2];
    double v_data[289];
    int hcol_size[1];
    double hcol_data[17];
    int u_size[2];
    double u_data[289];
    signed char varargin_1[2];
    signed char s_size[2];
    int loop_ub;
    double s_data[289];
    int ko;
    double absxk;
    int exponent;
    double hrow_data[17];
    int mc;
    int nc;
    int nb;
    int32_T exitg2;
    int32_T exitg1;
    m = b_size[0];
    n = b_size[1];
    for (i12 = 0; i12 < 2; i12++) {
        stencil_size[i12] = b_size[i12];
    }


    for (j = 1; j <= n; j++) {
        for (i = 1; i <= m; i++) {
            stencil_data[(i + stencil_size[0] * (j - 1)) - 1] = b_data[(m - i) +
                    b_size[0] * (n - j)];
        }
    }

    if (stencil_size[0] == 1) {
        conv2(stencil_data, stencil_size, x, y);
    } else if (stencil_size[1] == 1) {
        b_conv2(stencil_data, stencil_size, x, y);
    } else {
        trysepp = (stencil_size[0] == 0);
        b0 = (stencil_size[1] == 0);
        emxInit_real_T(&work, 2);
        guard1 = false;
        if ((!(trysepp || b0)) && (stencil_size[0] * stencil_size[1] <= x->size[0] *
                x->size[1])) {
            trysepp = true;
            i12 = stencil_size[0] * stencil_size[1];
            k = 0;
            exitg3 = false;
            while ((!exitg3) && (k <= i12 - 1)) {
                if (!((!rtIsInf(stencil_data[k])) && (!rtIsNaN(stencil_data[k])))) {
                    trysepp = false;
                    exitg3 = true;
                } else {
                    k++;
                }
            }

            if (trysepp) {
                eml_xgesvd(stencil_data, stencil_size, u_data, u_size, hcol_data,
                        hcol_size, v_data, v_size);
                for (i12 = 0; i12 < 2; i12++) {
                    varargin_1[i12] = (signed char) stencil_size[i12];
                }

                j = varargin_1[0];
                s_size[0] = varargin_1[0];
                s_size[1] = varargin_1[1];
                loop_ub = varargin_1[0] * varargin_1[1];
                for (i12 = 0; i12 < loop_ub; i12++) {
                    s_data[i12] = 0.0;
                }

                for (k = 0; k < hcol_size[0]; k++) {
                    s_data[k + s_size[0] * k] = hcol_data[k];
                }

                for (i12 = 0; i12 < 2; i12++) {
                    varargin_1[i12] = s_size[i12];
                }

                ko = varargin_1[0];
                if (varargin_1[1] < varargin_1[0]) {
                    ko = varargin_1[1];
                }

                if ((0 == stencil_size[0]) || (0 == stencil_size[1])) {
                    n = 0;
                } else {
                    m = stencil_size[0];
                    n = stencil_size[1];
                    if (m >= n) {
                        n = m;
                    }
                }

                absxk = fabs(s_data[(ko + j * (ko - 1)) - 1]);
                if ((!rtIsInf(absxk)) && (!rtIsNaN(absxk))) {
                    if (absxk <= 2.2250738585072014E-308) {
                        absxk = 4.94065645841247E-324;
                    } else {
                        frexp(absxk, &exponent);
                        absxk = ldexp(1.0, exponent - 53);
                    }
                } else {
                    absxk = rtNaN;
                }

                absxk *= (double) n;
                m = 0;
                for (k = 0; k < ko; k++) {
                    if (s_data[k + s_size[0] * k] > absxk) {
                        m++;
                    }
                }

                if (m == 1) {
                    absxk = sqrt(s_data[0]);
                    loop_ub = u_size[0];
                    for (i12 = 0; i12 < loop_ub; i12++) {
                        hcol_data[i12] = u_data[i12] * absxk;
                    }

                    exponent = v_size[0];
                    for (i12 = 0; i12 < exponent; i12++) {
                        hrow_data[i12] = v_data[i12] * absxk;
                    }

                    if (x->size[0] < u_size[0] - 1) {
                        mc = 0;
                    } else {
                        mc = (x->size[0] - u_size[0]) + 1;
                    }

                    if (x->size[1] < v_size[0] - 1) {
                        nc = 0;
                    } else {
                        nc = (x->size[1] - v_size[0]) + 1;
                    }

                    nb = x->size[1];
                    m = x->size[1];
                    i12 = work->size[0] * work->size[1];
                    work->size[0] = mc;
                    emxEnsureCapacity((emxArray__common *) work, i12, (int) sizeof (double));
                    i12 = work->size[0] * work->size[1];
                    work->size[1] = m;
                    emxEnsureCapacity((emxArray__common *) work, i12, (int) sizeof (double));
                    m *= mc;
                    for (i12 = 0; i12 < m; i12++) {
                        work->data[i12] = 0.0;
                    }

                    i12 = y->size[0] * y->size[1];
                    y->size[0] = mc;
                    emxEnsureCapacity((emxArray__common *) y, i12, (int) sizeof (double));
                    i12 = y->size[0] * y->size[1];
                    y->size[1] = nc;
                    emxEnsureCapacity((emxArray__common *) y, i12, (int) sizeof (double));
                    m = mc * nc;
                    for (i12 = 0; i12 < m; i12++) {
                        y->data[i12] = 0.0;
                    }

                    if ((u_size[0] == 0) || (v_size[0] == 0) || ((x->size[0] == 0) ||
                            (x->size[1] == 0)) || ((mc == 0) || (nc == 0))) {
                    } else {
                        n = (u_size[0] - x->size[0]) + 1;
                        m = (mc + u_size[0]) - 1;
                        if (u_size[0] <= m) {
                            i12 = u_size[0];
                        } else {
                            i12 = m;
                        }

                        if (n > 1) {
                            k = n - 1;
                        } else {
                            k = 0;
                        }

                        while (k + 1 <= i12) {
                            ko = (k - loop_ub) + 2;
                            if (ko > 0) {
                                m = ko;
                            } else {
                                m = 1;
                            }

                            n = (x->size[0] + ko) - 1;
                            if (n > mc) {
                                n = mc;
                            }

                            if (hcol_data[k] != 0.0) {
                                for (j = 0; j + 1 <= nb; j++) {
                                    for (i = m; i <= n; i++) {
                                        work->data[(i + work->size[0] * j) - 1] += x->data[(i - ko)
                                                + x->size[0] * j] * hcol_data[k];
                                    }
                                }
                            }

                            k++;
                        }

                        n = (v_size[0] - x->size[1]) + 1;
                        m = (nc + v_size[0]) - 1;
                        if (v_size[0] <= m) {
                            i12 = v_size[0];
                        } else {
                            i12 = m;
                        }

                        if (n > 1) {
                            k = n - 1;
                        } else {
                            k = 0;
                        }

                        while (k + 1 <= i12) {
                            ko = (k - exponent) + 2;
                            m = (nb + ko) - 1;
                            if (m > nc) {
                                m = nc;
                            }

                            if (hrow_data[k] != 0.0) {
                                if (ko > 0) {
                                    j = ko - 1;
                                } else {
                                    j = 0;
                                }

                                while (j + 1 <= m) {
                                    n = (j - ko) + 1;
                                    for (i = 0; i + 1 <= mc; i++) {
                                        y->data[i + y->size[0] * j] += work->data[i + work->size[0] *
                                                n] * hrow_data[k];
                                    }

                                    j++;
                                }
                            }

                            k++;
                        }
                    }

                    k = 0;
                    do {
                        exitg2 = 0;
                        if (k <= stencil_size[0] * stencil_size[1] - 1) {
                            if (floor(stencil_data[k]) != stencil_data[k]) {
                                exitg2 = 1;
                            } else {
                                k++;
                            }
                        } else {
                            k = 0;
                            exitg2 = 2;
                        }
                    } while (exitg2 == 0);

                    if (exitg2 == 1) {
                    } else {
                        do {
                            exitg1 = 0;
                            if (k <= x->size[0] * x->size[1] - 1) {
                                if (floor(x->data[k]) != x->data[k]) {
                                    exitg1 = 1;
                                } else {
                                    k++;
                                }
                            } else {
                                b_round(y);
                                exitg1 = 1;
                            }
                        } while (exitg1 == 0);
                    }
                } else {
                    guard1 = true;
                }
            } else {
                guard1 = true;
            }
        } else {
            guard1 = true;
        }

        if (guard1) {
            c_conv2(x, stencil_data, stencil_size, y);
        }

        emxFree_real_T(&work);
    }
}

void histc(const double X[32768], const double edges_data[], const int
        edges_size[2], double N_data[], int N_size[1]) {
    int xind;
    int k;
    int low_i;
    int low_ip1;
    int high_i;
    int mid_i;
    N_size[0] = edges_size[1];
    xind = edges_size[1];
    for (k = 0; k < xind; k++) {
        N_data[k] = 0.0;
    }

    xind = 0;
    for (k = 0; k < 32768; k++) {
        low_i = 0;
        if (!rtIsNaN(X[xind])) {
            if ((X[xind] >= edges_data[0]) && (X[xind] < edges_data[edges_size[1] - 1])) {
                low_i = 1;
                low_ip1 = 2;
                high_i = edges_size[1];
                while (high_i > low_ip1) {
                    mid_i = (low_i >> 1) + (high_i >> 1);
                    if (((low_i & 1) == 1) && ((high_i & 1) == 1)) {
                        mid_i++;
                    }

                    if (X[xind] >= edges_data[mid_i - 1]) {
                        low_i = mid_i;
                        low_ip1 = mid_i + 1;
                    } else {
                        high_i = mid_i;
                    }
                }
            }

            if (X[xind] == edges_data[edges_size[1] - 1]) {
                low_i = edges_size[1];
            }
        }

        if (low_i > 0) {
            N_data[low_i - 1]++;
        }

        xind++;
    }
}

void hist(const emxArray_real_T *Y, const double X_data[], const int X_size[2],
        double no_data[], int no_size[2]) {
    int high_i;
    int low_i;
    int low_ip1;
    double xo_data[4096];
    unsigned int unnamed_idx_1;
    double edges_data[4097];
    int edges_size_idx_1;
    int k;
    double absxk;
    int mid_i;
    double nn_data[4097];
    int xind;
    int32_T exitg1;
    high_i = X_size[1];
    low_i = X_size[1];
    for (low_ip1 = 0; low_ip1 < low_i; low_ip1++) {
        xo_data[low_ip1] = X_data[low_ip1];
    }

    unnamed_idx_1 = (unsigned int) (X_size[1] + 1);
    edges_size_idx_1 = X_size[1] + 1;
    for (k = 0; k <= high_i - 2; k++) {
        edges_data[1 + k] = xo_data[k] + (xo_data[1 + k] - xo_data[k]) / 2.0;
    }

    edges_data[0] = rtMinusInf;
    edges_data[(int) unnamed_idx_1 - 1] = rtInf;
    for (k = 1; k - 1 <= high_i - 2; k++) {
        absxk = fabs(edges_data[k]);
        if ((!rtIsInf(absxk)) && (!rtIsNaN(absxk))) {
            if (absxk <= 2.2250738585072014E-308) {
                absxk = 4.94065645841247E-324;
            } else {
                frexp(absxk, &mid_i);
                absxk = ldexp(1.0, mid_i - 53);
            }
        } else {
            absxk = rtNaN;
        }

        edges_data[k] += absxk;
    }

    low_i = (int) unnamed_idx_1;
    for (low_ip1 = 0; low_ip1 < low_i; low_ip1++) {
        nn_data[low_ip1] = 0.0;
    }

    xind = 0;
    k = 0;
    do {
        exitg1 = 0;
        low_i = Y->size[0];
        if (k <= low_i - 1) {
            low_i = 0;
            if (!rtIsNaN(Y->data[xind])) {
                if ((Y->data[xind] >= edges_data[0]) && (Y->data[xind] <
                        edges_data[edges_size_idx_1 - 1])) {
                    low_i = 1;
                    low_ip1 = 2;
                    high_i = edges_size_idx_1;
                    while (high_i > low_ip1) {
                        mid_i = (low_i >> 1) + (high_i >> 1);
                        if (((low_i & 1) == 1) && ((high_i & 1) == 1)) {
                            mid_i++;
                        }

                        if (Y->data[xind] >= edges_data[mid_i - 1]) {
                            low_i = mid_i;
                            low_ip1 = mid_i + 1;
                        } else {
                            high_i = mid_i;
                        }
                    }
                }

                if (Y->data[xind] == edges_data[edges_size_idx_1 - 1]) {
                    low_i = edges_size_idx_1;
                }
            }

            if (low_i > 0) {
                nn_data[low_i - 1]++;
            }

            xind++;
            k++;
        } else {
            exitg1 = 1;
        }
    } while (exitg1 == 0);

    no_size[0] = 1;
    no_size[1] = edges_size_idx_1 - 1;
    for (k = 0; k <= (int) unnamed_idx_1 - 2; k++) {
        no_data[k] = nn_data[k];
    }

    if (edges_size_idx_1 - 1 > 0) {
        no_data[no_size[1] - 1] += nn_data[edges_size_idx_1 - 1];
    }
}

boolean_T b_isfinite(double x) {
    return (!rtIsInf(x)) && (!rtIsNaN(x));
}

mrdivide::mrdivide() {
}

mrdivide::mrdivide(const mrdivide& orig) {
}

mrdivide::~mrdivide() {
}

void rot90(const double A_data[], const int A_size[2], double B_data[], int
        B_size[2]) {
    int m;
    int n;
    int j;
    int i;
    m = A_size[0];
    n = A_size[1];
    for (j = 0; j < 2; j++) {
        B_size[j] = A_size[j];
    }

    for (j = 1; j <= n; j++) {
        for (i = 1; i <= m; i++) {
            B_data[(i + B_size[0] * (j - 1)) - 1] = A_data[(m - i) + A_size[0] * (n -
                    j)];
        }
    }
}

static double rt_roundd_snf(double u);

static double rt_roundd_snf(double u) {
    double y;
    if (fabs(u) < 4.503599627370496E+15) {
        if (u >= 0.5) {
            y = floor(u + 0.5);
        } else if (u > -0.5) {
            y = u * 0.0;
        } else {
            y = ceil(u - 0.5);
        }
    } else {
        y = u;
    }

    return y;
}

void b_round(emxArray_real_T *x) {
    int i16;
    int k;
    i16 = x->size[0] * x->size[1];
    for (k = 0; k < i16; k++) {
        x->data[k] = rt_roundd_snf(x->data[k]);
    }
}
#define NumBitsPerChar                 8U

real_T rtGetInf(void) {
    size_t bitsPerReal = sizeof (real_T) * (NumBitsPerChar);
    real_T inf = 0.0;
    if (bitsPerReal == 32U) {
        inf = rtGetInfF();
    } else {
        uint16_T one = 1U;

        enum {
            LittleEndian,
            BigEndian
        } machByteOrder = (*((uint8_T *) & one) == 1U) ? LittleEndian : BigEndian;
        switch (machByteOrder) {
            case LittleEndian:
            {

                union {
                    LittleEndianIEEEDouble bitVal;
                    real_T fltVal;
                } tmpVal;

                tmpVal.bitVal.words.wordH = 0x7FF00000U;
                tmpVal.bitVal.words.wordL = 0x00000000U;
                inf = tmpVal.fltVal;
                break;
            }

            case BigEndian:
            {

                union {
                    BigEndianIEEEDouble bitVal;
                    real_T fltVal;
                } tmpVal;

                tmpVal.bitVal.words.wordH = 0x7FF00000U;
                tmpVal.bitVal.words.wordL = 0x00000000U;
                inf = tmpVal.fltVal;
                break;
            }
        }
    }

    return inf;
}

/* Function: rtGetInfF ==================================================
 * Abstract:
 * Initialize rtInfF needed by the generated code.
 * Inf is initialized as non-signaling. Assumes IEEE.
 */
real32_T rtGetInfF(void) {
    IEEESingle infF;
    infF.wordL.wordLuint = 0x7F800000U;
    return infF.wordL.wordLreal;
}

/* Function: rtGetMinusInf ==================================================
 * Abstract:
 * Initialize rtMinusInf needed by the generated code.
 * Inf is initialized as non-signaling. Assumes IEEE.
 */
real_T rtGetMinusInf(void) {
    size_t bitsPerReal = sizeof (real_T) * (NumBitsPerChar);
    real_T minf = 0.0;
    if (bitsPerReal == 32U) {
        minf = rtGetMinusInfF();
    } else {
        uint16_T one = 1U;

        enum {
            LittleEndian,
            BigEndian
        } machByteOrder = (*((uint8_T *) & one) == 1U) ? LittleEndian : BigEndian;
        switch (machByteOrder) {
            case LittleEndian:
            {

                union {
                    LittleEndianIEEEDouble bitVal;
                    real_T fltVal;
                } tmpVal;

                tmpVal.bitVal.words.wordH = 0xFFF00000U;
                tmpVal.bitVal.words.wordL = 0x00000000U;
                minf = tmpVal.fltVal;
                break;
            }

            case BigEndian:
            {

                union {
                    BigEndianIEEEDouble bitVal;
                    real_T fltVal;
                } tmpVal;

                tmpVal.bitVal.words.wordH = 0xFFF00000U;
                tmpVal.bitVal.words.wordL = 0x00000000U;
                minf = tmpVal.fltVal;
                break;
            }
        }
    }

    return minf;
}

/* Function: rtGetMinusInfF ==================================================
 * Abstract:
 * Initialize rtMinusInfF needed by the generated code.
 * Inf is initialized as non-signaling. Assumes IEEE.
 */
real32_T rtGetMinusInfF(void) {
    IEEESingle minfF;
    minfF.wordL.wordLuint = 0xFF800000U;
    return minfF.wordL.wordLreal;
}

/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * rtGetNaN.cpp
 *
 * Code generation for function 'bsif'
 *
 */

#define NumBitsPerChar                 8U

/* Function: rtGetNaN ==================================================
 * Abstract:
 * Initialize rtNaN needed by the generated code.
 * NaN is initialized as non-signaling. Assumes IEEE.
 */
real_T rtGetNaN(void) {
    size_t bitsPerReal = sizeof (real_T) * (NumBitsPerChar);
    real_T nan = 0.0;
    if (bitsPerReal == 32U) {
        nan = rtGetNaNF();
    } else {
        uint16_T one = 1U;

        enum {
            LittleEndian,
            BigEndian
        } machByteOrder = (*((uint8_T *) & one) == 1U) ? LittleEndian : BigEndian;
        switch (machByteOrder) {
            case LittleEndian:
            {

                union {
                    LittleEndianIEEEDouble bitVal;
                    real_T fltVal;
                } tmpVal;

                tmpVal.bitVal.words.wordH = 0xFFF80000U;
                tmpVal.bitVal.words.wordL = 0x00000000U;
                nan = tmpVal.fltVal;
                break;
            }

            case BigEndian:
            {

                union {
                    BigEndianIEEEDouble bitVal;
                    real_T fltVal;
                } tmpVal;

                tmpVal.bitVal.words.wordH = 0x7FFFFFFFU;
                tmpVal.bitVal.words.wordL = 0xFFFFFFFFU;
                nan = tmpVal.fltVal;
                break;
            }
        }
    }

    return nan;
}

/* Function: rtGetNaNF ==================================================
 * Abstract:
 * Initialize rtNaNF needed by the generated code.
 * NaN is initialized as non-signaling. Assumes IEEE.
 */
real32_T rtGetNaNF(void) {
    IEEESingle nanF = {
        { 0}
    };

    uint16_T one = 1U;

    enum {
        LittleEndian,
        BigEndian
    } machByteOrder = (*((uint8_T *) & one) == 1U) ? LittleEndian : BigEndian;
    switch (machByteOrder) {
        case LittleEndian:
        {
            nanF.wordL.wordLuint = 0xFFC00000U;
            break;
        }

        case BigEndian:
        {
            nanF.wordL.wordLuint = 0x7FFFFFFFU;
            break;
        }
    }

    return nanF.wordL.wordLreal;
}
/*
 * Abstract:
 *      MATLAB for code generation function to initialize non-finites,
 *      (Inf, NaN and -Inf).
 */
real_T rtInf;
real_T rtMinusInf;
real_T rtNaN;
real32_T rtInfF;
real32_T rtMinusInfF;
real32_T rtNaNF;

/* Function: rt_InitInfAndNaN ==================================================
 * Abstract:
 * Initialize the rtInf, rtMinusInf, and rtNaN needed by the
 * generated code. NaN is initialized as non-signaling. Assumes IEEE.
 */
void rt_InitInfAndNaN(size_t realSize) {
    (void) (realSize);
    rtNaN = rtGetNaN();
    rtNaNF = rtGetNaNF();
    rtInf = rtGetInf();
    rtInfF = rtGetInfF();
    rtMinusInf = rtGetMinusInf();
    rtMinusInfF = rtGetMinusInfF();
}

/* Function: rtIsInf ==================================================
 * Abstract:
 * Test if value is infinite
 */
boolean_T rtIsInf(real_T value) {
    return ((value == rtInf || value == rtMinusInf) ? 1U : 0U);
}

/* Function: rtIsInfF =================================================
 * Abstract:
 * Test if single-precision value is infinite
 */
boolean_T rtIsInfF(real32_T value) {
    return (((value) == rtInfF || (value) == rtMinusInfF) ? 1U : 0U);
}

/* Function: rtIsNaN ==================================================
 * Abstract:
 * Test if value is not a number
 */
boolean_T rtIsNaN(real_T value) {

#if defined(_MSC_VER) && (_MSC_VER <= 1200)

    return _isnan(value) ? TRUE : FALSE;

#else

    return (value != value) ? 1U : 0U;

#endif

}

/* Function: rtIsNaNF =================================================
 * Abstract:
 * Test if single-precision value is not a number
 */
boolean_T rtIsNaNF(real32_T value) {

#if defined(_MSC_VER) && (_MSC_VER <= 1200)

    return _isnan((real_T) value) ? true : false;

#else

    return (value != value) ? 1U : 0U;

#endif

}

sum::sum() {
}

sum::sum(const sum& orig) {
}

sum::~sum() {
}

/* Function Declarations */
static void b_eml_xaxpy(int n, double a, const double x_data[], int ix0, double
        y_data[], int iy0);
static double b_eml_xnrm2(int n, const double x_data[], int ix0);
static void c_eml_xaxpy(int n, double a, const double x_data[], int ix0, double
        y_data[], int iy0);
static void eml_xaxpy(int n, double a, int ix0, double y_data[], int iy0);
static double eml_xdotc(int n, const double x_data[], int ix0, const double
        y_data[], int iy0);
static void eml_xgesvd(const double A_data[], const int A_size[2], double
        U_data[], int U_size[2], double S_data[], int S_size[1], double V_data[], int
        V_size[2]);
static double eml_xnrm2(int n, const double x_data[], int ix0);
static void eml_xrot(int n, double x_data[], int ix0, int iy0, double c, double
        s);
static void eml_xrotg(double *a, double *b, double *c, double *s);
static void eml_xscal(int n, double a, double x_data[], int ix0);
static void eml_xswap(int n, double x_data[], int ix0, int iy0);

void svd(const double A_data[], const int A_size[2], double U_data[], int
        U_size[2], double S_data[], int S_size[2], double V_data[], int V_size
        [2]) {
    int s_size[1];
    double s_data[17];
    signed char iv0[2];
    int k;
    int loop_ub;
    eml_xgesvd(A_data, A_size, U_data, U_size, s_data, s_size, V_data, V_size);
    for (k = 0; k < 2; k++) {
        iv0[k] = (signed char) A_size[k];
    }

    S_size[0] = iv0[0];
    S_size[1] = iv0[1];
    loop_ub = iv0[0] * iv0[1];
    for (k = 0; k < loop_ub; k++) {
        S_data[k] = 0.0;
    }

    for (k = 0; k < s_size[0]; k++) {
        S_data[k + S_size[0] * k] = s_data[k];
    }
}

//Main.cpp--------------------------------------------------------------------------------------------

#define MAX_UNSIGNED_16_BIT_VALUE 65535.0

/**
 * reads a BSIF-Filter in a double-array
 * @param filter_dims dim-array of the size 3
 * @param filter_data
 * @param filter_path
 * @param filter_name
 * @return 0 if reading was successfull, -1 if filter was not found and -2 if filter-name was wrong 
 */
int read_filter(int filter_dims[], double** filter_data, const char* filter_path, const char* matrix_name) {
    //read filter-----------------------------
    mat_t *matfp = Mat_Open(filter_path, MAT_ACC_RDONLY);
    if (NULL == matfp) return -1;
    matvar_t *matvar = Mat_VarRead(matfp, matrix_name);
    if (NULL == matvar) return -2;

    //-copy dims-------------------------------
    filter_dims[0] = matvar->dims[0];
    filter_dims[1] = matvar->dims[1];
    filter_dims[2] = matvar->dims[2];

    //-copy filter-data------------------------
    int size = filter_dims[0] * filter_dims[1] * filter_dims[2];
    double *mat_data = (double*) (matvar->data);
    double *copied_data = new double[size];
    for (int i = 0; i < size; i++) copied_data[i] = mat_data[i];
    *filter_data = copied_data;

    //clean up--------------------------------
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    return EXIT_SUCCESS;
}

/**
 * calculates the new index of a shifted pixel
 * @param index         the old index
 * @param indexWidth    the row length of this pixel
 * @param shift         the shift
 * @return              the new index
 */
int shiftIndex(const int index, const int indexWidth, const int shift) {
    int indexInRow = index % indexWidth;
    if (indexInRow + shift >= indexWidth) return index + shift - indexWidth;
    else if (indexInRow + shift < 0) return index + shift + indexWidth;
    else return index + shift;
}

/**
 * converts the Mat to a double-array
 * @param data  destination of the double array
 * @param mat   the source mat file
 * @param cell  the cell of the mat which should be converted
 * @param shift the alignment of the pixels
 */
void to_double(double *data, const Mat& mat, const BsifCell& cell, const int shift) {
    int index = 0;
    for (int x = cell.startX; x < cell.endX; x++) {
        for (int y = cell.startY; y < cell.endY; y++) {
            data[index++] = *mat.ptr<uchar>(y, shiftIndex(x, mat.cols, shift));
        }
    }
    return;
}

/**
 * parse the Mat to a typ of emxArray
 * @param mat   the opencv-Mat
 * @param cell  sector of the mat which will be parse
 * @return      a pointer of the created emxArray
 */
emxArray_real_T* parse_img(const Mat& mat, const BsifCell& cell, const int shift) {
    int size[2] = {cell.endY - cell.startY, cell.endX - cell.startX};
    emxArray_real_T *emx_struct = emxCreateND_real_T(2, &size[0]);

    to_double(emx_struct->data, mat, cell, shift);

    return emx_struct;
}

/**
 * normalize a histogram
 * @param normHist  the destination of the normalized histogram
 * @param hist      the source of the disarranged histogram
 * @param length    the length of the histograms
 */
void normalizeHistogram(double* normHist, const double* hist, const int length) {
    int histSum = 0;
    for (int i = 0; i < length; i++) histSum += hist[i];
    for (int i = 0; i < length; i++) normHist[i] = hist[i] / histSum;
}

/**
 * The BSIF feature extraction algorithm
 * @param code          the destinaton of the extracted data
 * @param texture       the iris code which should be extracted
 * @param filter        the filter for the bsif extraction algorithm
 * @param filterDims    the dimensions of the filter
 * @param shift         the alignment for the extraction
 */
void featureExtract(double** histogramPointer, size_t& histogramSize, const Mat& texture, const double filter[], const int filterDims[3], const int shift = 0) {
    histogramSize = (int) pow(2.0, filterDims[2]);

    int histogramDim[2];
    *histogramPointer = new double[histogramSize];
    double* histogram = *histogramPointer;

    BsifCell cell;

    cell.startX = 0;
    cell.startY = 0;
    cell.endX = cell.startX + texture.cols;
    cell.endY = cell.startY + texture.rows;

    for (unsigned int i = 0; i < histogramSize; i++) histogram[i] = 0; //set all values to 0
    
    emxArray_real_T* imgData = parse_img(texture, cell, shift);
    
    bsif(imgData, filter, filterDims, histogram, histogramDim); //entry-point for the matlab-implementation
    
    delete[] imgData->data;
    delete imgData;

    normalizeHistogram(histogram, histogram, histogramSize);

    return;
}

#endif /* BSIF_H */

