//
//  main.c
//  CBLAS
//
//  Created by Lu Zhang on 5/8/16.
//  Copyright © 2016 Lu Zhang. All rights reserved.
//


#include <jni.h>
#include <assert.h>
#include <Matrix.h>

/* Calling fortran blas from libblas */

extern void dscal_(int *n, double *alpha, double *a, int *incx);

extern void daxpy_(int *n, double *alpha, double *x, int *incx,
                   double *y, int *incy);

extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);

extern void dgemv_(char *trans, int *m, int *n, double *alpha, double *A,
                   int *lda, double *x, int *incx, double *beta,
                   double *y, int *incy);

extern void dtrmv_(char *uplo, char *trans, char *diag, int *n, double *A,
                   int *lda, double *x, int *incx);

extern void dsymv_(char *uplo, int *n, double *alpha, double *A, int *lda,
                   double *x, int *incx, double *beta, double *y, int *incy);

extern void dgemm_(char *transa, char *transb,int *m, int *n,
                   int *k, double *alpha, double *A, int *lda,
                   double *B, int *ldb, double *beta, double *C, int *ldc);

extern void dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m,
                   int *n, double *alpha, double *A, int *lda, double *B,
                   int *ldb);

extern void dsymm_(char *side, char *uplo, int *m, int *n, double *alpha,
                   double *A, int *lda, double *B, int *ldb, double *beta,
                   double *C, int *ldc);

#define jniRowMajor 101
#define jniColMajor 102

#define jniNoTrans 111
#define jniTrans   112
#define jniConjTrans    113

#define jniUpper   121
#define jniLower   122

#define jniNonUnit 131
#define jniUnit    132

#define jniLeft    141
#define jniRight   142

/* Level 1: dscal, daxpy, ddot */

JNIEXPORT void Java_JAMAJni_jni_1blas_Matrix_dscal
(JNIEnv *env, jclass klass, jint n, jdouble alpha, jdoubleArray x, jint incx){
    
    /* dscal:  x = alpha * x */
    
    double *xElems;
    xElems = (*env)-> GetDoubleArrayElements (env, x, NULL);
    assert(xElems);
    
    dscal_(&n, &alpha, xElems, &incx);
    
    (*env)-> ReleaseDoubleArrayElements (env, x, xElems, 0);
}


JNIEXPORT void Java_JAMAJni_jni_1blas_Matrix_daxpy
(JNIEnv *env, jclass klass, jint n, jdouble alpha, jdoubleArray x,
 jint incx, jdoubleArray y, jint incy){
    
    /* daxpy: y = alpha * x + y */
    
    double *xElems, *yElems;
    xElems = (*env)-> GetDoubleArrayElements (env, x, NULL);
    yElems = (*env)-> GetDoubleArrayElements (env, y, NULL);
    assert(xElems && yElems);
    
    daxpy_(&n, &alpha, xElems, &incx, yElems, &incy);
    
    (*env)-> ReleaseDoubleArrayElements (env, y, yElems, 0); 
    (*env)-> ReleaseDoubleArrayElements (env, x, xElems, JNI_ABORT);
}

JNIEXPORT jdouble Java_JAMAJni_jni_1blas_Matrix_ddot
(JNIEnv *env, jclass klass, jint n, jdoubleArray x, jint incx,
 jdoubleArray y, jint incy){
    
    /* ddot:  forms the dot product of two vectors x and y.*/
    
    double *xElems, *yElems ;
    double result;
    
    xElems = (*env)-> GetDoubleArrayElements (env, x, NULL);
    yElems = (*env)-> GetDoubleArrayElements (env, y, NULL);
    assert(xElems && yElems);
    
    result = ddot_(&n, xElems, &incx, yElems, &incy);
    
    (*env)-> ReleaseDoubleArrayElements (env, y, yElems, JNI_ABORT);
    (*env)-> ReleaseDoubleArrayElements (env, x, xElems, JNI_ABORT);
    
    return result;
}

/* Level 2: dgemv, dtrmv, dsymv */

JNIEXPORT void Java_JAMAJni_jni_1blas_Matrix_dgemv
(JNIEnv *env, jclass klass, jint Layout, jint Trans, jint m, jint n,
 jdouble alpha, jdoubleArray A, jdoubleArray x, jint incx, jdouble beta,
 jdoubleArray y, jint incy){
    
    /* DGEMV  performs one of the matrix-vector operations
     y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y, */
    
    double *AElems, *xElems, *yElems;
    char Ts;
    
    if (Layout == jniRowMajor){
        
        if (Trans == jniNoTrans) {Ts = 'T';}
        else if (Trans == jniTrans) {Ts = 'N';}
        else if (Trans == jniConjTrans) {Ts = 'N';}
        else {fprintf(stderr, "** Illegal Trans setting \n"); return;}
        
        AElems = (*env)-> GetDoubleArrayElements (env, A, NULL);
        xElems = (*env)-> GetDoubleArrayElements (env, x, NULL);
        yElems = (*env)-> GetDoubleArrayElements (env, y, NULL);
        assert(AElems && xElems && yElems);

        dgemv_(&Ts, &n, &m, &alpha, AElems, &n, xElems, &incx, &beta,
               yElems, &incy);
        
    }
    else if(Layout == jniColMajor){
        
        if (Trans == jniNoTrans) {Ts = 'N';}
        else if (Trans == jniTrans) {Ts = 'T';}
        else if (Trans == jniConjTrans) {Ts = 'C';}
        else {fprintf(stderr, "** Illegal Trans setting \n"); return;}
        
        AElems = (*env)-> GetDoubleArrayElements (env, A, NULL);
        xElems = (*env)-> GetDoubleArrayElements (env, x, NULL);
        yElems = (*env)-> GetDoubleArrayElements (env, y, NULL);
        assert(AElems && xElems && yElems);

        dgemv_(&Ts, &m, &n, &alpha, AElems, &m, xElems, &incx, &beta,
               yElems, &incy);
        
    }
    else {fprintf(stderr, "** Illegal Matrix_Layout setting \n"); return;}
    
    (*env)-> ReleaseDoubleArrayElements (env, y, yElems, 0);
    (*env)-> ReleaseDoubleArrayElements (env, x, xElems, JNI_ABORT);
    (*env)-> ReleaseDoubleArrayElements (env, A, AElems, JNI_ABORT);
}

JNIEXPORT void Java_JAMAJni_jni_1blas_Matrix_dtrmv
(JNIEnv *env, jclass klass, jint Layout, jint Uplo, jint Trans, jint Diag,
 jint n, jdoubleArray A, jdoubleArray x, jint incx){
    
    /*  DTRMV  performs one of the matrix-vector operations
     x := A*x,   or   x := A**T*x,
     where x is an n element vector and  A is an n by n unit, or non-unit,
     upper or lower triangular matrix. */
    
    // LDA specifies the first dimension of A; lda = n
    
    char Ts, uplo, diag;
    
    if (Diag == jniNonUnit) {diag = 'N';}
    else if (Diag == jniUnit) {diag = 'U';}
    else {fprintf(stderr, "** Illegal Diag setting \n"); return;}
    
    if (Layout == jniRowMajor){
        
        if (Trans == jniNoTrans) {Ts = 'T';}
        else if (Trans == jniTrans) {Ts = 'N';}
        else if (Trans == jniConjTrans) {Ts = 'N';}
        else {fprintf(stderr, "** Illegal Trans setting \n"); return;}
        
        if (Uplo == jniUpper) {uplo = 'L';}                // A is an upper triangular matrix.
        else if (Uplo == jniLower) {uplo = 'U';}           // A is a lower triangular matrix
        else {fprintf(stderr, "** Illegal Uplo setting \n"); return;}
        
    }
    else if(Layout == jniColMajor){
        
        if (Trans == jniTrans) {Ts = 'T';}
        else if (Trans == jniNoTrans) {Ts = 'N';}
        else if (Trans == jniConjTrans) {Ts = 'C';}
        else {fprintf(stderr, "** Illegal Trans setting \n"); return;}
        
        if (Uplo == jniUpper) {uplo = 'U';}                // A is an upper triangular matrix.
        else if (Uplo == jniLower) {uplo = 'L';}           // A is a lower triangular matrix
        else {fprintf(stderr, "** Illegal Uplo setting \n"); return;}
    
    }
    else{fprintf(stderr, "** Illegal Matrix_Layout setting \n"); return;}
    
    double *AElems, *xElems;
    
    AElems = (*env)-> GetDoubleArrayElements (env, A, NULL);
    xElems = (*env)-> GetDoubleArrayElements (env, x, NULL);
    
    assert(AElems && xElems);
    
    dtrmv_(&uplo, &Ts, &diag, &n, AElems, &n, xElems, &incx);
    
    (*env)-> ReleaseDoubleArrayElements (env, x, xElems, 0);
    (*env)-> ReleaseDoubleArrayElements (env, A, AElems, JNI_ABORT);
}

JNIEXPORT void Java_JAMAJni_jni_1blas_Matrix_dsymv
(JNIEnv *env, jclass klass, jint Layout, jint Uplo, jint n, jdouble alpha,
 jdoubleArray A, jdoubleArray x, jint incx, jdouble beta,
 jdoubleArray y, int incy){
    
    /*  DSYMV  performs the matrix-vector  operation
     y := alpha*A*x + beta*y
     where alpha and beta are scalars, x and y are n element vectors and
     A is an n by n symmetric matrix */
    char uplo;
    
    // LDA specifies the first dimension of A; lda = n
    if (Layout == jniRowMajor){
        
        //(switch 'U' and 'L' if input is row-major)
        if (Uplo == jniUpper) {uplo = 'L';}                // A is an upper triangular matrix.
        else if (Uplo == jniLower) {uplo = 'U';}           // A is a lower triangular matrix
        else {fprintf(stderr, "** Illegal Uplo setting \n"); return;}
    }
    else if (Layout == jniColMajor) {
        
        if (Uplo == jniUpper) {uplo = 'U';}                // A is an upper triangular matrix.
        else if (Uplo == jniLower) {uplo = 'L';}           // A is a lower triangular matrix
        else {fprintf(stderr, "** Illegal Uplo setting \n"); return;}
    }
    else{fprintf(stderr, "** Illegal Matrix_Layout setting \n"); return;}
    
    double *AElems, *xElems, *yElems;
    
    AElems = (*env)-> GetDoubleArrayElements (env, A, NULL);
    xElems = (*env)-> GetDoubleArrayElements (env, x, NULL);
    yElems = (*env)-> GetDoubleArrayElements (env, y, NULL);
    
    assert(AElems && xElems && yElems);
    
    dsymv_(&uplo, &n, &alpha, AElems, &n, xElems, &incx, &beta, yElems, &incy);
    
    (*env)-> ReleaseDoubleArrayElements (env, y, yElems, 0);
    (*env)-> ReleaseDoubleArrayElements (env, x, xElems, JNI_ABORT);
    (*env)-> ReleaseDoubleArrayElements (env, A, AElems, JNI_ABORT);
    
}


/* Level 3: dgemm, dtrmm, dsymm */

JNIEXPORT void Java_JAMAJni_jni_1blas_Matrix_dgemm
(JNIEnv *env, jclass klass, jint Layout, jint TransA, jint TransB, jint m,
 jint n, jint k, jdouble alpha, jdoubleArray  A, jdoubleArray B,
 jdouble beta, jdoubleArray C){
    /* DGEMM  performs one of the matrix-matrix operations
     C := alpha*op( A )*op( B ) + beta*C,
     where  op( X ) is one of op( X ) = X   or   op( X ) = X**T,
     alpha and beta are scalars, and A, B and C are matrices, with op( A )
     an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix. */
    
    double *aElems, *bElems, *cElems;
    int lda, ldb;
    char TA, TB;
    
    if (Layout == jniRowMajor){
        
        // if is row-major; switch A and B;
        // swith m and n; alter the value of transa and transb
        if (TransA == jniNoTrans) {TA = 'N'; lda = k;}
        else if (TransA == jniTrans) {TA = 'T'; lda = m;}
        else if (TransA == jniConjTrans) {TA = 'C'; lda = m;}
        else {fprintf(stderr, "** Illegal TransA setting \n"); return;}
        
        if (TransB == jniNoTrans) { TB = 'N'; ldb = n;}
        else if (TransB == jniTrans) {TB = 'T'; ldb = k;}
        else if (TransB == jniConjTrans) {TB = 'C'; ldb = k;}
        else {fprintf(stderr, "** Illegal TransB setting \n"); return;}
        
        aElems = (*env)-> GetDoubleArrayElements (env,A,NULL);
        bElems = (*env)-> GetDoubleArrayElements (env,B,NULL);
        cElems = (*env)-> GetDoubleArrayElements (env,C,NULL);
        
        assert(aElems && bElems && cElems);

        dgemm_(&TB, &TA, &n, &m, &k, &alpha, bElems, &ldb, aElems, &lda, &beta,
               cElems, &n);
        
    }
    else if(Layout == jniColMajor){
        
        if (TransA == jniNoTrans) { TA = 'N'; lda = m;}
        else if (TransA == jniTrans) {TA = 'T'; lda = k;}
        else if (TransA == jniConjTrans) {TA = 'C'; lda = k;}
        else {fprintf(stderr, "** Illegal TransA setting \n"); return;}
        
        if (TransB == jniNoTrans) { TB = 'N'; ldb = k;}
        else if (TransB == jniTrans) { TB = 'T'; ldb = n;}
        else if (TransB == jniConjTrans) {TB = 'C'; ldb = n;}
        else {fprintf(stderr, "** Illegal TransB setting \n"); return;}
        
        aElems = (*env)-> GetDoubleArrayElements (env,A,NULL);
        bElems = (*env)-> GetDoubleArrayElements (env,B,NULL);
        cElems = (*env)-> GetDoubleArrayElements (env,C,NULL);
        
        assert(aElems && bElems && cElems);
        
        dgemm_(&TA, &TB, &m, &n, &k, &alpha, aElems, &lda, bElems, &ldb,
               &beta, cElems, &m);
        
    }
    else{fprintf(stderr, "** Illegal Matrix_Layout setting \n"); return;}
    
    (*env)-> ReleaseDoubleArrayElements (env, C, cElems, 0);
    (*env)-> ReleaseDoubleArrayElements (env, B, bElems, JNI_ABORT);
    (*env)-> ReleaseDoubleArrayElements (env, A, aElems, JNI_ABORT);
}


JNIEXPORT void Java_JAMAJni_jni_1blas_Matrix_dtrmm
(JNIEnv *env, jclass klass, jint Layout, jint Side, jint Uplo, jint TransA,
 jint Diag, jint m, jint n, jdouble alpha, jdoubleArray  A, jdoubleArray B){
    
    /* dtrmm: performs one of the matrix-matrix operations
     B := alpha*op( A )*B,   or   B := alpha*B*op( A )
     where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
     non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
     op( A ) = A   or   op( A ) = A**T.*/
    
    double *aElems, *bElems;
    
    int lda;
    char side, uplo, TA, diag;
    
    if (TransA == jniNoTrans) {TA = 'N';}
    else if (TransA == jniTrans) {TA = 'T';}
    else if (TransA == jniConjTrans) {TA = 'C';}
    else {fprintf(stderr, "** Illegal TransA setting \n"); return;}
    
    if (Diag == jniNonUnit) {diag = 'N';}
    else if (Diag == jniUnit) {diag = 'U';}
    else {fprintf(stderr, "** Illegal Diag setting \n"); return;}
    
    if (Layout == jniRowMajor){
        
        if (Side == jniLeft){ side = 'R'; lda = m;}        // B := alpha*op( A )*B
        else if(Side == jniRight){side = 'L'; lda = n;}    // B := alpha*B*op( A )
        else{fprintf(stderr, "** Illegal Side setting \n"); return;}
        
        if (Uplo == jniUpper){ uplo = 'L';}                // B := alpha*op( A )*B
        else if(Uplo == jniLower){ uplo = 'U';}
        else{fprintf(stderr, "** Illegal Uplo setting \n"); return;}
        
        aElems = (*env)-> GetDoubleArrayElements (env,A, NULL);
        bElems = (*env)-> GetDoubleArrayElements (env,B, NULL);
        
        assert(aElems && bElems);
        
        dtrmm_(&side, &uplo, &TA, &diag, &n, &m, &alpha, aElems, &lda,
               bElems, &n);
        
    }
    else if(Layout == jniColMajor){
        
        if (Side == jniLeft){ side = 'L'; lda = m;}        // B := alpha*op( A )*B
        else if(Side == jniRight){side = 'R'; lda = n;}    // B := alpha*B*op( A )
        else{fprintf(stderr, "** Illegal Side setting \n"); return;}
        
        if (Uplo == jniUpper){ uplo = 'U';}                // B := alpha*op( A )*B
        else if(Uplo == jniLower){ uplo = 'L';}
        else{fprintf(stderr, "** Illegal Uplo setting \n"); return;}
        
        aElems = (*env)-> GetDoubleArrayElements (env,A, NULL);
        bElems = (*env)-> GetDoubleArrayElements (env,B, NULL);
        
        assert(aElems && bElems);
        
        dtrmm_(&side, &uplo, &TA, &diag, &m, &n, &alpha, aElems, &lda,
               bElems, &m);
        
    }
    else{fprintf(stderr, "** Illegal Matrix_Layout setting \n"); return;}
    
    (*env)-> ReleaseDoubleArrayElements (env, B, bElems, 0);
    (*env)-> ReleaseDoubleArrayElements (env, A, aElems, JNI_ABORT);
    
}

JNIEXPORT void Java_JAMAJni_jni_1blas_Matrix_dsymm
(JNIEnv *env, jclass klass, jint Layout, jint Side, jint Uplo,
 jint m, jint n, jdouble alpha, jdoubleArray  A, jdoubleArray B,
 jdouble beta, jdoubleArray C){
    
    /*DSYMM  performs one of the matrix-matrix operations:
     C := alpha*A*B + beta*C, or C := alpha*B*A + beta*C,
     where A is a symmetric matrix and  B and C are  m by n matrices.*/
    
    double *aElems, *bElems, *cElems;
    int lda;
    char side, uplo;
    
    if (Layout == jniRowMajor){
        
        if (Side == jniLeft){ side = 'R'; lda = m;}        // C := alpha*A*B + beta*C
        else if(Side == jniRight){side = 'L'; lda = n;}
        else{fprintf(stderr, "** Illegal Side setting \n"); return;}
        
        if (Uplo == jniUpper){ uplo = 'L';}
        // Only the upper triangular part of A is to be referenced

        else if(Uplo == jniLower){ uplo = 'U';}
        // Only the lower triangular part of B is to be referenced
        else{fprintf(stderr, "** Illegal Uplo setting \n"); return;}
        
        aElems = (*env)-> GetDoubleArrayElements (env,A, NULL);
        bElems = (*env)-> GetDoubleArrayElements (env,B, NULL);
        cElems = (*env)-> GetDoubleArrayElements (env,C, NULL);
        
        assert(aElems && bElems && cElems);
        
        dsymm_(&side, &uplo, &n, &m, &alpha, aElems, &lda, bElems, &n, &beta, cElems, &n);
        
    }
    else if(Layout == jniColMajor){
        
        if (Side == jniLeft){ side = 'L'; lda = m;}        // C := alpha*A*B + beta*C
        else if(Side == jniRight){side = 'R'; lda = n;}    //
        else{fprintf(stderr, "** Illegal Side setting \n"); return;}
        
        if (Uplo == jniUpper){ uplo = 'U';}                // B := alpha*op( A )*B
        else if(Uplo == jniLower){ uplo = 'L';}
        else{fprintf(stderr, "** Illegal Uplo setting \n"); return;}
        
        aElems = (*env)-> GetDoubleArrayElements (env,A, NULL);
        bElems = (*env)-> GetDoubleArrayElements (env,B, NULL);
        cElems = (*env)-> GetDoubleArrayElements (env,C, NULL);
        
        assert(aElems && bElems && cElems);
        dsymm_(&side, &uplo, &m, &n, &alpha, aElems, &lda, bElems, &m, &beta,
               cElems, &m);
        
    }
    else{fprintf(stderr, "** Illegal Matrix_Layout setting \n");}
    
    (*env)-> ReleaseDoubleArrayElements (env, C, cElems, 0);
    (*env)-> ReleaseDoubleArrayElements (env, A, aElems, JNI_ABORT);
    (*env)-> ReleaseDoubleArrayElements (env, B, bElems, JNI_ABORT);
    
}






















