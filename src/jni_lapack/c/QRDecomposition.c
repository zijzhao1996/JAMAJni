#include <jni.h>
#include <assert.h>
#include <QRDecomposition.h>
#include <stdlib.h>

/* two functions that deal with matrix layout */
void CRswitch (double *in, double *out, int m, int n, int ldin, int ldout);

void RCswitch (double *in, double *out, int m, int n, int ldin, int ldout);

double *create_vectord (int dim);

/* Calling fortran lapack from liblapack */


extern int dgeqrf_(int *m, int *n, double *a, int *lda, double *tau,
                   double *work, int *lwork, int *info);

extern int dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau,
                   double *work, int *lwork, int *info);

extern int dgeqp3_(int *m, int *n, double *a, int *lda, int *jpvt, double *tau,
                   double *work, int *lwork, int *info);

#define jniRowMajor 101
#define jniColMajor 102




JNIEXPORT jint JNICALL Java_JAMAJni_jni_1lapack_QRDecomposition_dgeqrf
  (JNIEnv *env, jclass obj, jint layout, jint m, jint n, jdoubleArray ja,
   jint lda, jdoubleArray jtau, jdoubleArray jwork, jint lwork,
   jintArray jinfo)
{
    double *a = (*env)-> GetDoubleArrayElements (env, ja, NULL);
    double *tau = (*env)-> GetDoubleArrayElements (env, jtau, NULL);
    double *work = (*env)-> GetDoubleArrayElements (env, jwork, NULL);
    int *info = (*env)-> GetIntArrayElements(env,jinfo,NULL);
    int result;
  
    if (layout == jniColMajor){
        result = dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
    } else if (layout == jniRowMajor) {
        if (lda < n) {
            fprintf(stderr, "** Illegal value of lda for row-major layout\n");
            return -1;
        }
          
        double *a_t = create_vectord(m*n);
        int lda_t = m;
        RCswitch(a, a_t, m, n, lda, lda_t);
        result = dgeqrf_(&m, &n, a_t, &lda_t, tau, work, &lwork, info);
        CRswitch(a_t, a, m, n, lda_t, lda);
        free(a_t);
    } else {
        fprintf(stderr, "** Illegal layout setting\n");
        return -1;
    }
      
    (*env)-> ReleaseDoubleArrayElements (env, ja, a, 0);
    (*env)-> ReleaseDoubleArrayElements (env, jtau, tau, 0);
    (*env)-> ReleaseDoubleArrayElements (env, jwork, work, 0);
    (*env)-> ReleaseIntArrayElements (env, jinfo, info, 0);

    return result;
}


JNIEXPORT jint JNICALL Java_JAMAJni_jni_1lapack_QRDecomposition_dorgqr
  (JNIEnv *env, jclass obj, jint layout, jint m, jint n, jint k,
   jdoubleArray ja, jint lda, jdoubleArray jtau, jdoubleArray jwork,
   jint lwork, jintArray jinfo)
{
    double *a = (*env)-> GetDoubleArrayElements (env, ja, NULL);
    double *tau = (*env)-> GetDoubleArrayElements (env, jtau, NULL);
    double *work = (*env)-> GetDoubleArrayElements (env, jwork, NULL);
    int *info = (*env)-> GetIntArrayElements(env,jinfo,NULL);
    int result;
  
    if (layout == jniColMajor) {
        result = dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
    } else if (layout == jniRowMajor) {
        if (lda < n) {
            fprintf(stderr, "** Illegal value of lda for row-major layout\n");
            return -1;
        }
          
        double *a_t = create_vectord(m*n);
        int lda_t = m;
        RCswitch(a, a_t, m, n, lda, lda_t);
        result = dorgqr_(&m, &n, &k, a_t, &lda_t, tau, work, &lwork, info);
        CRswitch(a_t, a, m, n, lda_t, lda);
        free(a_t);
    } else {
        fprintf(stderr, "** Illegal layout setting\n");
        return -1;
    }

    (*env)-> ReleaseDoubleArrayElements (env, ja, a, 0);
    (*env)-> ReleaseDoubleArrayElements (env, jtau, tau, JNI_ABORT);
    (*env)-> ReleaseDoubleArrayElements (env, jwork, work, 0);
    (*env)-> ReleaseIntArrayElements (env, jinfo, info, 0);

    return result;
}


JNIEXPORT jint JNICALL Java_JAMAJni_jni_1lapack_QRDecomposition_dgeqp3
  (JNIEnv *env, jclass obj, jint layout, jint m, jint n, jdoubleArray ja,
   jint lda, jintArray jjpvt, jdoubleArray jtau, jdoubleArray jwork,
   jint lwork, jintArray jinfo)
{
    double *a = (*env)-> GetDoubleArrayElements (env, ja, NULL);
    int *jpvt = (*env)-> GetIntArrayElements (env, jjpvt, NULL);
    double *tau = (*env)-> GetDoubleArrayElements (env, jtau, NULL);
    double *work = (*env)-> GetDoubleArrayElements (env, jwork, NULL);
    int *info = (*env)-> GetIntArrayElements(env,jinfo,NULL);
    int result;
  
    if (layout == jniColMajor){
        result = dgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, info);
    } else if (layout == jniRowMajor) {
        if (lda < n) {
            fprintf(stderr, "** Illegal value of lda for row-major layout\n");
            return -1;
        }
          
        double *a_t = create_vectord(m*n);
        int lda_t = m;
        RCswitch(a, a_t, m, n, lda, lda_t);
        result = dgeqp3_(&m, &n, a_t, &lda_t, jpvt, tau, work, &lwork, info);
        CRswitch(a_t, a, m, n, lda_t, lda);
        free(a_t);
    } else {fprintf(stderr, "** Illegal layout setting\n");  return -1;}

    (*env)-> ReleaseDoubleArrayElements (env, ja, a, 0);
    (*env)-> ReleaseDoubleArrayElements (env, jtau, tau, 0);
    (*env)-> ReleaseDoubleArrayElements (env, jwork, work, 0);
    (*env)-> ReleaseIntArrayElements (env, jjpvt, jpvt, 0);
    (*env)-> ReleaseIntArrayElements (env, jinfo, info, 0);
  
    return result;
}




/* switch row-major and col-major*/
void RCswitch (double *in, double *out, int m, int n, int ldin, int ldout)
{
    int i,j;
    
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            out[j * ldout + i] = in[i * ldin + j];
        }
    }
    
}


void CRswitch (double *in, double *out, int m, int n, int ldin, int ldout)
{
    int i, j;

    for(i = 0; i < m; i++)
    {
        for( j = 0; j < n; j++)
        {
            out[i * ldout + j] = in[j * ldin + i];
        }
    }

}


double *create_vectord (int dim)
{
  double *vector;
  
  vector = (double *) malloc (dim * sizeof (double));
  assert(vector);

  return vector;
}

