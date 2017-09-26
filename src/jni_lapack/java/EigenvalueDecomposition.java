package JAMAJni.jni_lapack;

import JAMAJni.jni_blas.*;


public class EigenvalueDecomposition  implements java.io.Serializable {
 private EigenvalueDecomposition() {}
 static {
     
    /* load library (which will contain wrapper for cblas function.)*/
    System.loadLibrary("lapack_lite_EigenvalueDecomposition");
 
 }

 /* ------------------------
  * Class variables
  * ------------------------ */
   /** Row and column dimension (square matrix).*/
   private int n;

   /** Symmetry flag.*/
   private boolean issymmetric;

   /**Array for elements storage of A */
   private double[] a;

   /**Array for eigenvalue of A */
   private double[] w;
  /* ------------------------
   * Constructor
   * ------------------------ */
      public EigenvalueDecomposition (Matrix A) {
      a = A.getRowPackedCopy();
      n = A.getColumnDimension();
      char jobvl = EigenvalueDecomposition.JOBV.Compute;
      char uplo = EigenvalueDecomposition.UPLO.Upper;
      int matrix_layout =  EigenvalueDecomposition.LAYOUT.RowMajor;
      int lda = n;
      double[] w = new double[n];
      int lwork = 4*n+1;
      double[] work = new double[lwork];
      int[] info = new int[]{0};
      dsyev(matrix_layout, jobvl, uplo, n, a, lda, w, work, lwork, info);
      }

    /* ------------------------
     * Public Methods
     * ------------------------ */
      /** Return the eigenvector matrix */
      public Matrix getV () {
	     Matrix X = new Matrix(n,n);
             double[][] V = X.getArray();
	     for (int i = 0; i < n; i++) {
		     for (int j = 0; j < n; j++) {
			     V[i][j] = a[i*n+j];
      }
	     }
	    return X;
      }

      /**Return the singular values of A */
      public double[] getD () {
             return w;
      }


    public final static class LAYOUT {
        private LAYOUT() {}
        public final static int RowMajor = 101;
        public final static int ColMajor = 102;
    }
    
    public final static class TRANSPOSE {
        private TRANSPOSE() {}
        public final static char NoTrans = 'N';         /** trans='N' */
        public final static char Trans= 'T';            /** trans='T' */
        public final static char ConjTrans= 'C';        /** trans='C' */
    }
    
    public final static class UPLO {
        private UPLO() {}
        public final static char Upper = 'U';           /** Upper triangular matrix */
        public final static char Lower= 'L';            /** Lower triangular matrix*/
    }
    
    public final static class JOBV {
        private JOBV() {}
        public final static char NoCompute = 'N';       /** eigenvectors are not computed */
        public final static char Compute= 'V';          /** eigenvectors are computed*/
    }
    
    public final static class JOB {
        private JOB() {}
        public final static char All = 'A';             /** all M columns of U are returned in array U */
        public final static char firstInU = 'S';        /** the first min(m,n) columns of U (the left singular
                                                         vectors) are returned in the array U;*/
        public final static char Overwritten = 'O';     /** the first min(m,n) columns of U (the left singular
                                                         vectors) are overwritten on the array A; */
        public final static char NoCompute = 'N';       /** no columns of U (no left singular vectors) are
                                                         computed.*/
    }
    
    public final static class ITYPE {
	private ITYPE(){}
	public final static int first = 1;
	public final static int second = 2;
	public final static int third = 3;
    }
    
    
    /* Eigenvector and SVD */
    public static native int dgeev(int matrix_layout, char jobvl, char jobvr,
                                   int n, double[] a, int lda, double[] wr, double[] wi,
                                   double[] vl, int ldvl, double[] vr, int ldvr,
                                   double[] work, int lwork, int[] info);
    
    public static native int dgesvd(int matrix_layout, char jobu, char jobvt,
                                    int m, int n, double[] a, int lda, double[] s,
                                    double[] u, int ldu, double[] vt, int ldvt,
                                    double[] work, int lwork, int[] info);
    
    public static native int dgesdd(int matrix_layout, char jobz, int m, int n,
                                    double[] a, int lda, double[] s, double[] u, int ldu,
                                    double[] vt, int ldvt, double[] work, int lwork,
                                    int[] iwork, int[] info);
    
    public static native void dsyev(int matrix_layout, char jobz, char uplo, int n,
                                    double[] a, int lda, double[] w, double[] work,
                                    int lwork, int[] info);
 

    public static native int dgesv( int matrix_layout, int n, int nrhs, double[] a,
                                   int lda, int[] ipiv, double[] b, int ldb, int[] info);

    public static native int dggev( int matrix_layout, char jobvl, char jobvr, int n,
                                   double[] a, int lda, double[] b, int ldb, double[] alphar,
                                   double[] alphai, double[] beta, double[] vl, int ldvl,
                                   double[] vr, int ldvr, double[] work, int lwork, int[] info);

    public static native int dsygv( int matrix_layout, int itype, char jobz, char uplo,
                                   int n, double[] a, int lda, double[] b, int ldb,
                                   double[] w, double[] work, int lwork, int[] info );

    /**inform java virtual machine that function is defined externally*/
    
}







