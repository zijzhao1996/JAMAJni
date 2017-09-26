package JAMAJni.jni_lapack;

import JAMAJni.jni_blas.*;

public class CholeskyDecomposition implements java.io.Serializable {
	private CholeskyDecomposition() {}
	static {
 /* load library (which will contain wrapper for cblas function.)*/
         System.loadLibrary("lapack_lite_CholeskyDecomposition");
	}

 /* ------------------------
  *  Class variables
  *  ------------------------ */
  /** Array for internal storage of decomposition.*/
	private double[] l;

  /** Row and column dimension (square matrix).*/
	private int n;

  /** Symmetric and positive definite flag.*/
	private boolean isspd;

  /* ------------------------
   * Constructor
   * ------------------------ */
	public CholeskyDecomposition (Matrix A) {
		 l = A.getRowPackedCopy();
		 n = A.getRowDimension();
		 int matrix_layout = CholeskyDecomposition.LAYOUT.RowMajor;
		 char uplo = CholeskyDecomposition.UPLO.Upper;
		 int lda = n;
		 int [] info = new int [1];
		 dpotrf(matrix_layout, uplo, n, l, lda, info);
         }

   /* ------------------------
    * Public Methods
    * ------------------------ */
        /** Return triangular factor.*/
	public Matrix getL () {
		Matrix X = new Matrix(n,n);
		double[][] L = X.getArray();
                for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				L[i][j] = this.l[i*n+j];
			}
		}
		return X;
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
	 public final static char firstInU = 'S';        /** the first min(m,n) columns of U (the left singular vectors) are returned in the array U;*/
	 public final static char Overwritten = 'O';     /** the first min(m,n) columns of U (the left singularvectors) are overwritten on the array A; */
	 public final static char NoCompute = 'N';       /** no columns of U (no left singular vectors) are computed.*/
 }

 public final static class ITYPE {
	 private ITYPE(){}
	 public final static int first = 1;
	 public final static int second = 2;
	 public final static int third = 3;
 }

 /* Cholesky */
 public static native void dpotrf(int matrix_layout, char uplo, int n,
		                                      double[] a, int lda, int[] info);
 public static native int dpotri(int matrix_layout, char uplo, int n,
		                                     double[] a, int lda, int[] info);
 public static native int dpotrs(int matrix_layout, char uplo, int n,
		                                     int nrhs, double[] a, int lda, double[] b,
						                                         int ldb, int[] info);
 /**inform java virtual machine that function is defined externally*/

}




