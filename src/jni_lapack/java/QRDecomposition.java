package JAMAJni.jni_lapack;

import JAMAJni.jni_blas.*;


public class QRDecomposition implements java.io.Serializable {
 private QRDecomposition() {}
 static {
     
    /* load library (which will contain wrapper for cblas function.)*/
    System.loadLibrary("lapack_lite_QRDecomposition");
 }
     
    /* ------------------------
     *  Class variables
     *  ------------------------ */
    /** Array for internal storage of decomposition.*/
    private double[] QR,QR2,QR3;
    /** Row and column dimensions.*/
    private int m, n;
    /** Array for internal storage of diagonal of R.*/
    private double[] Rdiag;

    /* ------------------------
     Constructor
     * ------------------------ */
    public QRDecomposition (Matrix A) {
	    QR = A.getRowPackedCopy();
	    m = A.getRowDimension();
	    n = A.getColumnDimension();
	    int lda = n;
	    int[] jpvt = new int[] {1, 2, 3};
	    double[] tau;
	    if (m>n) {
		    tau = new double[n];
	    }
		    else{
	            tau = new double[m]; }	    
	    int lwork = 4*n+1;
	    double[] work = new double[lwork];
            int k = m;	    
	    int [] info = new int [1];
	    int matrix_layout = QRDecomposition.LAYOUT.RowMajor;
	    dgeqrf(matrix_layout, m, n, QR, lda, tau, work, lwork, info);
	    QR2 = QR;
            dorgqr(matrix_layout, m, n, k, QR2, lda, tau, work, lwork, info);
	    QR3 = A.getRowPackedCopy();
	    dgeqp3(matrix_layout, m, n, QR3, lda, jpvt, tau, work, lwork, info);
             
    }

     /* ------------------------
      * Public Methods
      * ------------------------ */
    /** Return the upper triangular factor*/
       public Matrix getR () {
            Matrix X = new Matrix(m,n);
	    double[][] R = X.getArray();
	    for (int i = 0; i < n; i++) {
		    for (int j = 0; j < n; j++) {
			    R[i][j] = QR3[i*m+j];
		    }
	    }
	    return X;
       }


   /** Generate and return the (economy-sized) orthogonal factor*/
       public Matrix getQ () {
       Matrix X = new Matrix(m,n);
       double[][] Q = X.getArray();
       for (int i = 0; i < n; i++) {
	       for (int j = 0; j < n; j++) {
		       Q[i][j] = QR2[i*m+j];
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
    
    
    /* QR */
    public static native int dgeqrf(int matrix_layout, int m, int n, double[] a,
                                    int lda, double[] tau, double[] work, int lwork,
                                    int[] info);
    
    public static native int dorgqr(int matrix_layout, int m, int n, int k,
                                    double[] a, int lda, double[] tau, double[] work,
                                    int lwork, int[] info);
    
    public static native int dgeqp3(int matrix_layout, int m, int n, double[] a,
                                    int lda, int[] jpvt, double[] tau, double[] work,
                                    int lwork, int[] info);
    
    

    /**inform java virtual machine that function is defined externally*/
    
}







