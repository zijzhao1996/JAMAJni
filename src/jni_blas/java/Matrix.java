package JAMAJni.jni_blas;

/**CBLAS.java*/

public class Matrix implements Cloneable, java.io.Serializable {
 private Matrix() {}
 static {
     
    /* load library (which will contain wrapper for cblas function.)*/
    System.loadLibrary("blas_lite");
 
 }
    
    /**inform java virtual machine that function is defined externally*/
    
    
    /* -----------------------
     * Class variables
     * ---------------------- */
    
    private double[][] A;
    
    private int m, n;
    
    /* -----------------------
     * Constructors
     * ----------------------- */
    
    /** Construct an m-by-n matrix of zeros. */
    public Matrix (int m, int n) {
        this.m = m;
        this.n = n;
        A = new double[m][n];
    }
    
    /** Construct an m-by-n constant matrix. */
    public Matrix (int m, int n, double s) {
        this.m = m;
        this.n = n;
        A = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = s;
            }
        }
    }
    
    /** Construct a matrix from a 2-D array. */
    public Matrix (double[][] A) {
        m = A.length;
        n = A[0].length;
        for (int i = 0; i < m; i++) {
            if (A[i].length != n) {
                throw new IllegalArgumentException("All rows must have the same length.");
            }
        }
        this.A = A;
    }
    
    /** Construct a matrix quickly without checking arguments. */
    public Matrix (double[][] A, int m, int n) {
        this.A = A;
        this.m = m;
        this.n = n;
    }
    
    /** Construct a matrix from a one-dimensional packed array */
    public Matrix (double vals[], int m) {
        this.m = m;
        n = (m != 0 ? vals.length/m : 0);
        if (m*n != vals.length) {
            throw new IllegalArgumentException("Array length must be a multiple of m.");
        }
        A = new double[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                A[i][j]=vals[i*m+j];
            }
        }
    }
    
    /* ------------------------
     *    Public Methods
     * ------------------------ */
    
    /** Construct a matrix from a copy of a 2-D array.*/
    
    public static Matrix constructWithCopy(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        Matrix X = new Matrix(m,n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            if (A[i].length != n) {
                throw new IllegalArgumentException
                ("All rows must have the same length.");
            }
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j];
            }
        }
        return X;
    }
    
    /** Access the internal two-dimensional array. */
    public double[][] getArray () {
        return A;
    }

     /** Copy the internal two-dimensional array.*/
     public double[][] getArrayCopy () {
	 double[][] C = new double[m][n];
         for (int i = 0; i < m; i++) {	 
		 for (int j = 0; j < n; j++) {
			 C[i][j] = A[i][j];
		 }
	 }
	 return C;
     }

    /** Make a one-dimensional row packed copy of the internal array. */
    public double[] getRowPackedCopy() {
        double[] vals = new double[m*n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                vals[i*n+j] = A[i][j];
            }
        }
        return vals;
    }

   /** Make a one-dimensional column packed copy of the internal array.*/
   public double[] getColumnPackedCopy () {
	  double[] vals = new double[m*n];
	 for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
		       vals[i+j*m] = A[i][j];
		}
	 }
    return vals;
   }   

    /** Get row dimension.*/
    public int getRowDimension () {
        return m;
    }
    /** Get column dimension.*/
    public int getColumnDimension () {
        return n;
    }
    /** Get a single element.*/
    public double get (int i, int j) {
        return A[i][j];
    }
    
    /*Declaration*/
    
    public static Matrix times (Matrix A, double alpha){
        double[] a = A.getColumnPackedCopy();
        dscal(A.getRowDimension()*A.getColumnDimension(), alpha, a, 1);
        Matrix B = new Matrix(a, A.getRowDimension());
        return B;
    } 

    public static Matrix times (Matrix A, Matrix B) {
	double[] a = A.getColumnPackedCopy();
	double[] b = B.getColumnPackedCopy();
	Matrix C = new Matrix (A.getRowDimension(), B.getColumnDimension());
	double[] c = C.getColumnPackedCopy();
	dgemm(Matrix.LAYOUT.ColMajor, Matrix.TRANSPOSE.NoTrans, Matrix.TRANSPOSE.NoTrans,
              A.getRowDimension(), B.getColumnDimension(), A.getColumnDimension(),
	      1, a, b, 0, c);
        Matrix X = new Matrix(c, A.getRowDimension());
	return X;
    }


    public static class LAYOUT {
        private LAYOUT() {}
        /** row-major arrays */
        public final static int RowMajor= 101;
        /** column-major arrays */
        public final static int ColMajor= 102;
    }
    
    public static class TRANSPOSE {
        private TRANSPOSE() {}
        /** trans = 'N' */
        public final static int NoTrans = 111;
        /** trans = 'T' */
        public final static int Trans= 112;
        /** trans = 'C'*/
        public final static int ConjTrans= 113;
    }

    public static class UPLO {
        private UPLO() {}
        /** Upper triangular matrix */
        public final static int Upper = 121;
        /** Lower triangular matrix*/
        public final static int Lower = 122;
    }
    
    public static class DIAG {
        private DIAG() {}
        /** not assumed to be unit  */
        public final static int NonUnit = 131;
        /** assumed to be unit */
        public final static int Unit = 132;
    }
    
    public static class SIDE {
        private SIDE() {}
        /** B := alpha*op( A )*B */
        public final static int Left = 141;
        /** B := alpha*B*op( A ) */
        public final static int Right = 142;
    }
    
    /* Level 1: */
    public static native void dscal( int n, double alpha, double[] x, int incx);
    
    public static native void daxpy( int n, double alpha, double[] x, int incx,
                                    double[] y, int incy);
    
    public static native double ddot(int n, double[] x, int incx, double[] y,
                                     int incy);
    
    /* Level 2: */
    public static native void dgemv(int Layout, int Trans, int m, int n, double
                                    alpha, double[] A, double[] x, int incx,
                                    double beta, double[] y, int incy);
    
    public static native void dtrmv(int Layout, int Uplo, int Trans, int Diag,
                                    int n, double[] A, double[] x, int incx);
    
    public static native void dsymv(int Layout, int Uplo, int n, double alpha,
                                    double[] A, double[] x, int incx, double beta,
                                    double[] y, int incy);
    
    /* Level 3: */
    public static native void dgemm(int Layout, int TransA, int TransB, int m,
                                    int n, int k, double alpha, double[] A,
                                    double[] B, double beta, double[] C);
    
    public static native void dtrmm(int Layout, int Side, int Uplo, int TransA,
                                    int Diag, int m, int n, double alpha,
                                    double[] A, double[] B);
    
    public static native void dsymm(int Layout, int Side, int Uplo, int m, int n,
                                    double alpha, double[] A, double[] B,
                                    double beta, double[] C);
       
    
}







