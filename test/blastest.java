import JAMAJni.jni_blas.*;

public final class blastest {
    private blastest() {}
    public static void main(String[] args) {
        //
        // Prepare the matrices and other parameters
        //
        System.out.println("###   Parameter Preparation   ###");
        //
        // Parameter for test
        //
        double alpha=2, beta=-1;
        int L = 3;
        int M = 3, N = 3, K = 3;
        int M2 = 2, N2 = 4, K2 = 3;
        int incx = 1, incy = 1;
        //
        // vector for test
        //
        double[] x = new double[] {1, 2, 1};
        double[] cpx = new double[] {1, 2, 1};
        double[] y = new double[] {1, 2, 3};
        double[] cpy = new double[] {1, 2, 3};
        double[] y2 = new double[] {1, 2};
        double[] cpy2 = new double[] {1, 2};
        //
        // Square matrix in col-major and row-major for test
        //
        double[] As = new double[] {12, -51, 4, 6, 167, -68, -4, 24, -41};
        double[] Bs = new double[] {4, -2, -6, -2, 10, 9, -6, 9, 14};
        double[] Cs = new double[] {2, -4, 3, 7, -3, -7, 0, 18, -20};
        double[] Asc = new double[] {12, 6, -4, -51, 167, 24, 4, -68, -41};
        double[] Csc = new double[] {2, 7, 0, -4, -3, 18, 3, -7, -20};
        //
        double[] cpAs = new double[] {12, -51, 4, 6, 167, -68, -4, 24, -41};
        double[] cpBs = new double[] {4, -2, -6, -2, 10, 9, -6, 9, 14};
        double[] cpCs = new double[] {2, -4, 3, 7, -3, -7, 0, 18, -20};
        double[] cpAsc = new double[] {12, 6, -4, -51, 167, 24, 4, -68, -41};
        double[] cpCsc = new double[] {2, 7, 0, -4, -3, 18, 3, -7, -20};
        //
        // rectangular matrix in col-major and row-major for test
        //
        double[] Ac = new double[] {1, 4, 2, 5, 3, 6};
        double[] Bc = new double[] {7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18};
        double[] Cc = new double[] {2, -3, -4, -7, 3, 0, 7, 18};
        double[] Ec = new double[] {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
        double[] Gc = new double[] {4, 6, 2, 5, 3, 1};
        double[] Ar = new double[] {1, 2, 3, 4, 5, 6};
        double[] Br = new double[] {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        double[] Cr = new double[] {2, -4, 3, 7, -3, -7, 0, 18};
        double[] Er = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        double[] Fr = new double[] {1, 2, 3, 4, 5, 6, 7, 8};
        double[] Gr = new double[] {4, 2, 3, 6, 5, 1};
        double[] Br2 = new double[] {7, 8, 9, 10, 11, 12, 13, 14};
        double[] Bc2 = new double[] {7, 11, 8, 12, 9, 13, 10, 14};
        double[] C8 = new double[8];
        double[] C12 = new double[12];
        double[] cpAc = new double[] {1, 4, 2, 5, 3, 6};
        double[] cpBc = new double[] {7, 11, 15, 8, 12, 16, 9, 13, 17,
            10, 14, 18};
        double[] cpCc = new double[] {2, -3, -4, -7, 3, 0, 7, 18};
        double[] cpEc = new double[] {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
        double[] cpGc = new double[] {4, 6, 2, 5, 3, 1};
        double[] cpAr = new double[] {1, 2, 3, 4, 5, 6};
        double[] cpBr = new double[] {7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18};
        double[] cpCr = new double[] {2, -4, 3, 7, -3, -7, 0, 18};
        double[] cpEr = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        double[] cpFr = new double[] {1, 2, 3, 4, 5, 6, 7, 8};
        double[] cpGr = new double[] {4, 2, 3, 6, 5, 1};
        double[] cpBr2 = new double[] {7, 8, 9, 10, 11, 12, 13, 14};
        double[] cpBc2 = new double[] {7, 11, 8, 12, 9, 13, 10, 14};
        double[] cpC8 = new double[8];
        double[] cpC12 = new double[12];
        //
        // Working space set for saving results
        // vector
        double[] z = new double[3];
        double[] z2 = new double[2];
        // matrix
        double[] D6 = new double[6];
        double[] D9 = new double[9];
        double[] D8 = new double[8];
        double[] D12 = new double[12];
        double[] cpz = new double[3];
        double[] cpz2 = new double[2];
        double[] cpD6 = new double[6];
        double[] cpD9 = new double[9];
        double[] cpD8 = new double[8];
        double[] cpD12 = new double[12];
        //
        // Print the parameters
        //
        printMatrix("Array x", Matrix.LAYOUT.RowMajor, x, L, 1);
        printMatrix("Array y", Matrix.LAYOUT.RowMajor, y, L, 1);
        printMatrix("Array y2", Matrix.LAYOUT.RowMajor, y2, M2, 1);
        System.out.println();
        System.out.println("alpha=" + string(alpha));
        System.out.println("beta=" + string(beta));
        System.out.println();
        System.out.println("Square matrix:");
        printMatrix("Matrix As:", Matrix.LAYOUT.RowMajor, As, M, K);
        printMatrix("Matrix Bs:", Matrix.LAYOUT.RowMajor, Bs, K, N);
        printMatrix("Matrix Cs:", Matrix.LAYOUT.RowMajor, Cs, M, N);
        System.out.println();
        System.out.println("Rectangle Matrix:");
        printMatrix("Matrix A (2 by 3):", Matrix.LAYOUT.RowMajor, Ar, M2, K2);
        printMatrix("Matrix B (3 by 4):", Matrix.LAYOUT.RowMajor, Br, K2, N2);
        printMatrix("Matrix C (2 by 4):", Matrix.LAYOUT.RowMajor, Cr, M2, N2);
        printMatrix("Matrix E (3 by 4):", Matrix.LAYOUT.RowMajor, Er, K2, N2);
        printMatrix("Matrix F (4 by 2):", Matrix.LAYOUT.RowMajor, Fr, N2, M2);
        printMatrix("Matrix G (2 by 3):", Matrix.LAYOUT.RowMajor, Gr, M2, K2);
        //
        // Set Option
        //
        int Layout = Matrix.LAYOUT.RowMajor;
        int Layout2 = Matrix.LAYOUT.ColMajor;
        int TransA = Matrix.TRANSPOSE.NoTrans;
        int TransB = Matrix.TRANSPOSE.NoTrans;
        int Uplo = Matrix.UPLO.Upper;
        int Diag = Matrix.DIAG.NonUnit;
        int Side = Matrix.SIDE.Left;
        
        
        
        //
        /* Level 1 */
        //
        System.out.println();
        System.out.println("###   Level 1   ###");
        System.out.println();
        //
        // dscal
        //
        System.out.println("dscal: x = alpha * x");
        int n = L;
        x = cpx.clone();
        Matrix.dscal(n, alpha, x, incx);
        printMatrix("Resulting x", Matrix.LAYOUT.RowMajor, x, n, 1);
        //
        // daxpy
        //
        System.out.println();
        System.out.println("daxpy: y = alpha * x + y");
        x = cpx.clone();
        y = cpy.clone();
        Matrix.daxpy(n, alpha, x, incx, y, incy);
        printMatrix("Resulting y", Matrix.LAYOUT.RowMajor, y, n, 1);
        //
        // ddot
        //
        System.out.println();
        y = cpy.clone();
        System.out.println("ddot: x dot y = "+
                           string(Matrix.ddot(n, x, incx, y, incy)));
        //
        /* Level 2*/
        //
        System.out.println();
        System.out.println("###   Level 2   ###");
        //
        // dgemv
        //
        System.out.println();
        System.out.println("dgemv: y2 := alpha*A*x + beta*y2 ");
        z2 = cpy2.clone();
        System.out.println("Row-major: ");
        int layout = Layout;
        int trans = TransA;
        int m = M2;
        n = K2;
        double[] a = Ar;
        y = z2;
        Matrix.dgemv(layout, trans, m, n, alpha, a, x, incx, beta, y, incy);
        printMatrix("Resulting y", layout, y, m, 1);
        System.out.println("Col-major: ");
        z2 = cpy2.clone();
        layout = Layout2;
        a = Ac;
        y = z2;
        Matrix.dgemv(layout, trans, m, n, alpha, a, x, incx, beta, y, incy);
        printMatrix("Resulting y", layout, y, m, 1);
        //
        /* Level 3 */
        //
         System.out.println();
	 System.out.println("###   Level 3   ###");
	double[][] aaa  =  {{1, 2, 3},{4,5,6}};
        Matrix testA = new Matrix(aaa);
	
	printMatrix2("Matrix testA:", testA);
        printMatrix2("Matrix testB:", Matrix.times(testA, alpha));
   
         /** Linear algebraic matrix multiplication, A * B */   
	double[][] am = {{1,2},{3,4}};
	double[][] bm = {{4,3},{2,1}};
	Matrix Am  = new Matrix(am);
        Matrix Bm  = new Matrix(bm);
	printMatrix2("Matrix testA:", Am);
	printMatrix2("Matrix testB:", Bm);
	double[][] cm = {{8,5},{20,13}};     
        Matrix Cm = new Matrix(cm);
        printMatrix2("Multiplication of Matrix A and B:", Cm);
     }    


    
    
    private static void printMatrix2(String prompt, Matrix A) {
        System.out.println(prompt);
        for (int i=0; i<A.getRowDimension(); i++) {
            for (int j=0; j<A.getColumnDimension(); j++)
                System.out.print("\t"+string(A.get(i,j)));
            System.out.println();
        }
    }
    
    
    
    
    /** Print the matrix X. */
    private static void printMatrix(String prompt, int layout,
                                    double[] X, int I, int J) {
        System.out.println(prompt);
        if (layout == Matrix.LAYOUT.ColMajor) {
            for (int i=0; i<I; i++) {
                for (int j=0; j<J; j++)
                    System.out.print("\t" + string(X[j*I+i]));
                System.out.println();
            }
        }
        else if (layout == Matrix.LAYOUT.RowMajor){
            for (int i=0; i<I; i++) {
                for (int j=0; j<J; j++)
                    System.out.print("\t" + string(X[i*J+j]));
                System.out.println();
            }
        }
        else{System.out.println("** Illegal layout setting");}
    }
    
    /** Shorter string for real number. */
    private static String string(double re) {
        String s="";
        if (re == (long)re)
            s += (long)re;
        else
            s += re;
        return s;
    }
}







