import JAMAJni.jni_lapack.*;
import JAMAJni.jni_blas.*;

public final class lapacktest {
    private lapacktest() {}
    public static void main(String[] args) {
        //
        // Prepare the matrices and other parameters
        //
        System.out.println("###   Testfile for JAMAJni jni_lapack   ### ");
        int M=3, N=3, K=3;
        int[] info = new int[]{0};
        double[] A = new double[]  {12, -51, 4, 6, 167, -68, -4, 24, -41};
        double[] cpA = new double[] {12, -51, 4, 6, 167, -68, -4, 24, -41};
        double[] AT = new double[] {12, 6, -4, -51, 167, 24, 4, -68, -41};
        double[] B = new double[] {4, -2, -6, -2, 10, 9, -6, 9, 14};
        double[] D = new double[9];
        double[] D2 = new double[9];
        int[] ipiv = new int[] {0, 0, 0};
        double[] tau = new double[]{0, 0, 0};
        int[] jpvt = new int[] {1, 2, 3};
        double[] sym = new double[]{1,4,0,4,2,5,0,5,3};
        double[] cpsym = new double[]{1,4,0,4,2,5,0,5,3};
        double[] eq = new double[]{1,2,3};
        double[] I = new double[]{1,0,0,0,1,0,0,0,1};
        
        double[] VL = new double[M * N];
        double[] VR = new double[M * N];
        double[] WR = new double[N];
        double[] WI = new double[N];
        double[] S = new double[N];
        double[] U = new double[M * M];
        double[] VT = new double[N * N];
        double[] superb = new double[N - 1];

        int lwork = 4*N+1;
	double[] work = new double[lwork];
        int[] iwork = new int[8*M];
        //
        // Set Option
        //
        int matrix_layout = LUDecomposition.LAYOUT.RowMajor;
        char Trans = LUDecomposition.TRANSPOSE.NoTrans;
        char uplo = LUDecomposition.UPLO.Upper;
        char Jobvl = LUDecomposition.JOBV.Compute;
        char Jobvr = LUDecomposition.JOBV.Compute;
        char Jobu = LUDecomposition.JOB.All;
        char Jobvt = LUDecomposition.JOB.All;
        char Jobz = LUDecomposition.JOB.Overwritten;
        int itype = LUDecomposition.ITYPE.first;
        int result;
        //
        /* ---- LU ---- */
        //
        System.out.println("\n \n");
        System.out.println("###   LU   ###");
        //
        // dgetrf
        //
        System.out.println();
        System.out.println("The LU decomposition A=P*L*U (L has unit diagnoal elements):");
        D = A.clone();
	
        int m = M;          // m is the number of rows of matrix A
        int n = N;          // n is the number of columns of matrix A
        double[] a = D;     // a is a general m-by-n matrix
        int lda = N;        // lda is the leading dimension of the array a,
                            // here since it is row-major, lda >= max(1, n)
                            // ipiv is the pivot indices
                            //  info is integer (array)
        printMatrix("Matrix A = ", matrix_layout, a, m, n);
        result = LUDecomposition.dgetrf( matrix_layout, m, n, a, lda, ipiv, info);

        System.out.println("dgetrf");
        printMatrix("The LU matrix of A = ", matrix_layout, a, m, n);
        printIntArray("  The permutation vector ipiv = ", ipiv, m);
    
	System.out.println("dgetrf jama");
	double[][] AA = {{12, -51, 4},{6, 167, -68}, {-4, 24, -41}};
	Matrix AAA = new Matrix(AA);    
  	LUDecomposition AAAA = new LUDecomposition(AAA);
        Matrix AAAAA= AAAA.getL();
        printMatrix2("Get L = ", AAAAA);
	Matrix AAAAA2 = AAAA.getU();
	printMatrix2("Get U = ", AAAAA2);
	
       //
       /* ---- Cholesky ---- */                
       //
       System.out.println("\n \n");
       System.out.println("###   Cholesky   ###");
       //
       //dpotrf 
       //
       System.out.println();
       D = B.clone();
       n = M;
       a = D;
       lda = M;
       printMatrix("Matrix A = ", matrix_layout, a, n, n);
       CholeskyDecomposition.dpotrf(matrix_layout, uplo, n, a, lda, info);
       System.out.println("dpotrf");
       printMatrix("  The Cholesky factorization of matrix A = ", matrix_layout, a, n, n);
   
       System.out.println("dpotrf jama");
       double[][] BB = {{4, -2, -6},{-2, 10, 9},{-6, 9, 14}};
       Matrix BBB = new Matrix(BB);
       CholeskyDecomposition BBBB = new CholeskyDecomposition(BBB);
       Matrix BBBBB = BBBB.getL();
       printMatrix2("Get L = ", BBBBB);

       //
       /* ---- QR ---- */
       //
       System.out.println("\n \n");
       System.out.println("###   QR decomposition   ###");
       //
       //dgeqrf
       //
       System.out.println();
       System.out.println("A=QR:");
       D = A.clone();
       m = M;              // m = the number of rows of the matrix a
       n = N;              // n = the number of columns of matrix a
       a = D;              // a dimension (m, n)
       lda = n;            // lda = leading dimension of a
                           //tau = the scalar factors of the elementary
	        	   //reflectors, dimension (min(m,n))
			   //work dimension (max(1,lwork))
		           //lwork is integer
			   //info is integer(array)
       System.out.println("dgeqrf");
       printMatrix("Matrix A = ", matrix_layout, a, m, n);
       QRDecomposition.dgeqrf(matrix_layout, m, n, a, lda, tau, work, lwork, info);    printMatrix("  The rewritten matrix A = ", matrix_layout, a, m, n);
       //
       // dorgqr
       //
       System.out.println();
       m = M;              // m = the number of rows of matrix q
       n = N;              // n = the number of columns of matrix q
       int k = M;          // k = the number of elementary reflectors whose product defines the matrix q
                           // a is returned by dgeqrf
			   // lda = leading dimension of a
			   // tau is returned by dgeqrf
			   // work dimension (max(1,lwork))
			   // lwork = dimension of array work
			   // info integer(array)
       System.out.println("dorgqr");
       QRDecomposition.dorgqr(matrix_layout, m, n, k, a, lda, tau, work, lwork, info);
       printMatrix(" The matrix Q is :", matrix_layout, a, m, n);

       System.out.println("dorgqr jama");
       printMatrix2("Matrix A =", AAA);
       QRDecomposition CCCC = new QRDecomposition(AAA);
       Matrix CCCCC = CCCC.getQ();
       printMatrix2("Get Q =", CCCCC);
    
      //
      // dgeqp3
      //
      System.out.println();
      D = A.clone();
      lwork = 30;
      m = M;              // m = the number of rows of matrix a
      n = N;              // n = the number of columns of matrix a
      a = D;              // a dimension(m,n)
      lda = n;            // lda = leading dimension of a
                          //jpvt integer array, dimension (n)    
			  //tau double precision array, dimension (min(m,n))
			  //work double precision array, dimension (max(1,lwork))
			  //lwork integer, the dimension of array work
			  //info integer(array)
      System.out.println("dgeqp3");
      QRDecomposition.dgeqp3(matrix_layout, m, n, a, lda, jpvt, tau, work, lwork, info);
      printMatrix("  The R matrix computed by dgeqp3:", matrix_layout, a, m, n);

      System.out.println("dgeqp3 jama");
      QRDecomposition CCCC2 = new QRDecomposition(AAA);
      Matrix CCCCC2 = CCCC2.getR();
      printMatrix2("Get R =", CCCCC2);


      //
      /* ---- EigenvalueDecomposition---- */
      //
      //
      //dsyev
      //
      System.out.println();
      System.out.println("### EigenvalueDecomposition ###");
      System.out.println(" \ndsyev: computes the eigenvalues and right/left eigenvectors for symmetric matrix A");
      char jobvl = Jobvl;
      n = N;           // the order of matrix a
      a = sym;         // a is symmetric matrix
      lda = n;         // lda = leading dimension of array a
      double[] w = S;  // w = eigenvalues in ascending order, dimension (n)
                       //work double precision array, dimension(max(1,lwork))
		       //lwork = length of array work
		       //info integer(array)
      lwork = 4*n+1;
      work = new double[lwork];
      System.out.println("dsyev");
      printMatrix("matrix A = ", matrix_layout, a, n, n);
      EigenvalueDecomposition.dsyev(matrix_layout, jobvl, uplo, n, a, lda, w, work, lwork, info);
      printMatrix("  Eigenvectors of A :", matrix_layout, a, n, n);
      printMatrix("\n  The singular values of A :", matrix_layout, w, 1, n);
  
      System.out.println("dsyev jama"); 
      sym = new double[] {1,4,0,4,2,5,0,5,3}; 
      Matrix DDD = new Matrix(sym,3);
      printMatrix2("Matrix A = ",DDD);
      EigenvalueDecomposition DDDD = new EigenvalueDecomposition(DDD);
      Matrix DDDDD = DDDD.getV();
      printMatrix2("Get V =", DDDDD);
      double[] DDDDD2 = DDDD.getD();
      printMatrix("Get D =",matrix_layout,w,1,n); 
    
      //
      /* ----SingularValueDecomposition---- */
      //
      //
      //dgesvd
      //
      System.out.println();
      System.out.println("### SingularValueDecomposition ###");
      D = A.clone();
      lwork = 50;
      info = new int[]{0};
      char jobu = Jobu;
      char jobvt = Jobvt;
      m = M;
      n = N;
      a = D;
      lda = M;
      double[] s = S;
      double[] u = U;
      int ldu = N; 
      double[] vt = VT;
      int ldvt = N; 
      System.out.println("dgesvd");
      printMatrix("Matrix A = ", matrix_layout, a, m, n);
      SingularValueDecomposition.dgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
      printMatrix("\n  D = ", matrix_layout, s, 1, m);
      printMatrix("\n  U = ", matrix_layout, u, m, n);
      printMatrix("\n  VT = ", matrix_layout, vt, n, n);
 
      System.out.println("dgesvd jama");
      SingularValueDecomposition EEEE = new SingularValueDecomposition(AAA);
      printMatrix("\n Get D =", matrix_layout, s, 1, m);
      printMatrix("\n Get U =", matrix_layout, u, m, n);
      printMatrix("\n Get V =", matrix_layout, vt, n, n);
    }
   
     //print function//
     /** Print the matrix X. */
      private static void printMatrix2(String prompt, Matrix A) {
	    System.out.println(prompt);
	    for (int i=0; i<A.getRowDimension(); i++) {
		    for (int j=0; j<A.getColumnDimension(); j++)
			    System.out.print("\t"+string(A.get(i,j)));
		    System.out.println();
	    }
    }

   
    /** Print the matrix X. */
     private static void printMatrix(String prompt, int layout, double[] X, int I, int J) {
        System.out.println(prompt);
        if (layout == LUDecomposition.LAYOUT.ColMajor) {
            for (int i=0; i<I; i++) {
                for (int j=0; j<J; j++)
                    System.out.print("\t" + string(X[j*I+i]));
                System.out.println();
            }
        }	
        else if (layout == LUDecomposition.LAYOUT.RowMajor){
            for (int i=0; i<I; i++) {
                for (int j=0; j<J; j++)
                    System.out.print("\t" + string(X[i*J+j]));
                System.out.println();
            }
        }
        else{System.out.println("** Illegal layout setting");}
    }
    
    private static void printIntArray(String prompt, int[] X, int L) {
        System.out.println(prompt);
        for (int i=0; i<L; i++) {
                System.out.print("\t" + string(X[i]));
        }
        System.out.println();
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



