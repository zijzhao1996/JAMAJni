DEST = .
CC = gcc
CFLAGS = $(INCLUDE) $(LIB) 

JAVA_HOME = ..


JNI_LAPACK_SRC = $(JAVA_HOME)/src/jni_lapack
JNI_BLAS_SRC = $(JAVA_HOME)/src/jni_blas

INCLUDE = -I$(JNI_BLAS_SRC)/c/include \
	-I$(JAVA_HOME)/include \
	-I$(JNI_LAPACK_SRC)/c/include
LIB = -L/usr/lib -L/usr/local/lib -L/usr/local/lib64 
C_SRC_LAPACK1 = $(JNI_LAPACK_SRC)/c/LUDecomposition.c 
C_SRC_LAPACK2 = $(JNI_LAPACK_SRC)/c/CholeskyDecomposition.c
C_SRC_LAPACK3 = $(JNI_LAPACK_SRC)/c/QRDecomposition.c
C_SRC_LAPACK4 = $(JNI_LAPACK_SRC)/c/EigenvalueDecomposition.c
C_SRC_LAPACK5 = $(JNI_LAPACK_SRC)/c/SingularValueDecomposition.c
C_SRC_BLAS = $(JNI_BLAS_SRC)/c/Matrix.c
DYLIB_LAPACK1 = $(JAVA_HOME)/lib/liblapack_lite_LUDecomposition.so
DYLIB_LAPACK2 = $(JAVA_HOME)/lib/liblapack_lite_CholeskyDecomposition.so
DYLIB_LAPACK3 = $(JAVA_HOME)/lib/liblapack_lite_QRDecomposition.so
DYLIB_LAPACK4 = $(JAVA_HOME)/lib/liblapack_lite_EigenvalueDecomposition.so
DYLIB_LAPACK5 = $(JAVA_HOME)/lib/liblapack_lite_SingularValueDecomposition.so
DYLIB_BLAS = $(JAVA_HOME)/lib/libblas_lite.so
	

# For package JAMAJni
JC = javac
JH = javah
JFLAGS = -d
JAVA_SRC = $(JNI_LAPACK_SRC)/java/LUDecomposition.java $(JNI_LAPACK_SRC)/java/CholeskyDecomposition.java $(JNI_LAPACK_SRC)/java/QRDecomposition.java $(JNI_LAPACK_SRC)/java/EigenvalueDecomposition.java $(JNI_LAPACK_SRC)/java/SingularValueDecomposition.java $(JNI_BLAS_SRC)/java/Matrix.java
PACKAGE = $(JAVA_HOME)/class/*
HEADER_LAPACK1 = $(JNI_LAPACK_SRC)/c/include/LUDecomposition.h 
HEADER_LAPACK2= $(JNI_LAPACK_SRC)/c/include/CholeskyDecomposition.h
HEADER_LAPACK3= $(JNI_LAPACK_SRC)/c/include/QRDecomposition.h
HEADER_LAPACK4= $(JNI_LAPACK_SRC)/c/include/EigenvalueDecomposition.h
HEADER_LAPACK5= $(JNI_LAPACK_SRC)/c/include/SingularValueDecomposition.h
HEADER_BLAS = $(JNI_BLAS_SRC)/c/include/Matrix.h


all: $(PACKAGE) $(HEADER_LAPACK1) $(HEADER_LAPACK2) $(HEADER_LAPACK3) $(HEADER_LAPACK4) $(HEADER_LAPACK5) $(HEADER_BLAS) $(DYLIB_LAPACK1) $(DYLIB_LAPACK2) $(DYLIB_LAPACK3) $(DYLIB_LAPACK4) $(DYLIB_LAPACK5) $(DYLIB_BLAS)

$(PACKAGE): $(JAVA_SRC)
	@mkdir -p $(@D)
	$(JC) $(JFLAGS) $(JAVA_HOME)/class $(JAVA_SRC)

$(HEADER_LAPACK1):
	@mkdir -p $(@D)
	$(JH) -o $(HEADER_LAPACK1) -classpath $(JAVA_HOME)/class JAMAJni.jni_lapack.LUDecomposition
$(HEADER_LAPACK2):
	@mkdir -p $(@D)
	$(JH) -o $(HEADER_LAPACK2) -classpath $(JAVA_HOME)/class JAMAJni.jni_lapack.CholeskyDecomposition

$(HEADER_LAPACK3):
	 @mkdir -p $(@D)
	 $(JH) -o $(HEADER_LAPACK3) -classpath $(JAVA_HOME)/class JAMAJni.jni_lapack.QRDecomposition

$(HEADER_LAPACK4):
	@mkdir -p $(@D)
	$(JH) -o $(HEADER_LAPACK4) -classpath $(JAVA_HOME)/class JAMAJni.jni_lapack.EigenvalueDecomposition
$(HEADER_LAPACK5):
	@mkdir -p $(@D)
	$(JH) -o $(HEADER_LAPACK5) -classpath $(JAVA_HOME)/class JAMAJni.jni_lapack.SingularValueDecomposition
$(HEADER_BLAS):
	@mkdir -p $(@D)
	$(JH) -o $(HEADER_BLAS) -classpath $(JAVA_HOME)/class JAMAJni.jni_blas.Matrix

$(DYLIB_LAPACK1): $(C_SRC_LAPACK1)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -fPIC $(C_SRC_LAPACK1) -llapack -o $(DYLIB_LAPACK1)

$(DYLIB_LAPACK2): $(C_SRC_LAPACK2)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -fPIC $(C_SRC_LAPACK2) -llapack -o $(DYLIB_LAPACK2)
$(DYLIB_LAPACK3): $(C_SRC_LAPACK3)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -fPIC $(C_SRC_LAPACK3) -llapack -o $(DYLIB_LAPACK3)

$(DYLIB_LAPACK4): $(C_SRC_LAPACK4)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -fPIC $(C_SRC_LAPACK4) -llapack -o $(DYLIB_LAPACK4)

$(DYLIB_LAPACK5): $(C_SRC_LAPACK5)
	@mkdir -p $(@D)
	 $(CC) $(CFLAGS) -shared -fPIC $(C_SRC_LAPACK5) -llapack -o $(DYLIB_LAPACK5)

$(DYLIB_BLAS): $(C_SRC_BLAS) 
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -fPIC $(C_SRC_BLAS) -lblas -o $(DYLIB_BLAS)

clean: 
	rm -fr $(JAVA_HOME)/class
	rm -fr $(JAVA_HOME)/lib
	rm -f $(JNI_LAPACK_SRC)/c/include/*.h
	rm -f $(JNI_BLAS_SRC)/c/include/Matrix.h
	rm -f $(JAVA_HOME)/lib/*.so
	rm -f $(JAVA_HOME)/test/*.class
	rm -f $(JAVA_HOME)/test/*.txt
