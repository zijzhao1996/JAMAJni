JAVA_HOME = ..

LAPACK_CLASS = lapacktest.class
LAPACK_RUN = lapack_test_result.txt
BLAS_CLASS = blastest.class
BLAS_RUN = blas_result_result.txt

all: $(LAPACK_CLASS) $(LAPACK_RUN) $(BLAS_CLASS) $(BLAS_RUN)

$(LAPACK_CLASS):
	javac -classpath $(JAVA_HOME)/class:. lapacktest.java 

$(LAPACK_RUN):
	java -Djava.library.path=$(JAVA_HOME)/lib -classpath $(JAVA_HOME)/class:. lapacktest > lapack_test_result.txt

$(BLAS_CLASS):
	javac -classpath $(JAVA_HOME)/class:. blastest.java 

$(BLAS_RUN):
	java -Djava.library.path=$(JAVA_HOME)/lib -classpath $(JAVA_HOME)/class:. blastest > blas_test_result.txt

clean: 
	rm -f *.txt *.class
