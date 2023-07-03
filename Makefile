simpleKernel: simpleKernel.cu
	nvcc -o simpleKernel simpleKernel.cu

clean:
	rm simpleKernel
