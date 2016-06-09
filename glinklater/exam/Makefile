all: orig no-l1 opt
	echo Done

orig: junetranspose.cu
	nvcc junetranspose.cu -o junetranspose

no-l1: junetranspose.cu
	nvcc -Xptxas -dlcm=ca -o junetranspose-l2 junetranspose.cu

opt: junetranspose-opt.cu
	nvcc junetranspose-opt.cu -o junetranspose-opt

clean:
	rm junetranspose
	rm junetranspose-l2
	rm junetranspose-opt
