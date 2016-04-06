NAME=gameOfLife
NVCC=nvcc
OUTFILE=output
GCC=g++
NAIVE=$(NAME)-naive
MEM=$(NAME)-mem
CPU=$(NAME)-cpu
LIBS=mylib

all: directory cpu gpu-naive gpu-mem
	echo All dependencies successfully met

directory:
	mkdir -p ./bin

cpu: ./src/cpuLife/$(NAME).c $(LIBS)
	$(GCC) -o ./bin/$(CPU) ./src/cpuLife/$(NAME).c ./bin/mylib.o

gpu-naive: ./src/naiveLife/$(NAIVE).cu $(LIBS)
	$(NVCC) --gpu-architecture=sm_50 -rdc=true -lcudadevrt -I../NVIDIA_CUDA-7.5_Samples/common/inc/ -o ./bin/$(NAIVE) ./src/naiveLife/$(NAIVE).cu ./bin/mylib.o

gpu-mem: ./src/bitLife/bitLife.cu $(LIBS)
	$(NVCC) --gpu-architecture=sm_50 -rdc=true -lcudadevrt -I../NVIDIA_CUDA-7.5_Samples/common/inc/ -o ./bin/$(MEM) ./src/bitLife/bitLife.cu ./bin/mylib.o

mylib: ./src/mylib.c ./src/mylib.h
	$(GCC) -c ./src/mylib.c -o ./bin/mylib.o

generator: ./src/boardGenerator/boardGenerator.cu
	$(NVCC) -I../NVIDIA_CUDA-7.5_Samples/common/inc/ -lcurand -o ./bin/boardGenerator ./src/boardGenerator/boardGenerator.cu

generate-board: generator
	mkdir -p ./boards
	head -c 8 /dev/urandom | ./bin/boardGenerator -s $(BOARD_SIZE) > ./boards/$(BOARD_SIZE).board

test: all
	echo size,iterations,threads,time > $(OUTFILE) ; \
	./bin/$(CPU) -f ./boards/1024.board -s 1024 >> $(OUTFILE) ; \
	./bin/$(CPU) -f ./boards/2048.board -s 2048 >> $(OUTFILE) ; \
	./bin/$(CPU) -f ./boards/4096.board -s 4096 >> $(OUTFILE) ; \
	for threads in 32 64 128 256 512 ; do \
		./bin/$(NAIVE) -f ./boards/1024.board -s 1024 -t $$threads >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/1024.board -s 1024 -t $$threads -b 8 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/1024.board -s 1024 -t $$threads -b 16 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/1024.board -s 1024 -t $$threads -b 32 >> $(OUTFILE) ; \
		./bin/$(NAIVE) -f ./boards/2048.board -s 2048 -t $$threads >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/2048.board -s 2048 -t $$threads -b 8 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/2048.board -s 2048 -t $$threads -b 16 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/2048.board -s 2048 -t $$threads -b 32 >> $(OUTFILE) ; \
		./bin/$(NAIVE) -f ./boards/4096.board -s 4096 -t $$threads >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/4096.board -s 4096 -t $$threads -b 8 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/4096.board -s 4096 -t $$threads -b 16 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/4096.board -s 4096 -t $$threads -b 32 >> $(OUTFILE) ; \
		./bin/$(NAIVE) -f ./boards/8192.board -s 8192 -t $$threads >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/8192.board -s 8192 -t $$threads -b 8 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/8192.board -s 8192 -t $$threads -b 16 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/8192.board -s 8192 -t $$threads -b 32 >> $(OUTFILE) ; \
		./bin/$(NAIVE) -f ./boards/16384.board -s 16384 -t $$threads >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/16384.board -s 16384 -t $$threads -b 8 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/16384.board -s 16384 -t $$threads -b 16 >> $(OUTFILE) ; \
		./bin/$(MEM) -f ./boards/16384.board -s 16384 -t $$threads -b 32 >> $(OUTFILE) ; \
	done
