NAME=newton
NVCC=nvcc
OUTFILE=output.csv

all: $(NAME) makefile

$(NAME): $(NAME).cu
	@echo 'Building target: $@'
	nvcc -I../../NVIDIA_CUDA-7.5_Samples/common/inc/ -lcurand -o $(NAME) $(NAME).cu
	@echo 'Finished building target: $@'
	@echo ' '

test: $(NAME)
	rm -f $(OUTFILE)
	echo "curand_op,curand_exec,num_random,samples_per_second,blockx,blocky,gridx,gridy,sqrt_exec,iterations" >> $(OUTFILE)
	n=1000000 ; i=0 ; while [ $$i -lt 3 ] ; do \
		for alg in 1 2 3 ; do \
			./$(NAME) -x 16 -y 16 -$$alg -n $$n >> $(OUTFILE) ; \
			./$(NAME) -x 32 -y 16 -$$alg -n $$n >> $(OUTFILE) ; \
			./$(NAME) -x 16 -y 32 -$$alg -n $$n >> $(OUTFILE) ; \
			a=32 ; b=32 ; while [ $$b -gt 1 ] ; do \
				./$(NAME) -x $$a -y $$b -$$alg -n $$n >> $(OUTFILE) ; \
				./$(NAME) -x $$b -y $$a -$$alg -n $$n >> $(OUTFILE) ; \
				a=$$((a * 2)) ; \
				b=$$((b = b / 2)) ; \
				echo $$n $$i $$a $$b $$alg ; \
			done ; \
		done ; \
		n=$$((n * 10)) ; \
		i=$$((i + 1)) ; \
	done

	