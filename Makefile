ngpu = $(shell nvidia-smi -L | wc -l)

hello-world-all-reduce:
	torchrun --standalone --nproc-per-node $(ngpu) example.py