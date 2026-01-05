ngpu = $(shell nvidia-smi -L | wc -l)
intranode_run = torchrun --standalone --nproc-per-node $(ngpu)

hello-world-all-reduce:
	$(intranode_run) example.py

bench-ar:
	$(intranode_run) benchmark/benchmark_all_reduce.py

bench-mm-rs:
	$(intranode_run) benchmark/benchmark_matmul_reduce_scatter.py

bench-ag-mm:
	$(intranode_run) benchmark/benchmark_all_gather_matmul.py

bench-ar-bias:
	$(intranode_run) benchmark/benchmark_all_reduce_bias.py

bench-ar-bias-rms-norm:
	$(intranode_run) benchmark/benchmark_all_reduce_bias_rms_norm.py

bench-all:
	$(MAKE) bench-ar
	$(MAKE) bench-mm-rs
	$(MAKE) bench-ag-mm
	$(MAKE) bench-ar-bias
	$(MAKE) bench-ar-bias-rms-norm