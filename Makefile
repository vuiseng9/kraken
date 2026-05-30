ngpu = $(shell nvidia-smi -L | wc -l)
intranode_run = torchrun --standalone --nproc-per-node $(ngpu)
dbg ?= 0

# install-torch 128
# verified 2.11.0+cu128
install:
	pip install -r requirements.txt
	pip install -e .

hello-world-all-reduce:
	DBG_ATTACH=$(dbg) $(intranode_run) example.py

bench-ar:
	DBG_ATTACH=$(dbg) $(intranode_run) benchmark/benchmark_all_reduce.py

bench-mm-rs:
	DBG_ATTACH=$(dbg) $(intranode_run) benchmark/benchmark_matmul_reduce_scatter.py

bench-ag-mm:
	DBG_ATTACH=$(dbg) $(intranode_run) benchmark/benchmark_all_gather_matmul.py

bench-ar-bias:
	DBG_ATTACH=$(dbg) $(intranode_run) benchmark/benchmark_all_reduce_bias.py

bench-ar-bias-rms-norm:
	DBG_ATTACH=$(dbg) $(intranode_run) benchmark/benchmark_all_reduce_bias_rms_norm.py

# https://github.com/meta-pytorch/kraken/pull/32
a2a-ep:
	DBG_ATTACH=$(dbg) $(intranode_run) benchmark/benchmark_moe_a2a.py --save-path moe_a2a_results.csv

dispatch-torchtitan-test:
	DBG_ATTACH=$(dbg) $(intranode_run) moe_symm_mem_kernels/dispatch.py

combine-torchtitan-test:
	DBG_ATTACH=$(dbg) $(intranode_run) moe_symm_mem_kernels/combine.py

bench-all:
	$(MAKE) bench-ar
	$(MAKE) bench-mm-rs
	$(MAKE) bench-ag-mm
	$(MAKE) bench-ar-bias
	$(MAKE) bench-ar-bias-rms-norm
	$(MAKE) bench-a2a-ep
