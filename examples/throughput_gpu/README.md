# Throughput GPU Example

Install the standard MDCoreAlign package, add a CuPy build compatible with the
host CUDA runtime, run `mdcorealign doctor`, then run `bash
examples/throughput_gpu/run.sh`. The toy run is a functional smoke test, not a
speed benchmark. Explicit GPU requests fail rather than silently using CPU.
GPU and CPU throughput are backends of the same scientific workflow.
