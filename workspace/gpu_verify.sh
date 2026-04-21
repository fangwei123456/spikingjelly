#!/usr/bin/env bash
# GPU verification script for feature/flexsn-inductor-backend
#
# Run on an NVIDIA GPU host with torch (CUDA build) + triton + pytest installed.
# Verifies correctness + captures the Triton kernel Inductor emits; optional
# benchmark at the end decides whether M3.b (single-node lowering) is needed.
#
# Usage:
#   bash gpu_verify.sh                 # all sections
#   bash gpu_verify.sh --skip-bench    # skip the benchmark
#   bash gpu_verify.sh --only bench    # only the benchmark

set -euo pipefail

# ---- config -----------------------------------------------------------------
export PYTORCH_JIT=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/.gpu_verify}"
# Pick a GPU with enough free memory; override with CUDA_VISIBLE_DEVICES=N
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
mkdir -p "$LOG_DIR"

SKIP_BENCH=0
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-bench) SKIP_BENCH=1 ;;
        --only) ONLY="$2"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

run_step() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

hr() { printf '\n\033[1;34m==== %s ====\033[0m\n' "$1"; }
ok() { printf '\033[1;32m[OK]\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$1"; exit 1; }

# ---- 1. env -----------------------------------------------------------------
if run_step env; then
hr "1. environment"
$PYTHON - <<'PY' || fail "env check"
import torch, sys
assert torch.cuda.is_available(), "CUDA is not available"
print("torch:    ", torch.__version__)
print("cuda:     ", torch.version.cuda)
print("device:   ", torch.cuda.get_device_name(0))
try:
    import triton
    print("triton:   ", triton.__version__)
except ImportError:
    print("triton:    NOT INSTALLED — Inductor GPU path will fail", file=sys.stderr)
    sys.exit(1)
PY
ok "environment"
fi

# ---- 2. unit tests ----------------------------------------------------------
if run_step tests; then
hr "2. unit tests (M1+M2+M3.a, CPU-only correctness)"
$PYTHON -m pytest test/activation_based/test_flex_sn_inductor.py -v \
    --tb=short 2>&1 | tee "$LOG_DIR/tests.log"
grep -qE '[0-9]+ passed' "$LOG_DIR/tests.log" || fail "pytest did not report any passes"
! grep -qE 'failed|error' "$LOG_DIR/tests.log" || fail "pytest reported failures"
ok "unit tests"
fi

# ---- 3. verify Inductor emits Triton on GPU ---------------------------------
if run_step triton-emit; then
hr "3. Inductor emits Triton kernel on GPU"
TORCH_LOGS=output_code $PYTHON - 2>&1 <<'PY' | tee "$LOG_DIR/output_code.log" > /dev/null
import os
os.environ["PYTORCH_JIT"] = "0"
import torch
from spikingjelly.activation_based.neuron.flexsn import FlexSN

def lif_core(x, v):
    tau, v_th = 2.0, 1.0
    h = v + (x - v) / tau
    s = (h >= v_th).to(h.dtype)
    return s, h * (1.0 - s)

T, B, N = 8, 32, 1024
neuron = FlexSN(core=lif_core, num_inputs=1, num_states=1, num_outputs=1,
                step_mode="m", backend="inductor").cuda()
cmodel = torch.compile(neuron, fullgraph=True)
x = torch.randn(T, B, N, device="cuda")
out = cmodel(x); torch.cuda.synchronize()
print("compiled output shape:", tuple(out.shape), "device:", out.device)
PY
grep -q '@triton.jit' "$LOG_DIR/output_code.log" \
    || fail "no @triton.jit kernel in Inductor output — check model/input are on CUDA"
TRITON_KERNELS=$(grep -c '@triton.jit' "$LOG_DIR/output_code.log")
echo "Triton kernels emitted: $TRITON_KERNELS"
echo "(full output: $LOG_DIR/output_code.log)"
ok "Triton emission"
fi

# ---- 4. numerical parity: inductor vs triton baseline -----------------------
if run_step parity; then
hr "4. numerical parity — backend=\"inductor\" vs \"triton\""
$PYTHON - <<'PY' || fail "numerical parity"
import os
os.environ["PYTORCH_JIT"] = "0"
import torch
from spikingjelly.activation_based.neuron.flexsn import FlexSN

def lif_core(x, v):
    tau, v_th = 2.0, 1.0
    h = v + (x - v) / tau
    s = (h >= v_th).to(h.dtype)
    return s, h * (1.0 - s)

torch.manual_seed(0)
T, B, N = 8, 32, 1024
x = torch.randn(T, B, N, device="cuda")

def mk(backend):
    return FlexSN(core=lif_core, num_inputs=1, num_states=1, num_outputs=1,
                  step_mode="m", backend=backend).cuda()

n_tri = mk("triton")
n_ind = mk("inductor")
c_ind = torch.compile(n_ind, fullgraph=True)

out_tri = n_tri(x)
out_ind = c_ind(x)
torch.testing.assert_close(out_ind, out_tri, atol=1e-5, rtol=1e-5)
print("forward parity: OK  (max abs diff =",
      (out_ind - out_tri).abs().max().item(), ")")
PY
ok "numerical parity"
fi

# ---- 5. micro-benchmark (design doc G2) -------------------------------------
if run_step bench && [[ "$SKIP_BENCH" -eq 0 ]]; then
hr "5. micro-benchmark — inductor vs triton baseline"
$PYTHON - <<'PY' 2>&1 | tee "$LOG_DIR/bench.log"
import os
os.environ["PYTORCH_JIT"] = "0"
import torch
from spikingjelly.activation_based.neuron.flexsn import FlexSN

def lif_core(x, v):
    tau, v_th = 2.0, 1.0
    h = v + (x - v) / tau
    s = (h >= v_th).to(h.dtype)
    return s, h * (1.0 - s)

def mk(backend):
    return FlexSN(core=lif_core, num_inputs=1, num_states=1, num_outputs=1,
                  step_mode="m", backend=backend).cuda()

def cuda_time_ms(fn, iters=200):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record(); [fn() for _ in range(iters)]; end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

configs = [(8, 128, 1024), (32, 128, 1024), (8, 128, 4096)]
print(f"{'T':>4} {'B':>5} {'N':>6} {'triton (ms)':>14} {'inductor (ms)':>15} {'ratio':>8}")
print("-" * 60)
for T, B, N in configs:
    n_tri = mk("triton"); n_ind = mk("inductor")
    c_ind = torch.compile(n_ind, fullgraph=True)
    x = torch.randn(T, B, N, device="cuda")
    # warm up (inductor first-run compilation cost must not count)
    for _ in range(3):
        c_ind(x); n_tri(x)
    torch.cuda.synchronize()
    ms_tri = cuda_time_ms(lambda: n_tri(x))
    ms_ind = cuda_time_ms(lambda: c_ind(x))
    ratio = ms_ind / ms_tri
    flag = "OK" if ratio <= 1.1 else ("CLOSE" if ratio <= 1.5 else "CONSIDER M3.b")
    print(f"{T:>4} {B:>5} {N:>6} {ms_tri:>14.3f} {ms_ind:>15.3f} {ratio:>7.2f}x  [{flag}]")

print("\nG2 criterion: ratio <= 1.1 => inductor is acceptable; skip M3.b.")
print("               ratio >= 1.5 => single-node lowering (M3.b) worth investigating.")
PY
echo "(full benchmark: $LOG_DIR/bench.log)"
ok "benchmark"
fi

hr "done"
echo "logs: $LOG_DIR/"
