#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
    echo 'usage: nsys_snn.sh capture OUTPUT_PREFIX -- COMMAND [ARGS...]' >&2
    echo '       nsys_snn.sh capture-graph OUTPUT_PREFIX -- COMMAND [ARGS...]' >&2
    echo '       nsys_snn.sh analyze REPORT.nsys-rep OUTPUT_DIR [BENCHMARK.json]' >&2
    echo '       nsys_snn.sh compare BASELINE/summary.json CANDIDATE/summary.json OUTPUT.json' >&2
    exit 2
}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in
    capture|capture-graph)
        [[ $# -ge 4 && $3 == -- ]] || usage
        trace=cuda,nvtx,cublas,cudnn,osrt,python-gil
        graph_trace=node
        if [[ $1 == capture-graph ]]; then
            graph_trace=node:nvtx-precapture
        fi
        prefix=$2
        shift 3
        command -v nsys >/dev/null || { echo 'nsys is required on the GPU host' >&2; exit 1; }
        [[ ! -e ${prefix}.nsys-rep ]] || { echo "report already exists: ${prefix}.nsys-rep" >&2; exit 1; }
        mkdir -p "$(dirname "$prefix")"
        nsys --version > "${prefix}.nsys-version.txt"
        printf '%q ' "$@" > "${prefix}.command.txt"
        printf '\n' >> "${prefix}.command.txt"
        python - "$prefix" "$trace" "$graph_trace" "$@" <<'PY'
import datetime
import json
import os
import pathlib
import sys

prefix = pathlib.Path(sys.argv[1])
trace = sys.argv[2]
graph_trace = sys.argv[3]
pathlib.Path(f"{prefix}.manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "cwd": os.getcwd(),
            "command": sys.argv[4:],
            "nsys_version": pathlib.Path(f"{prefix}.nsys-version.txt").read_text().strip(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_revision": os.environ.get("SJ_BENCH_COMMIT"),
            "sj_use_triton_op": os.environ.get("SJ_USE_TRITON_OP"),
            "capture_range": "cudaProfilerApi",
            "trace": trace,
            "cuda_graph_trace": graph_trace,
            "pytorch_trace": "none",
            "python_sampling": False,
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
        nsys profile --trace="$trace" --cuda-graph-trace="$graph_trace" \
            --pytorch=none --python-sampling=false \
            --sample=none --cpuctxsw=none \
            --capture-range=cudaProfilerApi --capture-range-end=stop \
            --output="$prefix" "$@"
        ;;
    analyze)
        [[ $# -ge 3 && $# -le 4 ]] || usage
        report=$2
        output_dir=$3
        mkdir -p "$output_dir"
        nsys export --type=sqlite --force-overwrite=true \
            --output="${output_dir}/trace.sqlite" "$report"
        nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
            --report nvtx_gpu_proj_sum --format csv \
            --force-overwrite=true \
            --output "${output_dir}/stats" "${output_dir}/trace.sqlite"
        args=(analyze "${output_dir}/trace.sqlite" --output-dir "$output_dir")
        if [[ $# -eq 4 ]]; then
            args+=(--benchmark-json "$4")
        fi
        python "$root/analyze_nsys_snn.py" "${args[@]}"
        ;;
    compare)
        [[ $# -eq 4 ]] || usage
        python "$root/analyze_nsys_snn.py" compare "$2" "$3" --output "$4"
        ;;
    *) usage ;;
esac
