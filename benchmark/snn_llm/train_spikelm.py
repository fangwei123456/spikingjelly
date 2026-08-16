from importlib import import_module
from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if sys.argv[1:2] != ["spikelm-pretrain"]:
        sys.argv.insert(1, "spikelm-pretrain")
    import_module("benchmark.snn_llm.cli").main()
