import subprocess
import sys

import pytest


@pytest.mark.parametrize("missing", ["nir", "nirtorch"])
def test_nir_exchange_imports_without_optional_dependencies(missing):
    code = f"""
import sys
sys.modules[{missing!r}] = None
import torch
from spikingjelly.activation_based import nir_exchange
for function, args in (
    (nir_exchange.import_from_nir, (None,)),
    (nir_exchange.export_to_nir, (torch.nn.Identity(), torch.zeros(1))),
):
    try:
        function(*args)
    except ImportError as error:
        assert "spikingjelly[nir]" in str(error), str(error)
    else:
        raise AssertionError("Missing optional dependency must raise ImportError")
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_neuron_import_does_not_require_base_node_to_be_exported_first():
    code = """
import importlib.abc
import importlib.util
import sys
from pathlib import Path
import spikingjelly

path = Path(spikingjelly.__file__).parent / "activation_based/neuron/__init__.py"
class Reordered(importlib.abc.SourceLoader):
    def get_filename(self, fullname):
        return str(path)
    def get_data(self, filename):
        source = path.read_text().replace("from .base_node import *\\n", "")
        return (source + "\\nfrom .base_node import *\\n").encode()
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, locations, target=None):
        if fullname == "spikingjelly.activation_based.neuron":
            return importlib.util.spec_from_file_location(
                fullname, path, loader=Reordered(),
                submodule_search_locations=[str(path.parent)],
            )
sys.meta_path.insert(0, Finder())
from spikingjelly.activation_based import neuron, functional
assert neuron.LIFNode.__mro__[1] is neuron.BaseNode
assert callable(functional.set_threshold_margin)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
