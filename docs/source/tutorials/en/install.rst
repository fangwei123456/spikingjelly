Installation Guide
==========================

中文版本：:doc:`../cn/install`

Prepare the environment
--------------------------

SpikingJelly 2.0 requires Python 3.11 or later and PyTorch 2.6 or later. Create a
virtual environment, then install Torch, torchvision and torchaudio for your
device with the command from the
`PyTorch installation selector <https://pytorch.org/get-started/locally/>`_:

.. code-block:: bash

    uv venv --python 3.11
    source .venv/bin/activate

Install the three packages together with that one command. If you install only
some of them, the installer may replace your Torch to satisfy the others.

This page follows the development source. For a PyPI release, read the
documentation of that version.

Install SpikingJelly
--------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Install
      - Command
    * - PyPI release
      - ``uv pip install spikingjelly``
    * - PyPI pre-release
      - ``uv pip install --pre spikingjelly``
    * - Release with Triton
      - ``uv pip install "spikingjelly[triton]"``
    * - Latest development source
      - ``uv pip install git+https://github.com/fangwei123456/spikingjelly.git``

For CPU only, install the release. For NVIDIA GPUs, install the release with
Triton: IF, LIF, PLIF and other neurons then use Triton kernels on CUDA
automatically. If your environment already has a Triton build that matches Torch
(3.2.0 or later), you do not need to install it again. Check its version with:

.. code-block:: bash

    python -c "import triton; print(triton.__version__)"

Native CUDA extensions are built manually. You only need them for native
operators such as fused projections, or when measurements show that the native
implementation is faster; see :ref:`install-native-cuda-en`. Triton and native
extensions can be installed together, and modules need no backend configuration.

.. figure:: /_static/tutorials/install/installation.svg
    :alt: Installation decision tree: prepare Python and Torch; CPU uses a regular installation, while NVIDIA CUDA can use reference execution, optional Triton or manually built native CUDA.
    :width: 100%

    Choose an installation for your device and needs.

For the development source with Triton:

.. code-block:: bash

    uv pip install "spikingjelly[triton] @ git+https://github.com/fangwei123456/spikingjelly.git"

Source is also available from `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_.

Developers with a source checkout can use ``uv pip install --editable ".[triton]"``.
Installation maps the root ``ops/`` directory to ``spikingjelly._ops``; setting
``PYTHONPATH`` alone does not replace installation. See ``CONTRIBUTING.md`` in
the repository for development conventions.

After installation, run ``uv pip check`` to check that dependency versions are consistent.

.. _install-native-cuda-en:

Build native CUDA manually
--------------------------

Regular wheels contain no precompiled native libraries. Manual source builds
are currently recommended, normally downloading the PyPI sdist and compiling
locally without a Git checkout. This command requires SpikingJelly >= 2.0.0;
older releases do not contain these operators.

Prepare CUDA-enabled Torch, a matching CUDA Toolkit with ``nvcc``, and a C++
compiler. Set ``CUDA_HOME`` if the toolkit is outside the default location.
Builds without a visible GPU must set ``TORCH_CUDA_ARCH_LIST`` to the actual
target architectures.

.. code-block:: bash

    uv pip install "setuptools>=77.0.3" ninja
    SJ_BUILD_NATIVE_CUDA=1 uv pip install \
      --no-build-isolation --no-binary spikingjelly \
      --reinstall-package spikingjelly --no-cache "spikingjelly>=2.0.0"

After a successful build, check the default eager binding in a new process on
an available target NVIDIA GPU:

.. code-block:: bash

    python -c \
      'import torch; from spikingjelly.activation_based import functional; print(functional.neuron_implementation("lif", torch.device("cuda:0")))'

A compatible native extension should report ``implementation: cuda``. If it
reports ``triton`` or ``torch``, check the native loading reason in
``unavailable``. The query does not run a neuron; profiles such as state
precision still affect individual calls. See the full checks below.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - Option
      - Effect
    * - ``SJ_BUILD_NATIVE_CUDA=1``
      - Require native extension compilation for this source build; it is disabled by default.
    * - ``--no-binary spikingjelly``
      - Download the SpikingJelly sdist; other dependencies can still use wheels.
    * - ``--no-build-isolation``
      - Use the current environment's Torch and build dependencies. The default isolated environment does not include the Torch needed for this build.
    * - ``--reinstall-package spikingjelly``
      - Reinstall SpikingJelly even if it is already installed.
    * - ``--no-cache``
      - Avoid reusing a previously built wheel after build settings change.

This command does not add optional Triton; install ``spikingjelly[triton]``
separately if needed. An explicit build request fails immediately if CUDA-enabled
Torch, the CUDA Toolkit, a compiler or required architecture settings are missing.
Compilation failures also fail installation. Without the environment variable,
builds remain pure Python. This check only runs during source builds; installing
an existing wheel or retaining an existing installation does not run it.
Reinstallation and disabling the cache remain necessary because ``--no-binary``
can still reuse uv's cached wheels.

If PyPI downloads are unavailable or you need development source, obtain an
OpenI checkout and use the same manual build:

.. code-block:: bash

    git clone https://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    uv pip install "setuptools>=77.0.3" ninja
    SJ_BUILD_NATIVE_CUDA=1 uv pip install --no-build-isolation .

A GitHub checkout works too. uv rebuilds and reinstalls local directories
explicitly passed on the command line, so the sdist's ``--no-binary`` and related
options are unnecessary. Changes to ``.cu`` or headers in an editable install
require another native build; they do not take effect like Python source changes.

Runtime only loads native binaries, without invoking a compiler. The current
loader checks the operator ABI, complete Torch version, CUDA version and target
GPU support; environment changes may require rebuilding. Triton retains its own
first-use JIT and caching, which require a working C compiler (for example GCC)
for its host launcher. A minimal runtime container may need that compiler even
without native CUDA extensions. Not every platform's CUDA Torch includes usable Triton.

Execution after installation
------------------------------

Move modules and inputs to the same device. Registered ordinary neurons follow
the selection below. CUDA checks candidates on first use for each device and
execution path, then reuses the binding without online benchmarking.

.. figure:: /_static/tutorials/install/execution.svg
    :alt: Ordinary registered neuron decision tree: CPU uses Torch; CUDA checks the input profile before choosing compatible implementations for eager or Inductor execution.
    :width: 100%

    CUDA arrows show candidate order. CUDA Graphs retain the captured function's selection.

Installing an implementation does not mean every call uses it:

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Feature
      - Requirements
    * - Registered ordinary neurons
      - Supported inputs, FP32 state and built-in surrogates can use fused execution. Ordinary module state follows input dtype; low-precision state or custom surrogates can use Torch reference equations.
    * - ComplementaryLIF (CLIF)
      - Matching CUDA FP32/FP16/BF16 tensors use optional Triton kernels, retaining the supplied spike function and state dtype. Auto mode can select Triton for training and inference, including ``torch.compile``; small batches may be slower. CPU, custom spike functions and other unsupported inputs use Torch formulas; an explicit Triton request rejects unsupported CUDA profiles. Select with ``SJ_CLIF_CUDA_IMPLEMENTATION=auto|triton|torch``; there is no native CUDA CLIF kernel.
    * - Explicit IF/LIF/PLIF precision
      - Requires CUDA Triton supporting the requested combination; other implementations cannot replace an explicitly requested numerical policy.
    * - FlexSN CUDA multi-step
      - Supported cores use Triton; known unsupported combinations can use Torch/HOP. Single-step execution uses Torch.
    * - Fused IF/LIF-Linear and packed/sparse projections
      - Compatible native extensions use the corresponding kernels; missing extensions use Torch reference execution. Triton does not provide these native fused projections' performance properties.

See :doc:`./precision` for input, state and recurrence precision;
:doc:`./triton_backend` for execution and compilation examples; and
:doc:`./flexsn` for custom dynamics.

Check the installation
--------------------------

First check the installation path and CUDA Torch:

.. code-block:: python

    import torch
    import spikingjelly

    print(spikingjelly.__file__)
    print(torch.__version__, torch.version.cuda, torch.cuda.is_available())

With an available NVIDIA GPU, query ordinary neuron bindings:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional

    device = torch.device("cuda:0")
    print(functional.neuron_implementation("lif", device, execution="eager"))
    print(functional.neuron_implementation("lif", device, execution="compile"))

A query initializes selection without computing neuron outputs or advancing
state. ``implementation`` reports the binding; ``unavailable`` explains why
earlier candidates were unavailable. Profiles such as low-precision state may
still use reference execution. Missing dependencies and known incompatibility
allow checking the next candidate; unknown JIT, kernel, OOM and gradient errors
are reported. Logging is silent by default; see :doc:`./triton_backend` for
logging and strict diagnostic environment variables. Restart after changing
dependencies, extensions or configuration.

Other optional dependencies
------------------------------

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Feature
      - Command
    * - :doc:`./nir_exchange`
      - ``uv pip install "spikingjelly[nir]"``
    * - Lightning integration
      - ``uv pip install "spikingjelly[lightning]"``
    * - Transformer Engine precision features
      - ``uv pip install "spikingjelly[fp8]"``; see :doc:`./precision` for scope. This does not enable every neuron FP8 combination.

See the repository's ``pyproject.toml`` for other extras and version constraints.
See :doc:`./migrate_from_legacy` for CuPy installation options.
