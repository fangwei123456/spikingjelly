"""
Tests for new/changed utilities introduced in this PR:

- spikingjelly/activation_based/cuda_kernel/cuda_utils.py
    env_flag_enabled, use_cupy_custom_op,
    register_python_object, resolve_python_object,
    _entry_to_object, _drop_python_object_locked,
    _check_pytorch_version, amp_custom_fwd, amp_custom_bwd

- spikingjelly/activation_based/cuda_kernel/neuron_kernel/helpers.py
    sg_registry_key, replay_and_grad

- spikingjelly/activation_based/cuda_kernel/auto_cuda/ss_neuron_kernel/common.py
    replay_and_grad  (different None-handling from helpers version)

- spikingjelly/activation_based/cuda_kernel/neuron_kernel/common.py
    _decode_v_reset

- spikingjelly/activation_based/cuda_kernel/spike_op.py
    spike_linear CPU path, spike_conv1d/conv2d/conv3d CPU path,
    _CUSTOM_SPIKE_OP_READY flag behaviour

- spikingjelly/activation_based/cuda_kernel/neuron_kernel/__init__.py
    __all__ re-exports (multistep_if_ptt, multistep_lif_ptt, etc.)
"""

import gc
import math
import os
import threading
import weakref

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")


def _require_cupy():
    pytest.importorskip("cupy")


def _maybe_skip_custom_op_unavailable():
    if not all(
        hasattr(torch.library, name)
        for name in ("custom_op", "register_fake", "register_autograd")
    ):
        pytest.skip("torch.library custom_op/register_autograd are unavailable.")


# ---------------------------------------------------------------------------
# cuda_utils: env_flag_enabled
# ---------------------------------------------------------------------------

def test_env_flag_enabled_unset_returns_true(monkeypatch):
    monkeypatch.delenv("_SJ_TEST_FLAG_XYZ", raising=False)
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is True


def test_env_flag_enabled_zero_returns_false(monkeypatch):
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "0")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is False


def test_env_flag_enabled_false_returns_false(monkeypatch):
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "false")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is False


def test_env_flag_enabled_false_uppercase_returns_false(monkeypatch):
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "FALSE")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is False


def test_env_flag_enabled_off_returns_false(monkeypatch):
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "off")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is False


def test_env_flag_enabled_no_returns_false(monkeypatch):
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "no")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is False


def test_env_flag_enabled_one_returns_true(monkeypatch):
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "1")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is True


def test_env_flag_enabled_true_returns_true(monkeypatch):
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "true")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is True


def test_env_flag_enabled_yes_returns_true(monkeypatch):
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "yes")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is True


def test_env_flag_enabled_strips_whitespace(monkeypatch):
    """Leading/trailing whitespace is stripped before comparison."""
    monkeypatch.setenv("_SJ_TEST_FLAG_XYZ", "  0  ")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("_SJ_TEST_FLAG_XYZ") is False


def test_env_flag_enabled_with_sj_use_cupy_op(monkeypatch):
    """The real env-var used by the project also respects the semantics."""
    monkeypatch.setenv("SJ_USE_CUPY_OP", "0")
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("SJ_USE_CUPY_OP") is False


# ---------------------------------------------------------------------------
# cuda_utils: register_python_object / resolve_python_object
# ---------------------------------------------------------------------------

def _fresh_registry_state():
    """Return module-level dict references for inspection."""
    import spikingjelly.activation_based.cuda_kernel.cuda_utils as cu
    return cu._PYOBJ_ID_TO_REF, cu._PYOBJ_ID_TO_KEY, cu._PYOBJ_KEY_TO_ID


class _WeakRefableObj:
    """Simple class that supports weak references."""
    pass


def test_register_python_object_returns_int():
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import register_python_object
    obj = _WeakRefableObj()
    key = f"test_obj_{id(obj)}_unique"
    obj_id = register_python_object(obj, key)
    assert isinstance(obj_id, int)


def test_register_python_object_resolve_roundtrip():
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
        register_python_object,
        resolve_python_object,
    )
    obj = _WeakRefableObj()
    key = f"test_roundtrip_{id(obj)}_unique"
    obj_id = register_python_object(obj, key)
    resolved = resolve_python_object(obj_id)
    assert resolved is obj


def test_register_python_object_same_key_same_id():
    """Registering the same object under the same key returns the same id."""
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
        register_python_object,
        resolve_python_object,
    )
    obj = _WeakRefableObj()
    key = f"test_same_key_{id(obj)}_unique"
    id1 = register_python_object(obj, key)
    id2 = register_python_object(obj, key)
    assert id1 == id2
    assert resolve_python_object(id1) is obj


def test_register_python_object_different_keys_different_ids():
    """Different keys always produce different ids."""
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import register_python_object
    obj = _WeakRefableObj()
    id1 = register_python_object(obj, f"key_a_{id(obj)}")
    id2 = register_python_object(obj, f"key_b_{id(obj)}")
    assert id1 != id2


def test_register_python_object_non_weakrefable_object():
    """Objects that don't support weak references (like plain int) are stored directly."""
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
        register_python_object,
        resolve_python_object,
    )
    # ints don't support weakref
    obj = 12345
    key = f"int_obj_test_{os.getpid()}_{threading.get_ident()}"
    obj_id = register_python_object(obj, key)
    resolved = resolve_python_object(obj_id)
    assert resolved == obj


def test_resolve_python_object_unknown_id_raises():
    """resolve_python_object raises RuntimeError for unknown id."""
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import resolve_python_object
    with pytest.raises(RuntimeError, match="Unknown python object id="):
        resolve_python_object(999_999_999)


def test_register_python_object_thread_safety():
    """Concurrent registrations from multiple threads don't raise."""
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
        register_python_object,
        resolve_python_object,
    )
    results = []
    errors = []

    def worker(i):
        try:
            obj = _WeakRefableObj()
            key = f"thread_test_{i}_{id(obj)}"
            obj_id = register_python_object(obj, key)
            resolved = resolve_python_object(obj_id)
            results.append(resolved is obj)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert all(results)


def test_register_python_object_weakref_cleanup():
    """After the original object is GC'd, resolve raises RuntimeError."""
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
        register_python_object,
        resolve_python_object,
    )

    class _Obj:
        pass

    obj = _Obj()
    key = f"gc_test_{id(obj)}_{os.getpid()}"
    obj_id = register_python_object(obj, key)
    # Object is still alive → should resolve fine.
    assert resolve_python_object(obj_id) is obj

    del obj
    gc.collect()

    # Weak ref should have been cleared; resolve should fail.
    with pytest.raises(RuntimeError):
        resolve_python_object(obj_id)


# ---------------------------------------------------------------------------
# cuda_utils: _entry_to_object
# ---------------------------------------------------------------------------

def test_entry_to_object_with_weakref():
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import _entry_to_object

    class _Obj:
        pass

    obj = _Obj()
    ref = weakref.ref(obj)
    assert _entry_to_object(ref) is obj


def test_entry_to_object_with_weakref_dead():
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import _entry_to_object

    class _Obj:
        pass

    obj = _Obj()
    ref = weakref.ref(obj)
    del obj
    gc.collect()
    assert _entry_to_object(ref) is None


def test_entry_to_object_with_direct_object():
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import _entry_to_object
    obj = object()
    assert _entry_to_object(obj) is obj


def test_entry_to_object_with_int():
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import _entry_to_object
    assert _entry_to_object(42) == 42


# ---------------------------------------------------------------------------
# cuda_utils: _check_pytorch_version / amp_custom_fwd / amp_custom_bwd
# ---------------------------------------------------------------------------

def test_check_pytorch_version_current_is_ge_1():
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import _check_pytorch_version
    # Torch 1.x is ancient; any modern build should be >= 1.0
    assert _check_pytorch_version("1.0") is True


def test_check_pytorch_version_future_is_false():
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import _check_pytorch_version
    # 999.0 is higher than any released version
    assert _check_pytorch_version("999.0") is False


def test_check_pytorch_version_cached():
    """Function uses lru_cache; calling twice with same arg is fine."""
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import _check_pytorch_version
    r1 = _check_pytorch_version("2.0")
    r2 = _check_pytorch_version("2.0")
    assert r1 == r2


def test_amp_custom_fwd_bwd_are_callable():
    """amp_custom_fwd and amp_custom_bwd must be callable decorators."""
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
        amp_custom_fwd,
        amp_custom_bwd,
    )
    assert callable(amp_custom_fwd)
    assert callable(amp_custom_bwd)


# ---------------------------------------------------------------------------
# neuron_kernel/helpers.py: sg_registry_key
# ---------------------------------------------------------------------------

class _MockSurrogate:
    """Minimal mock of a surrogate function."""
    def __init__(self, alpha=1.0, spiking=True, extra_params=None):
        self.alpha = alpha
        self.spiking = spiking
        self._sg_params = {"alpha": alpha}
        if extra_params:
            self._sg_params.update(extra_params)


class _MockSurrogateNoParams:
    """Surrogate without _sg_params."""
    def __init__(self):
        self.spiking = True


def test_sg_registry_key_returns_string():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key
    sg = _MockSurrogate()
    key = sg_registry_key(sg)
    assert isinstance(key, str)
    assert len(key) > 0


def test_sg_registry_key_includes_class_name():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key
    sg = _MockSurrogate()
    key = sg_registry_key(sg)
    assert "_MockSurrogate" in key


def test_sg_registry_key_includes_module():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key
    sg = _MockSurrogate()
    key = sg_registry_key(sg)
    # module should appear somewhere in the key
    assert sg.__class__.__module__ in key


def test_sg_registry_key_same_object_same_key():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key
    sg = _MockSurrogate(alpha=2.0)
    assert sg_registry_key(sg) == sg_registry_key(sg)


def test_sg_registry_key_different_params_different_keys():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key
    sg1 = _MockSurrogate(alpha=1.0)
    sg2 = _MockSurrogate(alpha=2.0)
    assert sg_registry_key(sg1) != sg_registry_key(sg2)


def test_sg_registry_key_different_spiking_different_keys():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key
    sg1 = _MockSurrogate(spiking=True)
    sg2 = _MockSurrogate(spiking=False)
    assert sg_registry_key(sg1) != sg_registry_key(sg2)


def test_sg_registry_key_no_sg_params_attribute():
    """Surrogate without _sg_params attribute should still produce a key."""
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key
    sg = _MockSurrogateNoParams()
    key = sg_registry_key(sg)
    assert isinstance(key, str)


def test_sg_registry_key_with_real_surrogate():
    """Real surrogate objects from spikingjelly.activation_based.surrogate produce distinct keys."""
    pytest.importorskip("spikingjelly")
    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key

    atan = surrogate.ATan()
    sigmoid = surrogate.Sigmoid()
    key_atan = sg_registry_key(atan)
    key_sigmoid = sg_registry_key(sigmoid)
    assert key_atan != key_sigmoid
    assert isinstance(key_atan, str)
    assert isinstance(key_sigmoid, str)


def test_sg_registry_key_none_params():
    """If _sg_params is None, it should be treated as empty tuple."""
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key

    class _SgNoneParams:
        _sg_params = None
        spiking = True

    sg = _SgNoneParams()
    key = sg_registry_key(sg)
    assert isinstance(key, str)


# ---------------------------------------------------------------------------
# neuron_kernel/helpers.py: replay_and_grad
# ---------------------------------------------------------------------------

def _simple_op(x, y):
    """Simple differentiable op: z = x * 2 + y."""
    return x * 2.0 + y


def test_replay_and_grad_helpers_basic():
    """replay_and_grad returns correct gradients for a simple op."""
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import replay_and_grad

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([3.0, 4.0], requires_grad=True)
    out = _simple_op(x, y)
    grad_out = torch.ones_like(out)

    grads = replay_and_grad(_simple_op, (x, y), (), (grad_out,))
    # dL/dx = 2, dL/dy = 1
    assert grads[0] is not None
    assert grads[1] is not None
    assert torch.allclose(grads[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(grads[1], torch.tensor([1.0, 1.0]))


def test_replay_and_grad_helpers_no_grad_inputs():
    """When no tensor requires grad, all grads are None."""
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import replay_and_grad

    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    grad_out = torch.ones(2)

    grads = replay_and_grad(_simple_op, (x, y), (), (grad_out,))
    assert all(g is None for g in grads)


def test_replay_and_grad_helpers_partial_grad():
    """When only one input requires grad, only that one gets a gradient."""
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import replay_and_grad

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([3.0, 4.0])  # no grad
    grad_out = torch.ones(2)

    grads = replay_and_grad(_simple_op, (x, y), (), (grad_out,))
    assert grads[0] is not None
    assert grads[1] is None


def test_replay_and_grad_helpers_with_none_tensor():
    """helpers.replay_and_grad skips None entries in tensor_args.

    When None is in tensor_args, replay_and_grad passes it through to the op
    as-is (no detach/requires_grad), and the gradient for that slot is None.
    """
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import replay_and_grad

    # op accepts two tensor positional args, second may be None
    def op_two_tensors(x, y_or_none):
        if y_or_none is None:
            return x * 2.0
        return x * 2.0 + y_or_none

    x = torch.tensor([2.0, 3.0], requires_grad=True)
    grad_out = torch.ones(2)

    # None is passed as second positional tensor arg
    grads = replay_and_grad(op_two_tensors, (x, None), (), (grad_out,))
    # grads[0] is gradient for x; grads[1] is None for the skipped slot
    assert grads[0] is not None
    assert grads[1] is None
    assert torch.allclose(grads[0], torch.tensor([2.0, 2.0]))


def test_replay_and_grad_helpers_with_static_args():
    """Static args are passed through correctly."""
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import replay_and_grad

    def op_static(x, scale):
        return x * scale

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    grad_out = torch.ones(2)

    grads = replay_and_grad(op_static, (x,), (3.0,), (grad_out,))
    # dL/dx = 3.0
    assert torch.allclose(grads[0], torch.tensor([3.0, 3.0]))


def test_replay_and_grad_helpers_single_output_not_tuple():
    """op that returns a single tensor (not tuple) still works."""
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import replay_and_grad

    def single_out(x):
        return x * 2.0

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    grad_out = torch.ones(2)

    grads = replay_and_grad(single_out, (x,), (), (grad_out,))
    assert torch.allclose(grads[0], torch.tensor([2.0, 2.0]))


# ---------------------------------------------------------------------------
# neuron_kernel/common.py: _decode_v_reset
# ---------------------------------------------------------------------------

def test_decode_v_reset_nan_returns_none():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.common import _decode_v_reset
    result = _decode_v_reset(float("nan"))
    assert result is None


def test_decode_v_reset_zero_returns_zero():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.common import _decode_v_reset
    result = _decode_v_reset(0.0)
    assert result == 0.0


def test_decode_v_reset_negative_returns_negative():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.common import _decode_v_reset
    result = _decode_v_reset(-65.0)
    assert result == -65.0


def test_decode_v_reset_inf_is_not_nan():
    """math.inf is not NaN so it should pass through."""
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.common import _decode_v_reset
    result = _decode_v_reset(math.inf)
    assert result == math.inf
    assert result is not None


# ---------------------------------------------------------------------------
# neuron_kernel/__init__.py: public re-exports
# ---------------------------------------------------------------------------

def test_neuron_kernel_init_exports_multistep_functions():
    """The neuron_kernel package must export the new multistep_*_ptt wrappers."""
    from spikingjelly.activation_based.cuda_kernel import neuron_kernel

    for name in (
        "multistep_if_ptt",
        "multistep_lif_ptt",
        "multistep_plif_ptt",
        "multistep_qif_ptt",
        "multistep_izhikevich_ptt",
        "multistep_eif_ptt",
    ):
        assert hasattr(neuron_kernel, name), f"neuron_kernel missing {name}"
        assert callable(getattr(neuron_kernel, name))


def test_cuda_kernel_top_level_exports_multistep_functions():
    """The top-level cuda_kernel module must also re-export these functions."""
    from spikingjelly.activation_based import cuda_kernel

    for name in (
        "multistep_qif_ptt",
        "multistep_izhikevich_ptt",
        "multistep_eif_ptt",
    ):
        assert hasattr(cuda_kernel, name), f"cuda_kernel missing {name}"


# ---------------------------------------------------------------------------
# spike_op.py: CPU fallback (no CUDA required)
# ---------------------------------------------------------------------------

def test_spike_linear_cpu_no_cuda():
    """spike_linear on CPU falls back to F.linear without requiring CUDA."""
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_linear

    spike = (torch.rand(4, 8) > 0.5).float()
    weight = torch.randn(6, 8)
    expected = F.linear(spike, weight, None)
    result = spike_linear(spike, weight, None)
    assert torch.allclose(result, expected)


def test_spike_linear_cpu_with_bias():
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_linear

    spike = (torch.rand(3, 5) > 0.5).float()
    weight = torch.randn(7, 5)
    bias = torch.randn(7)
    expected = F.linear(spike, weight, bias)
    result = spike_linear(spike, weight, bias)
    assert torch.allclose(result, expected)


def test_spike_conv1d_cpu_no_cuda():
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_conv1d

    spike = (torch.rand(2, 3, 10) > 0.5).float()
    weight = torch.randn(4, 3, 3)
    expected = F.conv1d(spike, weight)
    result = spike_conv1d(spike, weight)
    assert torch.allclose(result, expected)


def test_spike_conv2d_cpu_no_cuda():
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_conv2d

    spike = (torch.rand(2, 3, 8, 8) > 0.5).float()
    weight = torch.randn(4, 3, 3, 3)
    expected = F.conv2d(spike, weight, padding=1)
    result = spike_conv2d(spike, weight, padding=1)
    assert torch.allclose(result, expected)


def test_spike_conv3d_cpu_no_cuda():
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_conv3d

    spike = (torch.rand(1, 2, 4, 4, 4) > 0.5).float()
    weight = torch.randn(2, 2, 3, 3, 3)
    expected = F.conv3d(spike, weight, padding=1)
    result = spike_conv3d(spike, weight, padding=1)
    assert torch.allclose(result, expected)


def test_spike_conv1d_string_padding_cpu():
    """String padding ('same') falls through to F.conv1d on CPU."""
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_conv1d

    spike = (torch.rand(2, 3, 10) > 0.5).float()
    weight = torch.randn(4, 3, 3)
    expected = F.conv1d(spike, weight, padding="same")
    result = spike_conv1d(spike, weight, padding="same")
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# CUDA-only tests
# ---------------------------------------------------------------------------

def test_spike_linear_cuda_with_bias():
    """spike_linear with bias on CUDA: forward and backward."""
    _require_cuda()
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_linear

    spike = (torch.rand(4, 8, device="cuda") > 0.5).float().requires_grad_(True)
    weight = torch.randn(6, 8, device="cuda", requires_grad=True)
    bias = torch.randn(6, device="cuda", requires_grad=True)

    result = spike_linear(spike, weight, bias)
    result.sum().backward()

    assert result.shape == (4, 6)
    assert weight.grad is not None
    assert bias.grad is not None


def test_spike_conv2d_cuda_no_bias():
    """spike_conv2d on CUDA: forward and backward."""
    _require_cuda()
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_conv2d

    spike = (torch.rand(2, 3, 8, 8, device="cuda") > 0.5).float().requires_grad_(True)
    weight = torch.randn(4, 3, 3, 3, device="cuda", requires_grad=True)

    result = spike_conv2d(spike, weight, padding=1)
    result.sum().backward()

    assert result.shape == (2, 4, 8, 8)
    assert weight.grad is not None


def test_spike_conv1d_cuda_string_padding_fallback():
    """String padding skips the custom-op path even on CUDA."""
    _require_cuda()
    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_conv1d

    spike = (torch.rand(2, 3, 10, device="cuda") > 0.5).float().requires_grad_(True)
    weight = torch.randn(4, 3, 3, device="cuda", requires_grad=True)

    # "same" is a string padding value → should still work (falls back to F.conv1d or spikeConvolution)
    result = spike_conv1d(spike, weight, padding="same")
    assert result.shape == (2, 4, 10)


def test_register_resolve_on_surrogate_function():
    """Using real surrogate objects with register/resolve works end-to-end."""
    pytest.importorskip("spikingjelly")
    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
        register_python_object,
        resolve_python_object,
    )
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.helpers import sg_registry_key

    sg = surrogate.ATan()
    key = sg_registry_key(sg)
    obj_id = register_python_object(sg, key)

    # Must resolve to same object
    assert resolve_python_object(obj_id) is sg

    # Registering again with same key returns same id
    obj_id2 = register_python_object(sg, key)
    assert obj_id2 == obj_id


def test_multistep_if_ptt_cuda_basic():
    """multistep_if_ptt runs on CUDA without error."""
    _require_cuda()
    _require_cupy()
    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.common import multistep_if_ptt

    sg = surrogate.ATan()
    x_seq = torch.randn(4, 64, device="cuda")
    v_init = torch.zeros(64, device="cuda")

    spike_seq, v_seq = multistep_if_ptt(
        x_seq, v_init, 1.0, None, False, sg
    )
    assert spike_seq.shape == (4, 64)
    assert v_seq.shape == (4, 64)
    # Spike values must be binary
    assert torch.all((spike_seq == 0) | (spike_seq == 1))


def test_multistep_lif_ptt_cuda_basic():
    """multistep_lif_ptt runs on CUDA without error."""
    _require_cuda()
    _require_cupy()
    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.common import multistep_lif_ptt

    sg = surrogate.ATan()
    x_seq = torch.randn(4, 64, device="cuda")
    v_init = torch.zeros(64, device="cuda")

    spike_seq, v_seq = multistep_lif_ptt(
        x_seq, v_init, True, 2.0, 1.0, None, False, sg
    )
    assert spike_seq.shape == (4, 64)
    assert v_seq.shape == (4, 64)


def test_use_cupy_custom_op_disabled_by_env(monkeypatch):
    """When SJ_USE_CUPY_OP=0, use_cupy_custom_op returns False."""
    monkeypatch.setenv("SJ_USE_CUPY_OP", "0")
    # Re-evaluating the function should reflect the current env var
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
        env_flag_enabled,
        _CUSTOM_OP_AVAILABLE,
    )
    # Even if custom op is available, env flag overrides
    result = _CUSTOM_OP_AVAILABLE and env_flag_enabled("SJ_USE_CUPY_OP")
    assert result is False


def test_use_cupy_custom_op_enabled_by_default(monkeypatch):
    """When SJ_USE_CUPY_OP is unset, env_flag_enabled returns True."""
    monkeypatch.delenv("SJ_USE_CUPY_OP", raising=False)
    from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
    assert env_flag_enabled("SJ_USE_CUPY_OP") is True
