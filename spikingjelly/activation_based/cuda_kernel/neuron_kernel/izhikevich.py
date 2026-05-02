import torch

from ..cuda_utils import resolve_python_object, use_cupy_custom_op
from .common import (
    MultiStepIzhikevichNodePTT,
    _CapturedAutogradCtx,
    _decode_v_reset,
    _resolve_sg_cuda_code_fun,
    _sg_obj_id,
    _stash_capture_ctx,
    _take_capture_ctx,
    cupy,
)

__all__ = ["MultiStepIzhikevichNodePTT", "multistep_izhikevich_ptt"]


if use_cupy_custom_op() and cupy is not None:

    @torch.library.custom_op("sj::cupy_multistep_izhikevich_forward", mutates_args=())
    def cupy_multistep_izhikevich_forward(
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        w_init: torch.Tensor,
        tau: float,
        v_threshold: float,
        v_reset: float,
        v_rest: float,
        a: float,
        b: float,
        tau_w: float,
        v_c: float,
        a0: float,
        detach_reset: bool,
        sg_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sg = resolve_python_object(sg_id)
        captured_ctx = _CapturedAutogradCtx()
        spike_seq, v_seq, w_seq = MultiStepIzhikevichNodePTT.forward(
            captured_ctx,
            x_seq,
            v_init,
            w_init,
            tau,
            v_threshold,
            _decode_v_reset(v_reset),
            v_rest,
            a,
            b,
            tau_w,
            v_c,
            a0,
            detach_reset,
            _resolve_sg_cuda_code_fun(sg),
        )
        capture_id = _stash_capture_ctx(captured_ctx)
        capture_token = torch.tensor(capture_id, device=x_seq.device, dtype=torch.int64)
        return spike_seq, v_seq, w_seq, capture_token

    @torch.library.register_fake("sj::cupy_multistep_izhikevich_forward")
    def _cupy_multistep_izhikevich_forward_fake(*args):
        x_seq = args[0]
        return (
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty((), dtype=torch.int64),
        )

    def _setup_iz_ctx(ctx, inputs, output):
        capture_token = output[3]
        if capture_token.is_meta:
            ctx.captured = None
            return
        ctx.captured = _take_capture_ctx(int(capture_token.item()))

    def _iz_bw(ctx, grad_spike_seq, grad_v_seq, grad_w_seq, grad_capture_token):
        del grad_capture_token
        if ctx.captured is None:
            raise RuntimeError("Missing captured context for izhikevich backward.")
        grads = MultiStepIzhikevichNodePTT.backward(
            ctx.captured, grad_spike_seq, grad_v_seq, grad_w_seq
        )
        return (
            grads[0],
            grads[1],
            grads[2],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "sj::cupy_multistep_izhikevich_forward",
        _iz_bw,
        setup_context=_setup_iz_ctx,
    )


def multistep_izhikevich_ptt(
    x_seq,
    v_init,
    w_init,
    tau,
    v_threshold,
    v_reset,
    v_rest,
    a,
    b,
    tau_w,
    v_c,
    a0,
    detach_reset,
    surrogate_function,
):
    sg_id = _sg_obj_id(surrogate_function)
    v_reset_value = float("nan") if v_reset is None else float(v_reset)
    spike_seq, v_seq, w_seq, _ = cupy_multistep_izhikevich_forward(
        x_seq,
        v_init,
        w_init,
        tau,
        v_threshold,
        v_reset_value,
        v_rest,
        a,
        b,
        tau_w,
        v_c,
        a0,
        detach_reset,
        sg_id,
    )
    return spike_seq, v_seq, w_seq
