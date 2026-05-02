import torch

from ..cuda_utils import resolve_python_object, use_cupy_custom_op
from .common import (
    MultiStepEIFNodePTT,
    _CapturedAutogradCtx,
    _decode_v_reset,
    _resolve_sg_cuda_code_fun,
    _sg_obj_id,
    _stash_capture_ctx,
    _take_capture_ctx,
    cupy,
)

__all__ = ["MultiStepEIFNodePTT", "multistep_eif_ptt"]


if use_cupy_custom_op() and cupy is not None:

    @torch.library.custom_op("sj::cupy_multistep_eif_forward", mutates_args=())
    def cupy_multistep_eif_forward(
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        tau: float,
        v_threshold: float,
        v_reset: float,
        v_rest: float,
        theta_rh: float,
        delta_T: float,
        detach_reset: bool,
        sg_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sg = resolve_python_object(sg_id)
        captured_ctx = _CapturedAutogradCtx()
        spike_seq, v_seq = MultiStepEIFNodePTT.forward(
            captured_ctx,
            x_seq,
            v_init,
            tau,
            v_threshold,
            _decode_v_reset(v_reset),
            v_rest,
            theta_rh,
            delta_T,
            detach_reset,
            _resolve_sg_cuda_code_fun(sg),
        )
        capture_id = _stash_capture_ctx(captured_ctx)
        capture_token = torch.tensor(capture_id, device=x_seq.device, dtype=torch.int64)
        return spike_seq, v_seq, capture_token

    @torch.library.register_fake("sj::cupy_multistep_eif_forward")
    def _cupy_multistep_eif_forward_fake(*args):
        x_seq = args[0]
        return (
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty((), dtype=torch.int64),
        )

    def _setup_eif_ctx(ctx, inputs, output):
        capture_token = output[2]
        if capture_token.is_meta:
            ctx.captured = None
            return
        ctx.captured = _take_capture_ctx(int(capture_token.item()))

    def _eif_bw(ctx, grad_spike_seq, grad_v_seq, grad_capture_token):
        del grad_capture_token
        if ctx.captured is None:
            raise RuntimeError("Missing captured context for eif backward.")
        grads = MultiStepEIFNodePTT.backward(ctx.captured, grad_spike_seq, grad_v_seq)
        return grads[0], grads[1], None, None, None, None, None, None, None, None

    torch.library.register_autograd(
        "sj::cupy_multistep_eif_forward",
        _eif_bw,
        setup_context=_setup_eif_ctx,
    )


def multistep_eif_ptt(
    x_seq,
    v_init,
    tau,
    v_threshold,
    v_reset,
    v_rest,
    theta_rh,
    delta_T,
    detach_reset,
    surrogate_function,
):
    sg_id = _sg_obj_id(surrogate_function)
    v_reset_value = float("nan") if v_reset is None else float(v_reset)
    spike_seq, v_seq, _ = cupy_multistep_eif_forward(
        x_seq,
        v_init,
        tau,
        v_threshold,
        v_reset_value,
        v_rest,
        theta_rh,
        delta_T,
        detach_reset,
        sg_id,
    )
    return spike_seq, v_seq
