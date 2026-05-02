import torch

from ..cuda_utils import resolve_python_object, use_cupy_custom_op
from .common import (
    MultiStepQIFNodePTT,
    _CapturedAutogradCtx,
    _decode_v_reset,
    _resolve_sg_cuda_code_fun,
    _sg_obj_id,
    _stash_capture_ctx,
    _take_capture_ctx,
    cupy,
)

__all__ = ["MultiStepQIFNodePTT", "multistep_qif_ptt"]


if use_cupy_custom_op() and cupy is not None:

    @torch.library.custom_op("sj::cupy_multistep_qif_forward", mutates_args=())
    def cupy_multistep_qif_forward(
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        tau: float,
        v_threshold: float,
        v_reset: float,
        v_rest: float,
        v_c: float,
        a0: float,
        detach_reset: bool,
        sg_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sg = resolve_python_object(sg_id)
        captured_ctx = _CapturedAutogradCtx()
        spike_seq, v_seq = MultiStepQIFNodePTT.forward(
            captured_ctx,
            x_seq,
            v_init,
            tau,
            v_threshold,
            _decode_v_reset(v_reset),
            v_rest,
            v_c,
            a0,
            detach_reset,
            _resolve_sg_cuda_code_fun(sg),
        )
        capture_id = _stash_capture_ctx(captured_ctx)
        capture_token = torch.tensor(capture_id, device=x_seq.device, dtype=torch.int64)
        return spike_seq, v_seq, capture_token

    @torch.library.register_fake("sj::cupy_multistep_qif_forward")
    def _cupy_multistep_qif_forward_fake(
        x_seq, v_init, tau, v_threshold, v_reset, v_rest, v_c, a0, detach_reset, sg_id
    ):
        return (
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty((), dtype=torch.int64),
        )

    def _setup_qif_ctx(ctx, inputs, output):
        capture_token = output[2]
        if capture_token.is_meta:
            ctx.captured = None
            return
        ctx.captured = _take_capture_ctx(int(capture_token.item()))

    def _qif_bw(ctx, grad_spike_seq, grad_v_seq, grad_capture_token):
        del grad_capture_token
        if ctx.captured is None:
            raise RuntimeError("Missing captured context for qif backward.")
        grads = MultiStepQIFNodePTT.backward(ctx.captured, grad_spike_seq, grad_v_seq)
        return grads[0], grads[1], None, None, None, None, None, None, None, None

    torch.library.register_autograd(
        "sj::cupy_multistep_qif_forward",
        _qif_bw,
        setup_context=_setup_qif_ctx,
    )


def multistep_qif_ptt(
    x_seq,
    v_init,
    tau,
    v_threshold,
    v_reset,
    v_rest,
    v_c,
    a0,
    detach_reset,
    surrogate_function,
):
    sg_id = _sg_obj_id(surrogate_function)
    v_reset_value = float("nan") if v_reset is None else float(v_reset)
    spike_seq, v_seq, _ = cupy_multistep_qif_forward(
        x_seq,
        v_init,
        tau,
        v_threshold,
        v_reset_value,
        v_rest,
        v_c,
        a0,
        detach_reset,
        sg_id,
    )
    return spike_seq, v_seq
