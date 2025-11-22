from typing import Optional


def wrap_return_codes(y: Optional[str], codes: str):
    if y is None:
        return f'({codes})'
    else:
        return f'{y} = {codes};'


def float2half2(y: Optional[str], x: str):
    codes = f'__float2half2_rn({x})'
    return wrap_return_codes(y, codes)


def constant(y: Optional[str], x: float, dtype: str):
    if dtype == 'float':
        codes = f'{x}f'
    elif dtype == 'half2':
        codes = f'__float2half2_rn({x}f)'
    else:
        raise NotImplementedError(dtype)

    return wrap_return_codes(y, codes)



def abs(y: Optional[str], x: str, dtype: str):
    if dtype == 'float':
        codes = f'fabsf({x})'
    elif dtype == 'half2':
        codes = f'__habs2({x})'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(y, codes)


def power(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':
        codes = f'__powf({x, y})'
    elif dtype == 'half2':
        # CUDA FP16 does not provide powf function. We use z = 2 ** (log2(x) * y)
        codes = f'h2exp(__hmul2(h2log2({x}), {y}))'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(z, codes)

def if_else(z: Optional[str], x: str, y: str, mask: str, dtype: str):
    # z = x * mask + y * (1. - mask)
    if dtype == 'float':
        codes = f'{x} * {mask} + {y} * (1.0f - {mask})'
    elif dtype == 'half2':
        codes = f'__hfma2({x}, {mask}, __hmul2({y}, __hsub2(__float2half2_rn(1.0f), {mask})))'
    else:
        raise NotImplementedError(dtype)

    return wrap_return_codes(z, codes)

def if_else_else(w: Optional[str], x: str, y: str, z: str, mask_x: str, mask_y: str, dtype: str):
    # w = mask_x * x + mask_y * y + (1. - mask_x * mask_y) * z
    if dtype == 'float':
        codes = f'{mask_x} * {x} + {mask_y} * {y} + (1. - {mask_x} * {mask_y}) * {z}'
    else:
        codes = f'__hadd2(__hadd2(__hmul2({mask_x}, {x}), __hmul2({mask_y}, {y})), __hmul2({z}, __hsub2(__float2half_rn(1.0f), __hmul2({mask_x}, {mask_y}))))'

    return wrap_return_codes(w, codes)



def greater_equal(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':
        codes = f'(float) ({x} >= {y})'
    elif dtype == 'half2':
        codes = f'__hgeu2({x}, {y})'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(z, codes)

def greater_than(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':
        codes = f'(float) ({x} > {y})'
    elif dtype == 'half2':
        codes = f'__hgtu2({x}, {y})'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(z, codes)

def minimal(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':
        codes = f'min({x}, {y})'
    elif dtype == 'half2':
        codes = f'__hmin2({x}, {y})'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(z, codes)

def maximum(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':
        codes = f'max({x}, {y})'
    elif dtype == 'half2':
        codes = f'__hmax2({x}, {y})'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(z, codes)

def add(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':

        if x == '0.0f':
            codes = f'{y}'

        elif y == '0.0f':
            codes = f'{x}'

        else:
            codes = f'{x} + {y}'

    elif dtype == 'half2':
        if x == '__float2half2_rn(0.0f)':
            codes = f'{y}'

        elif y == '__float2half2_rn(0.0f)':
            codes = f'{x}'
        else:
            codes = f'__hadd2({x}, {y})'
    else:
        raise NotImplementedError(dtype)

    return wrap_return_codes(z, codes)


def sub(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':

        if y == '0.0f':
            codes = f'{x}'
        else:
            codes = f'{x} - {y}'

    elif dtype == 'half2':

        if y == '__float2half2_rn(0.0f)':
            codes = f'{x}'
        else:
            codes = f'__hsub2({x}, {y})'
    else:
        raise NotImplementedError(dtype)

    return wrap_return_codes(z, codes)


def mul(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':

        if x == '1.0f':
            codes = f'{y}'

        elif y == '1.0f':
            codes = f'{x}'

        else:
            codes = f'{x} * {y}'

    elif dtype == 'half2':

        if x == '__float2half2_rn(1.0f)':
            codes = f'{y}'

        elif y == '__float2half2_rn(1.0f)':
            codes = f'{x}'

        else:
            codes = f'__hmul2({x}, {y})'

    else:
        raise NotImplementedError(dtype)

    return wrap_return_codes(z, codes)


def div(z: Optional[str], x: str, y: str, dtype: str):
    if dtype == 'float':

        if y == '1.0f':
            codes = f'{x}'
        else:
            codes = f'{x} / {y}'
    elif dtype == 'half2':
        if y == '__float2half2_rn(1.0f)':
            codes = f'{x}'
        else:
            codes = f'__h2div({x}, {y})'
    else:
        raise NotImplementedError(dtype)

    return wrap_return_codes(z, codes)


def neg(y: Optional[str], x: str, dtype: str):
    if dtype == 'float':
        codes = f'- {x}'
    elif dtype == 'half2':
        codes = f'__hneg2({x})'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(y, codes)


def heaviside(y: Optional[str], x: str, dtype: str):
    if dtype == 'float':
        codes = f'{x} >= 0.0f ? 1.0f: 0.0f'
    elif dtype == 'half2':
        codes = f'__hgeu2({x}, __float2half2_rn(0.0f))'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(y, codes)


def exp(y: Optional[str], x: str, dtype: str):
    if dtype == 'float':
        codes = f'expf({x})'
    elif dtype == 'half2':
        codes = f'h2exp({x})'
    else:
        raise NotImplementedError(dtype)
    return wrap_return_codes(y, codes)


def sigmoid(y: Optional[str], x: str, alpha: float, dtype: str):
    alpha = constant(None, alpha, dtype)
    if dtype == 'float':
        codes = f'1.0f / (1.0f + expf(- {alpha} * {x}))'
    elif dtype == 'half2':
        codes = f'__h2div(__float2half2_rn(1.0f), __hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(__hmul2({alpha}, {x})))))'

    else:
        raise NotImplementedError(dtype)

    return wrap_return_codes(y, codes)


def sigmoid_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    codes = sigmoid(y=f'const {dtype} sigmoid_backward__sigmoid_ax', x=x, alpha=alpha, dtype=dtype) + '\n'
    alpha = constant(None, alpha, dtype)
    if dtype == 'float':
        codes += f'{y} = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * {alpha};'
        return codes
    elif dtype == 'half2':
        codes += f'{y} = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_backward__sigmoid_ax), sigmoid_backward__sigmoid_ax), {alpha});'
        return codes
    else:
        raise NotImplementedError(dtype)

def atan_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    alpha = constant(None, alpha, dtype)
    if dtype == 'float':
        codes = f'const float atan_backward__alpha_x = ((float) 1.57079632679489661923) * {alpha} * {x};'
        codes += f'{y} = {alpha} / 2.0f / (1.0f + atan_backward__alpha_x * atan_backward__alpha_x);'
        return codes

    elif dtype == 'half2':
        codes = f'const half2 atan_backward__alpha_x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), {alpha}), {x});'
        codes += f'{y} = __h2div({alpha}, __hmul2(__float2half2_rn(2.0f), __hfma2(atan_backward__alpha_x, atan_backward__alpha_x, __float2half2_rn(1.0f))));'
        return codes

    else:
        raise NotImplementedError(dtype)

def piecewise_leaky_relu_backward(y: str, x: str, w: float, c: float, dtype: str):
    assert y is not None
    w_inv = constant(None, 1. / w, dtype)
    w = constant(None, w, dtype)
    c = constant(None, c, dtype)

    codes = greater_equal(z=f'const {dtype} piecewise_leaky_relu_backward__mask', x=w, y=abs(y=None, x=x, dtype=dtype), dtype=dtype)

    codes += if_else(z=y, x=w_inv, y=c, mask=f'piecewise_leaky_relu_backward__mask', dtype=dtype)

    return codes


def s2nn_backward(y: str, x: str, alpha: float, beta: float, dtype: str):
    assert y is not None
    codes = sigmoid_backward(y=f'const {dtype} s2nn_backward__sgax', x=x, alpha=alpha, dtype=dtype)
    codes += greater_than(z=f'const {dtype} s2nn_backward__mask', x=constant(None, 0., dtype), y=x, dtype=dtype)

    codes += if_else(z=y, x=f's2nn_backward__sgax', y=div(z=None, x=constant(None, beta, dtype), y=add(z=None, x=x, y=constant(None, 1., dtype), dtype=dtype), dtype=dtype), mask=f's2nn_backward__mask', dtype=dtype)
    return codes


def q_pseudo_spike_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    alpha = constant(None, alpha, dtype)
    if dtype == 'float':
        return f'{y} = __powf(2.0f * fabsf({x}) / ({alpha} - 1.0f) + 1.0f, - {alpha});'
    elif dtype == 'half2':
        return power(z=y, x=f'__hadd2(__h2div(__hmul2(__float2half2_rn(2.0f), __habs2({x})), __hsub2({alpha}, __float2half2_rn(1.0f))), __float2half2_rn(1.0f))', y=f'__hneg2({alpha})', dtype=dtype)

def leaky_k_relu_backward(y: str, x: str, leak: float, k: float, dtype: str):
    assert y is not None
    leak = constant(None, leak, dtype)
    k = constant(None, k, dtype)
    codes = greater_equal(z=f'const {dtype} leaky_k_relu_backward__mask', x=x, y=constant(None, 0., dtype), dtype=dtype)
    codes += if_else(z=y, x=k, y=leak, mask=f'leaky_k_relu_backward__mask', dtype=dtype)
    return codes


def fake_numerical_gradient_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    alpha = constant(None, alpha, dtype)
    codes = greater_equal(z=f'{dtype} fake_numerical_gradient_backward__mask', x=x, y=constant(None, 0., dtype), dtype=dtype)
    codes += mul(z='fake_numerical_gradient_backward__mask', x='fake_numerical_gradient_backward__mask', y=constant(None, 2., dtype), dtype=dtype)
    codes += sub(z='fake_numerical_gradient_backward__mask', x='fake_numerical_gradient_backward__mask', y=constant(None, 1., dtype), dtype=dtype)
    codes += div(z='fake_numerical_gradient_backward__mask', x='fake_numerical_gradient_backward__mask', y=x, dtype=dtype)
    codes += minimal(z=y, x='fake_numerical_gradient_backward__mask', y=alpha, dtype=dtype)
    return codes


def log_tailed_relu_backward(y: str, x: str, alpha: float, dtype: str):
    alpha = constant(None, alpha, dtype)
    codes = greater_equal(z=f'const {dtype} log_tailed_relu_backward__mask_le0', x=constant(None, 0., dtype), y=x, dtype=dtype)
    codes += greater_than(z=f'const {dtype} log_tailed_relu_backward__mask_gt1', x=x, y=constant(None, 1, dtype), dtype=dtype)
    codes += if_else_else(w=y, x=alpha, y=div(z=None, x=constant(None, 1., dtype), y=x, dtype=dtype), z=constant(None, 1., dtype), mask_x=f'const {dtype} log_tailed_relu_backward__mask_le0', mask_y=f'const {dtype} log_tailed_relu_backward__mask_gt1', dtype=dtype)
    return codes