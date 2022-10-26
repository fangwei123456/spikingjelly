
def complete_eq(x: str):
    if x is None:
        return ''
    else:
        return f'{x} ='
def float2half2(y: str or None, x: str):
    y = complete_eq(y)
    return f'{y} __float2half2_rn({x});'


def constant(x: float, dtype: str):
    if dtype == 'float':
        return f'{x}f'
    elif dtype == 'half2':
        return f'__float2half2_rn({x}f)'
    else:
        raise NotImplementedError(dtype)


def abs(y: str or None, x: str, dtype: str):
    y = complete_eq(y)

    if dtype == 'float':
        return f'{y} __fabsf({x});'
    elif dtype == 'half2':
        return f'{y} __habs2({x});'
    else:
        raise NotImplementedError(dtype)

def power(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':
        return f'{z} __powf({x, y});'
    elif dtype == 'half2':
        # CUDA FP16 does not provide powf function. We use z = 2 ** (log2(x) * y)
        return f'{z} __h2exp(__hmul2(h2log2({x}), y));'
    else:
        raise NotImplementedError(dtype)

def if_else(z: str or None, x: str, y: str, mask: str, dtype: str):
    # z = x * mask + y * (1. - mask)
    z = complete_eq(z)
    if dtype == 'float':
        return f'{z} {x} * {mask} + {y} * (1.0f - {mask});'
    elif dtype == 'half2':
        return f'{z} __hfma2({x}, {mask}, __hmul2({y}, __hsub2(__float2half2_rn(1.0f), {mask})));'
    else:
        raise NotImplementedError(dtype)

def if_else_else(w: str or None, x: str, y: str, z: str, mask_x: str, mask_y: str, dtype: str):
    # w = mask_x * x + mask_y * y + (1. - mask_x * mask_y) * z
    w = complete_eq(w)
    if dtype == 'float':
        return f'{w} {mask_x} * {x} + {mask_y} * {y} + (1. - {mask_x} * {mask_y}) * {z};'
    else:
        return f'{w} __hadd2(__hadd2(__hmul2({mask_x}, {x}), __hmul2({mask_y}, {y})), __hmul2({z}, __hsub2(__float2half_rn(1.0f), __hmul2({mask_x}, {mask_y}))));'


def greater_equal(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':
        return f'{z} (float) ({x} >= {y});'
    elif dtype == 'half2':
        return f'{z} hgeu2({x}, {y});'
    else:
        raise NotImplementedError(dtype)

def greater_than(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':
        return f'{z} (float) ({x} > {y});'
    elif dtype == 'half2':
        return f'{z} hgtu2({x}, {y});'
    else:
        raise NotImplementedError(dtype)

def minimal(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':
        return f'{z} min({x}, {y});'
    elif dtype == 'half2':
        return f'{z} __hmin2({x}, {y});'
    else:
        raise NotImplementedError(dtype)

def maximum(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':
        return f'{z} max({x}, {y});'
    elif dtype == 'half2':
        return f'{z} __hmax2({x}, {y});'
    else:
        raise NotImplementedError(dtype)

def add(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':

        if x == '0.0f':
            return f'{z} {y};'

        if y == '0.0f':
            return f'{z} {x};'

        return f'{z} {x} + {y};'

    elif dtype == 'half2':
        if x == '__float2half2_rn(0.0f)':
            return f'{z} {y};'

        if y == '__float2half2_rn(0.0f)':
            return f'{z} {x};'

        return f'{z} __hadd2({x}, {y});'
    else:
        raise NotImplementedError(dtype)


def sub(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':

        if y == '0.0f':
            return f'{z} {x};'

        return f'{z} {x} - {y};'

    elif dtype == 'half2':

        if y == '__float2half2_rn(0.0f)':
            return f'{z} {x};'

        return f'{z} __hsub2({x}, {y});'
    else:
        raise NotImplementedError(dtype)


def mul(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':

        if x == '1.0f':
            return f'{z} {y};'

        if y == '1.0f':
            return f'{z} {x};'


        return f'{z} {x} * {y};'

    elif dtype == 'half2':

        if x == '__float2half2_rn(1.0f)':
            return f'{z} {y};'

        if y == '__float2half2_rn(1.0f)':
            return f'{z} {x};'


        return f'{z} __hmul2({x}, {y});'

    else:
        raise NotImplementedError(dtype)


def div(z: str or None, x: str, y: str, dtype: str):
    z = complete_eq(z)
    if dtype == 'float':

        if y == '1.0f':
            return f'{z} {x};'

        return f'{z} {x} / {y};'
    elif dtype == 'half2':
        if y == '__float2half2_rn(1.0f)':
            return f'{z} {x};'

        return f'{z} __h2div({x}, {y});'
    else:
        raise NotImplementedError(dtype)


def neg(x: str, dtype: str):
    if dtype == 'float':
        return f'- {x}'
    elif dtype == 'half2':
        return f'__hneg2({x})'
    else:
        raise NotImplementedError(dtype)


def heaviside(y: str or None, x: str, dtype: str):
    y = complete_eq(y)
    if dtype == 'float':
        return f'{y} {x} >= 0.0f ? 1.0f: 0.0f;'
    elif dtype == 'half2':
        return f'{y} __hgeu2({x});'
    else:
        raise NotImplementedError(dtype)


def exp(y: str or None, x: str, dtype: str):
    y = complete_eq(y)
    if dtype == 'float':
        return f'{y} expf({x});'
    elif dtype == 'half2':
        return f'{y} h2exp({x})'
    else:
        raise NotImplementedError(dtype)


def sigmoid(y: str or None, x: str, alpha: float, dtype: str):
    y = complete_eq(y)
    alpha = constant(alpha, dtype)
    if dtype == 'float':
        return f'{y} 1.0f / (1.0f + expf(- {alpha} * {x}));'
    elif dtype == 'half2':
        return f'{y} __h2div(__float2half2_rn(1.0f), __hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(__hmul2({alpha}, {x}))));'

    else:
        raise NotImplementedError(dtype)


def sigmoid_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    codes = sigmoid(y=f'const {dtype} sigmoid_backward__sigmoid_ax', x=x, alpha=alpha, dtype=dtype) + '\n'
    alpha = constant(alpha, dtype)
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
    alpha = constant(alpha, dtype)
    if dtype == 'float':
        codes = f'const float atan_backward__alpha_x = ((float) 1.57079632679489661923) * {alpha} * {x};'
        codes += f'{y} = {alpha} / 2.0f / (1.0f + atan_backward__alpha_x * atan_backward__alpha_x);'
        return codes

    elif dtype == 'half2':
        codes = f'const half2 atan_backward__alpha_x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), {alpha}), {x});'
        codes += f'{y} = __h2div({alpha}, __hmul2(__float2half2_rn(2.0f), __hfma2(atan_backward__alpha_x, atan_backward__alpha_x, __float2half2_rn(1.0f))));'

    else:
        raise NotImplementedError(dtype)

def piecewise_leaky_relu_backward(y: str, x: str, w: float, c: float, dtype: str):
    assert y is not None
    w_inv = constant(1. / w, dtype)
    w = constant(w, dtype)
    c = constant(c, dtype)

    codes = greater_equal(z=f'const {dtype} piecewise_leaky_relu_backward__mask', x=w, y=abs(y=None, x=x, dtype=dtype), dtype=dtype)

    codes += if_else(z=y, x=w_inv, y=c, mask=f'const {dtype} piecewise_leaky_relu_backward__mask', dtype=dtype)

    return codes


def s2nn_backward(y: str, x: str, alpha: float, beta: float, dtype: str):
    assert y is not None
    codes = sigmoid_backward(y=f'const {dtype} s2nn_backward__sgax', x=x, alpha=alpha, dtype=dtype)
    codes += greater_than(z=f'const {dtype} s2nn_backward__mask', x=constant(0., dtype), y=x, dtype=dtype)

    codes += if_else(z=y, x=f'const {dtype} s2nn_backward__sgax', y=mul(z=None, x=constant(beta, dtype), y=add(z=None, x=x, y=constant(1., dtype), dtype=dtype), dtype=dtype), mask=f'const {dtype} s2nn_backward__mask', dtype=dtype)
    return codes


def q_pseudo_spike_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    alpha = constant(alpha, dtype)
    if dtype == 'float':
        return f'{y} = __powf(2.0f * __fabsf({x}) / ({alpha} - 1.0f) + 1.0f, - {alpha});'
    elif dtype == 'half2':
        return power(z=y, x=f'__hadd2(__h2div(__hmul2(__float2half2_rn(2.0f), __habs2({x})), __hsub2({alpha}, __float2half2_rn(1.0f))), __float2half2_rn(1.0f))', y=f'__hneg2({alpha})', dtype=dtype)

def leaky_k_relu_backward(y: str, x: str, leak: float, k: float, dtype: str):
    assert y is not None
    leak = constant(leak, dtype)
    k = constant(k, dtype)
    codes = greater_equal(z=f'const {dtype} leaky_k_relu_backward__mask', x=x, y=constant(0., dtype), dtype=dtype)
    codes += if_else(z=y, x=k, y=leak, mask=f'const {dtype} leaky_k_relu_backward__mask', dtype=dtype)
    return codes


def fake_numerical_gradient_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    alpha = constant(alpha, dtype)
    codes = greater_equal(z=f'const {dtype} fake_numerical_gradient_backward__mask', x=x, y=constant(0., dtype), dtype=dtype)
    codes += mul(z='fake_numerical_gradient_backward__mask', x='fake_numerical_gradient_backward__mask', y=constant(2., dtype), dtype=dtype)
    codes += sub(z='fake_numerical_gradient_backward__mask', x='fake_numerical_gradient_backward__mask', y=constant(1., dtype), dtype=dtype)
    codes += div(z='fake_numerical_gradient_backward__mask', x='fake_numerical_gradient_backward__mask', y=x, dtype=dtype)
    codes += maximum(z=y, x='fake_numerical_gradient_backward__mask', y=alpha, dtype=dtype)
    return codes


def log_tailed_relu_backward(y: str, x: str, alpha: float, dtype: str):
    alpha = constant(alpha, dtype)
    codes = greater_equal(z=f'const {dtype} log_tailed_relu_backward__mask_le0', x=constant(0., dtype), y=x, dtype=dtype)
    codes += greater_than(z=f'const {dtype} log_tailed_relu_backward__mask_gt1', x=x, y=constant(1, dtype), dtype=dtype)
    codes += if_else_else(w=y, x=alpha, y=div(z=None, x=constant(1., dtype), y=x, dtype=dtype), z=constant(1., dtype), mask_x=f'const {dtype} log_tailed_relu_backward__mask_le0', mask_y=f'const {dtype} log_tailed_relu_backward__mask_gt1')
    return codes