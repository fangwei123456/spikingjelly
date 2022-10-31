import base
def float2half2(y: str, x: str):
    return f'{y} = __float2half2_rn({x});'

def constant(x: float, dtype: str):
    if dtype == 'float':
        return f'{x}f'
    elif dtype == 'half2':
        return f'__float2half2_rn({x}f)'
    else:
        raise NotImplementedError(dtype)


def add(z: str, x: str, y: str, dtype: str):
    if dtype == 'float':
        return f'{z} = {x} + {y};'
    elif dtype == 'half2':
        return f'{z} = __hadd2({x}, {y});'
    else:
        raise NotImplementedError(dtype)

def sub(z: str, x: str, y: str, dtype: str):
    if dtype == 'float':
        return f'{z} = {x} - {y};'
    elif dtype == 'half2':
        return f'{z} = __hsub2({x}, {y});'
    else:
        raise NotImplementedError(dtype)


def mul(z: str, x: str, y: str, dtype: str):
    if dtype == 'float':
        return f'{z} = {x} * {y};'
    elif dtype == 'half2':
        return f'{z} = __hmul2({x}, {y});'
    else:
        raise NotImplementedError(dtype)

def div(z: str, x: str, y: str, dtype: str):
    if dtype == 'float':
        return f'{z} = {x} / {y};'
    elif dtype == 'half2':
        return f'{z} = __h2div({x}, {y});'
    else:
        raise NotImplementedError(dtype)

def neg(x: str, dtype: str):
    if dtype == 'float':
        return f'- {x}'
    elif dtype == 'half2':
        return f'__hneg2({x})'
    else:
        raise NotImplementedError(dtype)


def heaviside(y: str, x: str, dtype: str):
    if dtype == 'float':
        return f'{y} = {x} >= 0.0f ? 1.0f: 0.0f;'
    elif dtype == 'half2':
        return f'{y} = __hgeu2({x});'
    else:
        raise NotImplementedError(dtype)

def exp(y: str, x: str, dtype: str):
    if dtype == 'float':
        return f'{y} = expf({x});'
    elif dtype == 'half2':
        return f'{y} = h2exp({x})'
    else:
        raise NotImplementedError(dtype)

def sigmoid(y: str, x: str, alpha: float, dtype: str):
    alpha = constant(alpha, dtype)

    if dtype == 'float':
        return f'{y} = 1.0f / (1.0f + expf(- {alpha} * {x}));'
    elif dtype == 'half2':
        return f'{y} = __h2div(__float2half2_rn(1.0f), __hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(__hmul2({alpha}, {x}))));'

    else:
        raise NotImplementedError(dtype)

def sigmoid_backward(y: str, x: str, alpha: float, dtype: str):
    codes = sigmoid(y='sigmoid_ax', x=x, alpha=alpha, dtype=dtype) + '\n'
    alpha = constant(alpha, dtype)
    if dtype == 'float':
        codes += f'{y} = (1.0f - sigmoid_ax) * sigmoid_ax * {alpha};'
        return codes
    elif dtype == 'half2':
        codes += f'{y} = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), {alpha});'
        return codes
    else:
        raise NotImplementedError(dtype)


