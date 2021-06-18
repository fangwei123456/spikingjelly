def add_comma(x: str):
    if x == '':
        return x
    else:
        return ', ' + x

def bind_fun(fun_names: list):
    ret = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
    for fname in fun_names:
        ret += fname
        ret += '\n'

    ret += '}\n'
    return ret

def add_include(headers: list):
    ret = ''
    for header in headers:
        ret += f'#include <{header}>\n'
    return ret


def write_str_to_file(fname: str, codes: str, append=False):
    if append:
        mode = 'a+'
    else:
        mode = 'w+'
    with open(fname, mode) as txt_file:
        txt_file.write(codes)




