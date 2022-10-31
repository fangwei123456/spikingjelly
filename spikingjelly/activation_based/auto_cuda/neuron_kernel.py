from typing import Callable
import base, cfunction


def neuronal_hard_reset(v_next: str, h: str, spike: str, v_reset: str, dtype: str = 'float'):
    if dtype == 'float':
        return f'{v_next} = {h} * (1.0f - {spike}) + {v_reset} * {spike};'
    elif dtype == 'half2':
        return f'{v_next} = __hfma2({h}, __hsub2(__float2half2_rn(1.0f), {spike}), __hmul2(v_reset, spike));'
    else:
        raise NotImplementedError(dtype)

def neuronal_soft_reset(v_next: str, h: str, spike: str, v_th: str, dtype: str = 'float'):
    if dtype == 'float':
        return f'{v_next} = {h} - {v_th} * {spike};'
    elif dtype == 'half2':
        return f'{v_next} = __hsub2({h}, __hmul2({v_th}, {spike}));'
    else:
        raise NotImplementedError(dtype)

def neuronal_fire(spike: str, v: str, v_th: str, dtype: str = 'float'):
    if dtype == 'float':
        return cfunction.heaviside(y=spike, x=f'{v} - {v_th}', dtype=dtype)
    elif dtype == 'half2':
        return cfunction.heaviside(y=spike, x=f'__hsub2({v}, {v_th})', dtype=dtype)
    else:
        raise NotImplementedError(dtype)


class NeuronFPTT(base.CKernel2D):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(kernel_name=f'{self.__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}', reverse=False)
        self.hard_reset = hard_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='x_seq')
        self.add_param(ctype=f'{dtype} *', cname='v_v_seq')
        self.add_param(ctype=f'{dtype} *', cname='h_seq')
        self.add_param(ctype=f'{dtype} *', cname='spike_seq')
        self.add_param(ctype=f'{dtype} &', cname='v_th')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

    @property
    def neuronal_charge(self) -> str:
        # e.g., for IFNode, this function shoule return:
        #   cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=dtype)
        raise NotImplementedError


    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(self.neuronal_charge)

        core_codes.append(neuronal_fire(spike='spike[t]', v='v_v_seq[t]', v_th='v_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(neuronal_hard_reset(v_next='v_v_seq[t]', h='h_seq[t]', spike='spike_seq[t]', v_reset='v_reset', dtype=self.dtype))
        else:
            core_codes.append(neuronal_soft_reset(v_next='v_v_seq[t]', h='h_seq[t]', spike='spike_seq[t]', v_th='v_th', dtype=self.dtype))

        self._core = core_codes.codes
        return self._core



class NeuronBPTT(base.CKernel2D):
    def __init__(self, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(kernel_name=f'{__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}_{"detach_reset" if detach_reset else "nodetach_reset"}', reverse=True)
        self.surrogate_function = surrogate_function
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='grad_spike_seq')
        self.add_param(ctype=f'const {dtype} *', cname='grad_v_seq')
        self.add_param(ctype=f'const {dtype} *', cname='h_seq')
        self.add_param(ctype=f'const {dtype} *', cname='spike_seq')
        self.add_param(ctype=f'{dtype} *', cname='grad_x_seq')
        self.add_param(ctype=f'{dtype} *', cname='grad_v_init')
        self.add_param(ctype=f'{dtype} &', cname='v_th')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

        codes = base.CodeTyper(16)
        if dtype == 'float':
            codes.append('float grad_h = 0.0f;')
        elif dtype == 'half2':
            codes.append(cfunction.float2half2(y='half2 grad_h', x='0.0f'))
        else:
            raise NotImplementedError(dtype)

        self.pre_core = codes.codes

        codes = base.CodeTyper(16)
        codes.append(cfunction.mul(z='grad_v_init[index]', x='grad_h', y=self.grad_h_to_x, dtype=self.dtype))
        self.post_core = codes.codes


    @property
    def grad_h_next_to_v(self) -> str:
        raise NotImplementedError

    @property
    def grad_h_to_x(self) -> str:
        raise NotImplementedError


    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(cfunction.sub(z=f'const {self.dtype} over_th', x='h_seq[t]', y='v_th', dtype=self.dtype))

        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_s_to_h', x='over_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(cfunction.sub(z=f'{self.dtype} grad_v_to_h', x=cfunction.constant(x=1., dtype=self.dtype), y='spike_seq[t]', dtype=self.dtype))

            if not self.detach_reset:
                core_codes.append(cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='h_seq[t]', dtype=self.dtype))
                core_codes.append(cfunction.mul(z=f'temp_var', x='temp_var', y='grad_s_to_h', dtype=self.dtype))
                core_codes.append(cfunction.add(z=f'grad_v_to_h', x='temp_var', y='grad_v_to_h', dtype=self.dtype))


        else:
            core_codes.append(f'{self.dtype} grad_v_to_h = {cfunction.constant(1., dtype=self.dtype)}')

            if not self.detach_reset:
                core_codes.append(cfunction.mul(z=f'{self.dtype} temp_var', x='v_th', y='grad_s_to_h', dtype=self.dtype))
                core_codes.append(cfunction.add(z=f'grad_v_to_h', x='temp_var', y='grad_v_to_h', dtype=self.dtype))


        core_codes.append(cfunction.mul(z=f'grad_h', x='grad_h', y=self.grad_h_next_to_v, dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x='grad_v_seq[t]', y='grad_h', dtype=self.dtype))
        core_codes.append(cfunction.mul(z='grad_h', x='grad_h', y='grad_v_to_h', dtype=self.dtype))
        core_codes.append(cfunction.mul(z=f'{self.dtype} temp_var', x='grad_spike_seq[t]', y='grad_s_to_h', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='temp_var', dtype=self.dtype))

        core_codes.append(cfunction.mul(z='grad_x_seq[t]', x='grad_h', y=self.grad_h_to_x, dtype=self.dtype))

        self._core = core_codes.codes
        return self._core





class IFNodeFPTT(NeuronFPTT):
    @property
    def neuronal_charge(self) -> str:
        return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

class IFNodeBPTT(NeuronBPTT):
    @property
    def grad_h_next_to_v(self) -> str:
        return cfunction.constant(x=1., dtype=self.dtype)

    @property
    def grad_h_to_x(self) -> str:
        return cfunction.constant(x=1., dtype=self.dtype)



if __name__ == '__main__':
    def sigmoid_backward(y: str, x: str, dtype: str):
        return cfunction.sigmoid_backward(y, x, 2., dtype)

    print(IFNodeBPTT(surrogate_function=sigmoid_backward, hard_reset=True, detach_reset=False, dtype='half2').full_codes)
