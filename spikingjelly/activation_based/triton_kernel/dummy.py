class DummyTriton:

    def jit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func 