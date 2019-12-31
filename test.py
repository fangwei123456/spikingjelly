
class A:
    def __init__(self, f, *args):
        self.f = f
        self.args = args
    def g(self, a, b):
        self.f(a, b, *self.args)

def fun(a,b,c,d,e):
    print(a,b,c,d,e)

aa = A(fun,9,8,7)
aa.g(2,4)