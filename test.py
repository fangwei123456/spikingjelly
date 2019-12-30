def f1(a, b, c):
    print(a,b,c)

def f2(d, f, *args):
    print(d)
    f1(d, *args)


f2(1, f1, 2,3)