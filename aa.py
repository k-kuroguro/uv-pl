def f(a, b, c, **kwargs):
    print(a, b, c, kwargs)


f(a=4, b=5, c=6, e=1)  # 4 5 6 {'e': 1}
