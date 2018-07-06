import math

def func(x, x0):
    return x**2 - x0

def deviv_func(x):
    return 2*x

def new_ton_raphson(x0, n):
    x = x0/2
    for i in range(n):
        h = func(x, x0)/deviv_func(x)
        x = x - h
    return x
n = 4
x = 5
print('Newton Raspson: ', new_ton_raphson(x, n))
print('Root: ', math.sqrt(x))
print('Error: ', abs(math.sqrt(x) - new_ton_raphson(x, n)))