from dolfin import *

import sympy as sp
from sympy.printing import ccode

def setup_mms(eps):    
    pi = sp.pi
    x, y = sp.symbols('x[0] x[1]')
    
    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])
    
    sp_Grad = lambda f: sp.Matrix([[f[0].diff(x, 1), f[0].diff(y, 1)],
                                   [f[1].diff(x, 1), f[1].diff(y, 1)]])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)
    
    sp_Div = lambda f: sp.Matrix([sp_div(f[0, :]), sp_div(f[1, :])])

    u = sp.Matrix([sp.cos(pi*y)*sp.sin(pi*x), -sp.cos(pi*x)*sp.sin(pi*y)])

    p = pi*sp.cos(pi*x)*sp.cos(pi*y)

    X = -sp_Grad(u)
    I = sp.eye(2)
    
    lambda_ = ((X - p*I)*sp.Matrix([-1, 0])).subs(x, 0)

    h = -((X - p*I)*sp.Matrix([1, 0])).subs(x, 1)
    f = -sp_Div(sp_Grad(u)) + u - sp_grad(p)
    u0 = u
    up = map(as_expression, (u, p, lambda_))
    fg = map(as_expression, (f, h, u0))
    
    return up, fg

def expr_body(expr, **kwargs):
    if not hasattr(expr, '__len__'):
        # Defined in terms of some coordinates
        xyz = set(sp.symbols('x[0], x[1], x[2]'))
        xyz_used = xyz & expr.free_symbols
        assert xyz_used <= xyz
        # Expression params which need default values
        params = (expr.free_symbols - xyz_used) & set(kwargs.keys())
        # Body
        expr = ccode(expr).replace('M_PI', 'pi')
        # Default to zero
        kwargs.update(dict((str(p), 0.) for p in params))
        # Convert
        return expr
    # Vectors, Matrices as iterables of expressions
    else:
        return [expr_body(e, **kwargs) for e in expr]


def as_expression(expr, degree=4, **kwargs):
    '''Turns sympy expressions to Dolfin expressions.'''
    return Expression(expr_body(expr), degree=degree, **kwargs)

