import sympy as sym
sym.init_printing(use_latex='mathjax')

#  If you were not in a notebook environment,
#  but working within a terminal, use:
#
#  sym.init_printing(use_unicode=True)

phi, x = sym.symbols('\phi, x')

#  x here is a sympy symbol, and we form a list:
[ phi, x ]

sym.diff('sqrt(phi)')

sym.factor( phi**3 - phi**2 + phi - 1 )

((phi+1)*(phi-4)).expand()

x = sym.expand('(t+1)*2')
x

x, y, z = sym.symbols('x y z')

H = sym.Matrix([sym.sqrt(x**2 + z**2)])

state = sym.Matrix([x, y, z])

H.jacobian(state)

dt = sym.symbols('\Delta{t}')

F_k = sym.Matrix([[1, dt, dt**2/2],
                  [0,  1,      dt],
                  [0,  0,      1]])

Q = sym.Matrix([[0,0,0],
                [0,0,0],
                [0,0,1]])

sym.integrate(F_k*Q*F_k.T,(dt, 0, dt))

x = sym.symbols('x')

w = (x**2) - (3*x) + 4
w.subs(x, 4)

LHS = (x**2) - (8*x) + 15
RHS = 0
#  where both RHS and LHS can be complicated expressions.

solved = sym.solveset( sym.Eq(LHS, RHS), x, domain=sym.S.Reals )
#  Notice how the domain solution can be specified.

solved
#  A set of solution(s) is returned.

#  Testing whether any solution(s) were found:
if solved != sym.S.EmptySet:
    print("Solution set was not empty.")

#  sympy sets are not like the usual Python sets...
type(solved)

#  ... but can easily to converted to a Python list:
l = list(solved)
print( l, type(l) )

LHS = (x**2)
RHS = -4
#  where both RHS and LHS can be complicated expressions.

solved = sym.solveset( sym.Eq(LHS, RHS), x )
#  Leaving out the domain will include the complex domain.

solved

