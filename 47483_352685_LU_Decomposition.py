get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, multiply, dot
from scipy.linalg import lu

def solve_l(m, y):  # solves x from m*x = y
    assert (m==tril(m)).all()        # assert matrix is lower diagonal
    assert (m.shape[0]==m.shape[1])  # Assert matrix is square matrix
    N=m.shape[0]
    x=zeros(N)                      # Vector of roots
    for r in range(N):
        s = 0
        for c in range(r):
            s += m[r,c]*x[c]            
        x[r] = (y[r]-s) / m[r,r]
    assert allclose(dot(m,x), y)    # Check solution
    return x

def solve_u(m, y):
    m2 = fliplr(flipud(m))     # flip matrix LR and UD, so upper diagonal matrix becomes lower diagonal
    y2 = y[::-1]               # flip array
    x2 = solve(m2, y2)
    x = x2[::-1]
    assert allclose(dot(m,x), y) # Check solution
    return x

def solve(m, y):
    if (m==tril(m)).all():
        return solve_l(m,y)
    else:
        return solve_u(m,y)

# Unknowns
x_org = array([2, 4, 1])
print(x_org)

# Coefficients
m = array([[2,-1,1],[3,3,9],[3,3,5]])
print(m)

# Results
y = dot(m,x_org)
print(y)

# Note: matrix dot-product is not commutative, but is associative
p, l, u = lu(m, permute_l=False)
pl, u = lu(m, permute_l=True)
assert (dot(p,l)==pl).all()
assert (dot(pl,u)==m).all()
assert (pinv(p)==p).all()

print(l) # Lower diagonal matrix, zero element above the principal diagonal

print(u) # Upper diagnonal matrix, zero elements below the principal diagonal

print(p) # Permutation matrix for "l"

assert (l*u==multiply(l,u)).all()          # memberwise multiplication
assert (m==dot(dot(p,l),u)).all()          # matrix multiplication, M=LU

assert (pinv(p)==p).all()
#   P*L*U*X = Y
#   L*U*X = pinv(P)*Y
#   set Z=U*X
#   L*Z = P*Y (solve Z)
z = solve(l, dot(p,y))
#   solve X from U*X=Z
x = solve(u, z)

assert allclose(x_org,x)
print(x)



