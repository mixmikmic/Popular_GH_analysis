import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot') # emulate pretty r-style plots

def power():
    """ prints out 2**3 """
    print(2**3)
    
power()

def power2(x,a):
    """ prints x to the power of a:float """
    print(x**a)
    

power2(3,8)
power2(10,3)
power2(8,17)
power2(131,3)

def power3(x,a):
    """ returns x raised to float a """
    return(x**a)

# call power3 with exponent=2 on an array
x = np.arange(1,10)
y = power3(x,2)

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize=(16,4))

# Plot x vs y
ax1.plot(x,y,linestyle='-.', marker='o')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Plot log(x) vs y
ax2.semilogx(x,y, linestyle='-.', marker='o')
ax2.set_xlabel('log(x)')
ax2.set_ylabel('y')

# Plot x vs log(y)
ax3.semilogy(x,y, linestyle='-.', marker='o')
ax3.set_xlabel('x')
ax3.set_ylabel('log(y)')

# Plot log log
ax4.loglog(x,y, linestyle='-.', marker='o')
ax4.set_xlabel('log(x)')
ax4.set_ylabel('log(y)')

plt.tight_layout()

def plot_power(x,a):
    """Plots x vs x**a """
    # generate dependent
    y = x**a
    
    # create plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x,y, linestyle = '-.', marker = 'o')
    ax.set_xlabel('x',fontsize=16)
    ax.set_ylabel('x**'+str(a),fontsize=16)

plot_power(np.arange(1,11),3)

