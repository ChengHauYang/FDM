import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size':14,'font.family':'serif'})
matplotlib.rcParams.update({'text.usetex':'true'})

# Define the axis box
axisbox = [-5, 5, -5, 5]
xa, xb, ya, yb = axisbox
nptsx = 501
nptsy = 501

# Create a 2D grid of points
x = np.linspace(xa, xb, nptsx)
y = np.linspace(ya, yb, nptsy)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Define the 4th order RK
RK4 = lambda z: (1 + z + 1/2 * z**2 + 1/6 * z**3 + 1/24 * z**4)

# Evaluate R(z)
Rval4 = RK4(Z)

# Evaluate |R(z)|
Rabs4 = abs(Rval4)

# Plot the contour plot with a single color
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([xa, xb], [0, 0], 'k-', linewidth=2)
ax.plot([0, 0], [ya, yb], 'k-', linewidth=2)

Z = Rabs4
Z[Z > 1] = np.nan
ax.contourf(X, Y, Z, colors='blue')

ax.set_xlabel('Re(z)', fontsize=15)
ax.set_ylabel('Im(z)', fontsize=15)
ax.set_title('Region of absolute stability for RK4', fontsize=15)
ax.axis(axisbox)
ax.tick_params(axis='both', labelsize=15)
ax.grid(True, which='both', linewidth=1.1, color=[0.87, 0.87, 0.87], linestyle='-', alpha=0.5)
ytick_locs = range(-5,6)
ax.set_yticks(ytick_locs)
xtick_locs = range(-5,6)
ax.set_xticks(xtick_locs)



#plt.show()
plt.savefig('RAS.pdf',format='pdf',bbox_inches='tight')

