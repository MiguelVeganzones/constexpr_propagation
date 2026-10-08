import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl

# ROOT-ish style
mpl.rcParams.update({
    "figure.figsize": (7,5),
    "font.size": 16,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "legend.frameon": False,
})


# Example data
np.random.seed(1)

x = np.linspace(0,10,30)
y = 2.5*x + 3 + np.random.normal(0,3,len(x))


# Define fit function
def linear(x, a, b):
    return a*x + b


# Fit
params, covariance = curve_fit(linear, x, y)

a, b = params
errors = np.sqrt(np.diag(covariance))


# Plot data
plt.scatter(
    x,
    y,
    s=40,
    label="data"
)


# Plot fit
xfit = np.linspace(min(x), max(x), 200)

plt.plot(
    xfit,
    linear(xfit,a,b),
    linewidth=2,
    label=rf"fit: $y={a:.2f}x+{b:.2f}$"
)


plt.xlabel("x")
plt.ylabel("y")

plt.legend()

plt.tight_layout()
plt.show()
