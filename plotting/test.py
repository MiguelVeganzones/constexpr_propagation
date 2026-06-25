import style
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,100)

plt.plot(x, np.sin(x), label="data")

plt.xlabel("x")
plt.ylabel("Entries")
plt.legend()

plt.tight_layout()
plt.show()
