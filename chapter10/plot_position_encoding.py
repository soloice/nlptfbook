import numpy as np
import matplotlib.pyplot as plt

max_len, d_model = 500, 256
half_dim = d_model // 2
pe = np.zeros((d_model, max_len))  # [D, T]
pos = np.arange(max_len)  # [T]
freq = 10000 ** (2 * np.arange(half_dim) / d_model)  # [D//2]
pos_freq = pos.reshape((1, -1)) / freq.reshape((-1, 1))  # [D//2, T]
pe[:d_model // 2, :] = np.sin(pos_freq)
pe[d_model // 2:, :] = np.cos(pos_freq)

plt.imshow(pe, cmap='rainbow')
plt.title("Position encoding")
plt.colorbar()
plt.ylabel("model dim")
plt.xlabel("time steps")
plt.show()
