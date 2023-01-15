import numpy as np
import matplotlib.pyplot as plt
from itertools import product

max_steps = 20000
d_model = 512
steps = np.arange(max_steps)


def generate_lr_schedule(warmup_steps=4000, d_model=512):
  linear_lr = steps * warmup_steps ** -1.5
  sqrt_lr = steps ** -0.5
  label = f'warmup{warmup_steps}-d_model{d_model}'
  return d_model ** -0.5 * np.minimum(linear_lr, sqrt_lr), label


for warmup_steps, d_model in product([4000, 8000], [512, 1024]):
  lr, label = generate_lr_schedule(warmup_steps, d_model)
  plt.plot(steps, lr, label=label)
plt.xlabel('steps')
plt.ylabel('learning rate')
plt.title('Transformer lr schedule')
plt.legend()
plt.show()
