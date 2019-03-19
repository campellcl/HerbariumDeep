import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

num_unique_initializers = 3
num_unique_optimizers = 2

gs = gridspec.GridSpec(num_unique_initializers, num_unique_initializers)
ax1 = plt.subplot(gs[2, 0])
# Hide major tick labels:
ax1.set_xticks(np.arange(0, num_unique_initializers+1, 1), minor=False)
ax1.set_xticklabels('', minor=False)

# Customize minor tick labels:
ax1.set_xticks(np.arange(0, num_unique_initializers, 0.5), minor=True)
ax1.set_xticklabels(np.arange(0, num_unique_initializers, 0.5), minor=True)
ax1.set_xticklabels(['', 'HE_NORMAL', '', 'HE_UNIFORM', '', 'NORM_TRUNC', ''], minor=True)
ax1.xaxis.tick_bottom()



# ax1.xaxis.tick_top()
# ax1 = plt.subplot(gs[0, 0])
# ax2 = plt.subplot(gs[0, 1])
# ax2.yaxis.tick_right()
#
# ax3 = plt.subplot(gs[1, 0])
# ax4 = plt.subplot(gs[1, 1])
# ax4.yaxis.tick_right()
# ax = plt.subplot2grid((2, 2), (0, 0))

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

