import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

num_unique_initializers = 3
num_unique_optimizers = 2

gs0 = gridspec.GridSpec(2, 3)

ax1 = plt.subplot(gs0[0, 0])
# ax6.set_xticks([0, 1, 2], minor=False)
# ax6.set_xticklabels(['', 'HE_NORMAL', ''])
ax1.set_xticklabels('')
ax1.set_yticks([0, 1, 2], minor=False)
ax1.set_yticklabels(['', 'NESTEROV', ''])

ax4 = plt.subplot(gs0[1, 0])
ax4.set_xticks([0, 1, 2], minor=False)
ax4.set_xticklabels(['', 'HE_NORMAL', ''])
ax4.set_yticks([0, 1, 2], minor=False)
ax4.set_yticklabels(['', 'ADAM', ''])

ax5 = plt.subplot(gs0[1, 1])
ax5.set_xticks([0, 1, 2], minor=False)
ax5.set_xticklabels(['', 'HE_UNIFORM', ''])

ax6 = plt.subplot(gs0[1, 2])
ax6.set_xticks([0, 1, 2], minor=False)
ax6.set_xticklabels(['', 'NORMAL_TRUNC', ''])

# Create inner layer:
gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0])
ax01 = plt.subplot(gs00[0, 0])
ax01.set_yticks([0, 1, 2], minor=False)
ax01.set_yticklabels(['NESTEROV', '', ''])
ax01.set_xticks([0, 1, 2, 3], minor=False)
ax01.set_xticklabels(['', 'Adam', 'Nesterov', ''])
ax01.xaxis.tick_top()

ax02 = plt.subplot(gs00[0, 1])
ax03 = plt.subplot(gs00[1, 0])
ax03.set_yticks([0, 1, 2], minor=False)
ax03.set_yticklabels('')
ax04 = plt.subplot(gs00[1, 1])

# Create inner-inner layer:
gs000 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs00[0])
ax001 = plt.subplot(gs000[0, 0])
ax001_bot = ax001.twiny()
ax001.set_xticks([0, 1, 2], minor=False)
ax001.set_xticklabels(['', '', 'ELU'])
ax001.xaxis.tick_top()
ax001.set_yticks([0, 1, 2], minor=False)
ax001.set_yticklabels('')
ax001_figure_position = ax001.get_position()
color_box = patches.Rectangle((0, 0), 1, 2, facecolor='red')
ax001.add_patch(color_box)
# ax001.patch.set_facecolor('green')
# ax001.patch.set_alpha(0.1)

ax001_bot.set_xlim(ax001.get_xlim())
# ax001_bot.set_xticks(ax001.get_xticks())
ax001_bot.set_xticks([0, 1, 2, 3, 4], minor=True)
ax001_bot.set_xticklabels(['', 'TB=10', '', 'TB=20', ''], minor=True)
ax001_bot.set_xticklabels('', minor=False)
ax001_bot.set_yticklabels('', minor=False)
ax001_bot.xaxis.tick_bottom()


ax002 = plt.subplot(gs000[0, 1])
ax002_bot = ax002.twiny()
ax002.set_xticks([0, 1, 2], minor=False)
ax002.set_xticklabels('', minor=False)
ax002_bot.set_xticks([0, 1, 2], minor=False)
ax002_bot.set_xticklabels('', minor=False)
ax002_bot.set_yticklabels('', minor=False)

# ax003 = plt.subplot(gs000[1, 0])
# ax003.set_yticks([0, 1, 2], minor=False)
# ax003.set_yticklabels(['NESTEROV', '', ''])


# ax1 = plt.subplot(gs[2, 0])
# # Hide major tick labels:
# ax1.set_xticks(np.arange(0, num_unique_initializers+1, 1), minor=False)
# ax1.set_xticklabels('', minor=False)
#
# # Customize minor tick labels:
# ax1.set_xticks(np.arange(0, num_unique_initializers, 0.5), minor=True)
# ax1.set_xticklabels(np.arange(0, num_unique_initializers, 0.5), minor=True)
# ax1.set_xticklabels(['', 'HE_NORMAL', '', 'HE_UNIFORM', '', 'NORM_TRUNC', ''], minor=True)
# ax1.xaxis.tick_bottom()



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

