import numpy as np
import matplotlib.pyplot as plt
# heatmap_dims = (4, 6)
#
# x_tick_labels_bot_major = []
# x_tick_labels_bot_minor = []
# x_ticks_bot_major = np.arange(0, heatmap_dims[1], 1)
# x_ticks_bot_minor = np.arange(0, heatmap_dims[1], 0.5)
#
# x_tick_labels_top_major = []
# x_tick_labels_top_minor = []
# x_ticks_top_major = np.arange(0, heatmap_dims[1], 1)
# x_ticks_top_minor = np.arange(0, heatmap_dims[1], 0.5)
#
# y_tick_labels_left_major = []
# y_tick_labels_left_minor = []
# y_ticks_left_major = np.arange(0, heatmap_dims[0], 1)
# y_ticks_left_minor = np.arange(0, heatmap_dims[0], 0.5)

fig, ax = plt.subplots(1, 1, sharex='row', sharey='col')
print(type(ax))
ax_bot = ax
print(type(ax_bot))
print(dir(ax_bot))
ax_bot.set_adjustable('box')
# ax_bot.set_adjustable('box-forced')
ax_bot.set_aspect('equal')

ax_top = ax_bot.twiny()
# plt.setp(ax_bot, aspect=1.0, adjustable='box-forced')
# This one:
# plt.setp(ax_bot, adjustable='datalim')

ax_right = ax_top.twinx()

# plt.setp(ax_bot, aspect='equal', datalim='')

# ax_bot = ax_bot.twinx()
# ax_right = ax_bot.twinx()

plt.show()
