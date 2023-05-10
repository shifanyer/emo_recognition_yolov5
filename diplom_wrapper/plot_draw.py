import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
from pylab import *
import time

matplotlib.use("TkAgg")

# np.random.seed(3)
# x = 0.5 + np.arange(8)
# y = np.random.uniform(2, 7, len(x))
#
# plt.ion()
# fig, ax = plt.subplots()
# barPlot = ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
#
# for phase in np.linspace(0, 10 * np.pi, 100):
#     y = np.random.uniform(2, 7, len(x))
#     for rect, h in zip(barPlot, y):
#         rect.set_height(h)
#     # ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
#     # line1.set_ydata(np.sin(0.5 * x + phase))
#     fig.canvas.draw()
#     fig.canvas.flush_events()


# def bar_plots_first_run(names):
#     x = 0.5 + np.arange(len(names))
#     y = np.random.uniform(0, 1, len(names))
#     barPlot = ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
#
#
# def redraw_plot(order_list, data_list):
#     data_array = data_list.tolist()
#     y = np.array(data_array)
#     for rect, h in zip(barPlot, y):
#         rect.set_height(h)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#
# class BarPlotClass(object):
#
#     def __init__(self, names):
#         x = 0.5 + np.arange(len(names))
#         y = np.random.uniform(0, 1, len(names))
#         plt.ion()
#         fig, ax = plt.subplots()
#         self.barPlot = ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
#
#     def draw(self, x, y, names):
#         plt.ion()
#         fig, ax = plt.subplots()
#
#         ax.set_xticklabels(names)
#
#     def redraw_plot(self, order_list, data_list):
#         data_array = data_list.tolist()
#         y = np.array(data_array)
#         for rect, h in zip(self.barPlot, y):
#             rect.set_height(h)
#         fig.canvas.draw()
#         fig.canvas.flush_events()


x = np.linspace(0, 10 * np.pi, 100)
y = np.sin(x)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')

for phase in np.linspace(0, 10 * np.pi, 100):
    line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()
