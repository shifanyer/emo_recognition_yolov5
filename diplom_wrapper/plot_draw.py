import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation

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

fig, ax = plt.subplots()
x = [3, 5, 8]
y = [9, 8, 4]
ln, = ax.plot(x, y, '-')

def update(frame):
    global x, y
    x.append(9)
    y.append(6)

    ln.set_data(x, y)
    fig.gca().relim()
    fig.gca().autoscale_view()
    return ln,


animation = FuncAnimation(fig, update, interval=2000)
plt.show()