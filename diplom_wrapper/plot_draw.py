import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

np.random.seed(3)
x = 0.5 + np.arange(8)
y = np.random.uniform(2, 7, len(x))

plt.ion()
fig, ax = plt.subplots()
barPlot = ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

for phase in np.linspace(0, 10 * np.pi, 100):
    y = np.random.uniform(2, 7, len(x))
    for rect, h in zip(barPlot, y):
        rect.set_height(h)
    # ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
    # line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()