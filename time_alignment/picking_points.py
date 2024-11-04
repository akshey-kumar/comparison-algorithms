"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.arange(20)
y = x/2 + np.random.randn(20)
points, = ax.plot(x, y, 'o', picker=3)  # 'o' marker for points

def on_pick(event):
    ind = event.ind[0]  # Get index of the picked point
    x, y = points.get_data()
    print("Selected point:", x[ind], y[ind])

fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()


exit()
"""

import matplotlib.pyplot as plt
import numpy as np

plt.plot([0, 1], [0, 1])
points = np.array(plt.ginput(5))  # Capture two points from clicks
print("You clicked:", points)

plt.show()  # Show the plot and wait for clicks
