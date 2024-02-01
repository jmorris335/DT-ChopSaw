from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

rect1 = Rectangle(xy=(0, 0), width=2, height=1, color='blue')
rect2 = Rectangle(xy=(0, 0), width=2, height=1, angle=45, rotation_point='xy', color='red')
rect3 = Rectangle(xy=(0, 0), width=2, height=1, color='green')
rect3.set_angle(90)

# Plot patches
fig, ax = plt.subplots()
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.set_aspect('equal')
ax.set_xlim(-2, 3)
ax.set_ylim(-1, 3)
plt.show()