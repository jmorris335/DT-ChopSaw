from src.objects.chopsaw import ChopSaw
from src.objects.blade import Blade

import matplotlib.pyplot as plt
import numpy as np

def simple_sim():
    """Simulates a torque being applied to a blade."""
    saw = ChopSaw()
    delta = 0.1
    time = np.arange(0, 10, delta)
    omega = list()
    for t in time:
        omega.append(saw.blade.omega)
        if t < 1:
            saw.blade.applyTorque(1.2)
        saw.blade._step(delta)
    
    plt.plot(time, omega)
    plt.xlabel("Time (s)")
    plt.ylabel("Rotational Velocity (rad/s)")
    plt.show()


if __name__ == "__main__":
    simple_sim()