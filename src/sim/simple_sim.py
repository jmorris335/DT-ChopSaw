from src.objects.chopsaw import ChopSaw
from src.objects.blade import Blade
from src.objects.motor import Motor

import matplotlib.pyplot as plt
import numpy as np

def simple_sim():
    """Simulates a torque being applied to a blade."""
    saw = ChopSaw()
    delta = 0.1
    time = np.arange(0, 10, delta)
    omega = list()
    (T, omega) = saw.blade._ss.step()
    # for t in time:
    #     omega.append(saw.blade.omega)
    #     if t < 1:
    #         saw.blade.applyTorque(1.2)
    #     saw.blade._step(delta)
    
    plt.plot(T, omega)
    plt.xlabel("Time (s)")
    plt.ylabel("Rotational Velocity (rad/s)")
    plt.show()

def simple_blade():
    blade = Blade()
    omega, theta = list(), list()
    time = np.linspace(0, 10, 2035)
    for t in time:
        if t > 0: blade.applyTorque(0)
        if t > 2: blade.applyTorque(1.5)
        if t > 4: blade.applyTorque(-0.2)
        blade.step(dt=time[1] - time[0])
        xout = blade.getStates()
        omega.append(xout[1])
        theta.append(xout[0])

    plt.plot(time, omega)
    plt.xlabel("Time (s)")
    plt.ylabel("Rotational Velocity (rad/s)")
    plt.show()

def simple_motor():
    motor = Motor()
    theta, omega, current = list(), list(), list()
    time = np.linspace(0, 10, 100)
    for t in time:
        if t > 2: motor.applyVoltage(12)
        if t > 8: motor.applyVoltage(0)
        if t > 5: motor.applyLoad(0.1)
        if t > 8: motor.applyLoad(0)
        motor.step(dt = time[1] - time[0])
        xout = motor.getStates()
        theta.append(xout[0])
        omega.append(xout[1])
        current.append(xout[2])
    
    plt.plot(time, omega)
    plt.xlabel("Time (s)")
    plt.ylabel("Rotational Velocity (rad/s)")
    plt.show()

if __name__ == "__main__":
    simple_blade()