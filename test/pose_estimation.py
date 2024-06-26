import matplotlib.pyplot as plt

import os, sys
os.chdir("/Users/john_morris/Documents/Clemson/Research/DTs-Public/DT-ChopSaw")
sys.path.insert(0, '')
from src.mocap.markers import Marker
from src.mocap.mocap import *
    
stem_offset = 1
stem_height = 1
slider_length = 1
arm_length = 2

base = Marker('base', 0)
base.center = [(0, 0, -1)]
miter = Marker('miter', 1)
miter.center = [(1, 0, 0)]
miter_offset = (0,0,0)
bevel = Marker('bevel',2)
bevel.center = [(-stem_offset, stem_height, 0)]
bevel_offset = (0,0,0)
slider = Marker('slider',3)
slider.center = [(slider_length-stem_offset, stem_height, 0)]
slider_m_offset = (0,0,0)
crash = Marker('crash',4)
crash.center = [(arm_length+slider_length-stem_offset, stem_height, 0)]
crash_offset = (0,0,0)

miter_angle = findMiterAngle(miter, miter_offset)
bevel_COR = calcBevelCOR(miter_angle, stem_offset)
bevel_angle = findBevelAngle(bevel, bevel_offset, bevel_COR)
stem_top = calcStemTop(miter_angle, stem_offset, bevel_angle, bevel_COR,
                        stem_height)
slider_offset = findSliderOffset(slider, slider_m_offset, stem_top)
crash_COR = calcCrashCOR(miter_angle, stem_offset, bevel_angle, bevel_COR,
                            stem_height, slider_offset, slider_length)
crash_zero = calcCrashZero(miter_angle, stem_offset, bevel_angle, bevel_COR,
                            stem_height, slider_offset, slider_length)
crash_angle = calcCrashAngle(crash, crash_offset, crash_zero, crash_COR)

print(f"Miter \tAngle: {(miter_angle*180/np.pi):.1f}; \tCOR: (0, 0, 0)")
print(f"Bevel \tAngle: {(bevel_angle*180/np.pi):.1f}; \tCOR: {bevel_COR}")
print(f"Slider \tOffset: {slider_offset}; \tStem Top: {stem_top}")
print(f"Crash \tAngle: {(crash_angle*180/np.pi):.1f}; \tCOR: {crash_COR}")

x, y, z = zip((0,0,0), bevel_COR, stem_top, crash_COR)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z)
plt.xlabel('x')
plt.ylabel('y')
plt.show()