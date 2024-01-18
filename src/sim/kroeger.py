from src.objects.chopsaw import ChopSaw
from src.objects.blade import Blade, ToothType
from src.objects.motor import Motor

"""
DWS780  Type 22
                12” Double Bevel Sliding Compound Miter Saw
                Aluminum Extrusion
D28715 Type 3
                14” Chop Saw
                Aluminum Extrusion
DWS716 Type 20
                Double Bevel Compound Miter Saw
                Composite decking and wood
"""

def DWS780():
    blade = Blade(radius=0.1525, tooth_type=ToothType["TCG"], num_teeth=40,
                  arbor_dia=0.015875)
    motor = Motor()
    saw = ChopSaw(id="DWS780", blade=blade, motor=motor)
