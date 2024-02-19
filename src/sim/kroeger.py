from src.objects.saw import Saw
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
    #Blade: Diablo 1296N: https://www.diablotools.com/products/D1296N
    blade = Blade(id="1296N-1", radius_blade=0.1525, tooth_type=ToothType["TCG"], num_teeth_blade=96,
                  arbor_dia=0.015875, kerf_blade=.0023, hook=-0.0872664626, thickness=0.0018)
    motor = Motor()
    saw = Saw(id="DWS780-1", blade=blade, motor=motor)
