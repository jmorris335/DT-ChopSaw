import matplotlib.pyplot as plt

from src.objects.workpiece import Workpiece
from src.objects.saw import Saw

def plotSawAndWkp(saw: Saw, wkp: Workpiece):
    blade_patch, radial_line_patch = saw.blade.plot(*saw.bladePosition())
    blade_patch.set(facecolor="orange", lw=1, edgecolor="white", label="Saw Blade")
    wkp_patch = wkp.plot()[0]
    wkp_patch.set(facecolor="silver", lw=3, edgecolor="black", label="Workpiece")

    fig, ax = plt.subplots()
    ax.add_patch(blade_patch)
    ax.add_patch(wkp_patch)
    ax.axis('equal')
    ax.legend(loc="lower center", ncols=2)
    fig.suptitle("Workpiece and Sawblade")
    subtitle = f"Saw Position: {saw.bladePosition()}"
    plt.title(subtitle)
    plt.show()

