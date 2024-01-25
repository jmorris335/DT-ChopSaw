import matplotlib.pyplot as plt

import matplotlib.patches as patches
from matplotlib.path import Path

from src.objects.workpiece import Workpiece
from src.objects.chopsaw import ChopSaw

def plotSawAndWkp(saw: ChopSaw, wkp: Workpiece):
    # fig, ax = plt.subplots()
    saw_patch = saw.plot()
    wkp_patch = wkp.plot()
    # ax.add_patch(saw_patch)
    # # ax.add_patch(wkp_patch)
    # plt.show()

    verts = [
    (0., 0.),  # left, bottom
    (0., 1.),  # left, top
    (1., 1.),  # right, top
    (1., 0.),  # right, bottom
    (0., 0.),  # ignored
    ]

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    path = Path(verts, codes)

    fig, ax = plt.subplots()
    patch = patches.PathPatch(path, facecolor='orange', lw=2)
    ax.add_patch(patch)
    ax.add_patch(saw_patch)
    # ax.add_patch(wkp_patch)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.show()

