import numpy as np
import matplotlib.pyplot as plt

directory = "./data_simulation/"
show_only = "./data_simulation/Cluster5_viz.npz"  # put _viz.npz file path here
                                                  # if you want to show only one cluster dataset
from os import listdir
from os.path import isfile, join

XYIDFiles = {}
VizFiles = {}

if show_only is "":  # plot everything in the directory
    for f in listdir(directory):
        path = join(directory, f)
        if (isfile(path) and f.endswith("_XYID.csv")):
            XYIDFiles[f[:f.index("_XYID.csv")]] = path
        elif (isfile(path) and f.endswith("_viz.npz")):
            VizFiles[f[:f.index("_viz.npz")]] = path
else:  # plot only show_only
    pathbase = show_only[:show_only.index("_viz.npz")]
    name = pathbase[show_only.rindex("/") + 1:]
    VizFiles[name] = show_only
    XYIDpath = pathbase + "_XYID.csv"
    if isfile(XYIDpath):
        XYIDFiles[name] = XYIDpath

for name, path in VizFiles.items():
    npzfile = np.load(path)
    XYID_data = np.genfromtxt(XYIDFiles[name], delimiter=',').T
    ClustersX = npzfile["clX"]
    ClustersY = npzfile["clY"]
    # Plot the data
    if (name in XYIDFiles):
        fig, ax = plt.subplots(1, 2)
        for i in range(len(ClustersX)):
            ax[0].scatter(ClustersX[i], ClustersY[i])

        ax[1].scatter(XYID_data[0], XYID_data[1])
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
    else:
        fig, ax = plt.subplots(1, 1)
        for i in range(len(ClustersX)):
            ax.scatter(ClustersX[i], ClustersY[i])
        ax.set_aspect('equal')
    fig.canvas.set_window_title(name)
    fig.suptitle(name, fontsize=16)

# Actually show the plot
plt.tight_layout()
plt.show()
