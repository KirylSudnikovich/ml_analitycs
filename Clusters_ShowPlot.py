import numpy as np
import matplotlib.pyplot as plt

directory = "./data_simulation/"
show_only = "./data_simulation/Cluster5_viz.npz"  # put _viz.npz file path here
                                                  # if you want to show only one cluster dataset
from os import listdir
from os.path import isfile, join

VizFiles = {}
Xfiles = {}
Yfiles = {}

if show_only is "":  # plot everything in the directory
    for f in listdir(directory):
        path = join(directory, f)
        if (isfile(path) and f.endswith("_Y.csv")):
            Yfiles[f[:f.index("_Y.csv")]] = path
        elif (isfile(path) and f.endswith("_X.csv")):
            Xfiles[f[:f.index("_X.csv")]] = path
        elif (isfile(path) and f.endswith("_viz.npz")):
            VizFiles[f[:f.index("_viz.npz")]] = path
else:  # plot only show_only
    pathbase = show_only[:show_only.index("_viz.npz")]
    name = pathbase[show_only.rindex("/") + 1:]
    VizFiles[name] = show_only
    Xpath = pathbase + "_X.csv"
    Ypath = pathbase + "_Y.csv"
    if isfile(Xpath):
        Xfiles[name] = Xpath
    if isfile(Ypath):
        Yfiles[name] = Ypath

for name, path in VizFiles.items():
    npzfile = np.load(path)
    ClustersX = npzfile["clX"]
    ClustersY = npzfile["clY"]
    # Plot the data
    if (name in Yfiles) and (name in Xfiles):
        fig, ax = plt.subplots(1, 2)
        for i in range(len(ClustersX)):
            ax[0].scatter(ClustersX[i], ClustersY[i])

        ax[1].scatter(np.genfromtxt(Xfiles[name], delimiter=','), np.genfromtxt(Yfiles[name], delimiter=','))
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
