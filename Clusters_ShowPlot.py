import numpy as np
import matplotlib.pyplot as plt

directory = "./data_simulation/"
show_only = "./data_simulation/Cluster_10cl_X4_5K_9K_73630_XID.csv" # put _XID.csv file path here
                                                  # if you want to show only one cluster dataset
from os import listdir
from os.path import isfile, join

XIDFiles = {}

if show_only is "":  # plot everything in the directory
    for f in listdir(directory):
        path = join(directory, f)
        if (isfile(path) and f.endswith("_XID.csv")):
            XIDFiles[f[:f.index("_XID.csv")]] = path
else:  # plot only show_only
    name = show_only[show_only.rindex("/") + 1:show_only.index("_XID.csv")]
    XIDFiles[name] = show_only

for name, path in XIDFiles.items():

    # Plot the data

    XID_data = np.genfromtxt(XIDFiles[name], delimiter=',').T
    fig, ax = plt.subplots(1, 1)
    ax.scatter(XID_data[0], XID_data[1], c=XID_data[len(XID_data)-1])

    ax.set_aspect('equal')

    fig.canvas.set_window_title(name)
    fig.suptitle(name, fontsize=16)

# Actually show the plot
plt.tight_layout()
plt.show()
