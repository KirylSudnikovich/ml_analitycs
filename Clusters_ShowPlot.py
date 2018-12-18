import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import tkinter as tk
from tkinter.filedialog import askopenfilename
from os import listdir
from os.path import isfile, join

directory = "./data_simulation/"
#show_only = "./data_simulation/Cluster_20cl_X4_5K_9K_142009_XID.csv" # put _XID.csv file path here
                                                  # if you want to show only one cluster dataset
root = tk.Tk()
root.withdraw()
show_only = askopenfilename() # open file dialog
root.destroy()



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
    var_num = len(XID_data)-1
    fig, ax = plt.subplots(1, 1)
    #ax.position =
    sel_1st_axis = 0
    sel_2nd_axis = 1
    axcolor = 'lightgoldenrodyellow'
    rax = plt.axes([0.05, 0.4, 0.1, 0.05*var_num], facecolor=axcolor)
    radio_btns1 = RadioButtons(rax, [str(i+1) for i in range(var_num)],active = sel_1st_axis)
    rax = plt.axes([0.15, 0.4, 0.1, 0.05*var_num], facecolor=axcolor)
    radio_btns2 = RadioButtons(rax, [str(i+1) for i in range(var_num)],active = sel_2nd_axis)
    plt.subplots_adjust(left=0.3, top=0.3)
    ax.set_anchor('E')
    def Xfunc1(label):
        sel_1st_axis = int(label)-1
        ax.clear()
        ax.set_xlabel("X"+str(sel_1st_axis+1))
        ax.set_ylabel("X"+str(sel_2nd_axis+1))
        ax.scatter(XID_data[sel_1st_axis], XID_data[sel_2nd_axis], c=XID_data[var_num])
        plt.draw()
    def Xfunc2(label):
        sel_2nd_axis = int(label)-1
        ax.clear()
        ax.set_xlabel("X"+str(sel_1st_axis+1))
        ax.set_ylabel("X"+str(sel_2nd_axis+1))
        ax.set_scatter(XID_data[sel_1st_axis], XID_data[sel_2nd_axis], c=XID_data[var_num])
        plt.draw()
    radio_btns1.on_clicked(Xfunc1)
    radio_btns2.on_clicked(Xfunc2)
    ax.scatter(XID_data[sel_1st_axis], XID_data[sel_2nd_axis], c=XID_data[var_num])
    ax.set_xlabel("X"+str(sel_1st_axis+1))
    ax.set_ylabel("X"+str(sel_2nd_axis+1))
    ax.set_aspect('equal')
    fig.canvas.set_window_title(name)
    fig.suptitle(name, fontsize=16)

# Actually show the plot
plt.tight_layout()
plt.show()
