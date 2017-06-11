import numpy as np
def plotIt(indices, gridSize, pathBase, grid_cell=(10, 10), decay=.5):
        colors = np.zeros((gridSize[0], gridSize[1], 3))
        fig, ax = plt.subplots()
        ax.grid(which='major', axis='both', linestyle='-', color='w', linewidth=2)
        axs = np.arange(0, gridSize[0], 1)
        ays = np.arange(0, gridSize[1], 1)
        ax.set_xticks(axs*grid_cell[0]);
        ax.set_yticks(ays*grid_cell[1]);
        ax.set_xticklabels(axs)
        ax.set_yticklabels(ays)
        for i, idx in enumerate(indices):
            colors *= decay
            if idx is not None:
                for id in idx:
                    colors[id[0], id[1], 2] = 1
            img = np.repeat(np.repeat(colors, grid_cell[0], axis=1), grid_cell[1], axis=0)
            ax.imshow(img)
            fig.suptitle("TimeStep: {:5d}".format(i), fontsize=20)
            fig.savefig(pathBase + "00" + str(i) + ".png")

    plotIt(indices, mem_size, name_, grid_cell=(50,50))
    print("Creating "+name_.split("/")[-1]+".gif")
    from subprocess import check_call
    check_call(["plotIt.sh",name_.split("/")[-1]])