import matplotlib.pyplot as plt


def sample_stack(stack, rows=4, cols=6, start_with=0, show_every=1, vmin=-1000, vmax=1000):
    fig,ax = plt.subplots(rows,cols,figsize=[18,20])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols),int(i % cols)].set_title(f'slice {ind}')
        ax[int(i/cols),int(i % cols)].imshow(stack[ind], vmin=vmin, vmax=vmax,cmap='gray')
        ax[int(i/cols),int(i % cols)].axis('off')
    plt.show()
