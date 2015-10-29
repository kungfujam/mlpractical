%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def hinton(matrix, max_weight=None, ax=None, coords=None, N=10, M=10):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()
#     ax.matshow(matrix)
    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

#     ax.patch.set_facecolor('gray')
#     ax.set_aspect('equal', 'box')
#     ax.xaxis.set_major_locator(plt.NullLocator())
#     ax.yaxis.set_major_locator(plt.NullLocator())
    
    if coords:
        matrix = matrix / np.sqrt(M*N)
    for (x,y),w in np.ndenumerate(matrix):
        if coords:
            n, m = coords
            x, y = m + x/M - M/2, n + x/N - N/2
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        ax.scatter(x,y)
#         rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
#                              facecolor=color, edgecolor=color)
#         ax.add_patch(rect)
    if not coords:
        ax.autoscale_view()
        ax.invert_yaxis()

sigmoid_Ws = [config['model'].layers[0].W for config in grid]

# So that all plots are relative, find the min and max values over all models
max_w = np.max([(np.max(W), np.min(W)) for W in sigmoid_Ws])
min_w = np.min([(np.max(W), np.min(W)) for W in sigmoid_Ws])

# # This currently takes 40 minutes to run
# plt.close('all')
# for i, W in enumerate(sigmoid_Ws):
#     fig, axList = plt.subplots(10, 10)  # There are 100 weights in the layer
#     fig.set_size_inches(14, 14)
#     axList = axList.flatten()
#     for j, ax in enumerate(axList):
#         hinton(W[:,j].reshape(28,28), max_weight=max_w, ax=ax)
#     fig.suptitle("Model weights for learning rate %0.3f" % learning_rates[i])
#     fig.savefig("/Users/kungfujam/git/mlpractical/data/figures/weights_hinton__lr%0.3f.eps" \
#                 %learning_rates[i], dpi=300)
#     plt.close('all')

# Trick for speed - plot in a single axes


for i, W in enumerate([sigmoid_Ws[0]]):
    M = 10
    N = 10
    ax = plt.subplot(111)
    x_offset = 3
    y_offset = 3
    plt.setp(ax, 'frame_on', False)
    ax.set_ylim([0, (M + 1)*y_offset])
    ax.set_xlim([0, (N + 1)*x_offset])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')
    for (m, n), j in np.ndenumerate(np.arange(M*N).reshape(M,N)):
        if j > 10: break
        hinton(W[:,j].reshape(28,28), max_weight=max_w, ax=ax, coords=(m,n), N=M, M=N)
        ax = plt.gca()
        ax.invert_yaxis
    plt.show()
    plt.savefig("/Users/kungfujam/git/mlpractical/data/figures/weights_hinton__lr%0.3f.eps" \
                %learning_rates[i], dpi=300)