from collections import OrderedDict
import matplotlib.pyplot as plt
from bipca.experiments.base import ABC, abstractclassattribute, abstractmethod, classproperty

## Params for exporting to illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
## Params for plotting
SMALL_SIZE=8
BIGGER_SIZE=10
plt.rcParams['text.usetex'] = True

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


## Orders, cmaps, etc used for all figures
plotting_alg_to_npg_cmap_idx = OrderedDict(
    [("log1p", 0), ("log1p_z", 4), ("SCT", 3), ("Sanity", 1), ("ALRA", 8), ("bipca", 2)]
)
def npg_cmap(alpha=1):
    string = '"#E64B35FF" "#4DBBD5FF" "#00A087FF" "#3C5488FF" "#F39B7FFF" "#8491B4FF" "#91D1C2FF" "#DC0000FF" "#7E6148FF" "#B09C85FF"'
    output = extract_color_list_from_string(string)
    cmap = mpl.colors.ListedColormap(output)
    return get_alpha_cmap_from_cmap(cmap,alpha=alpha)


class Figure(ABC):
    _required_datasets = abstractclassattribute()
