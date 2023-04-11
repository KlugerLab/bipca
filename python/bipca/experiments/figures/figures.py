from bipca.experiments.figures.base import Figure, is_subfigure, plots
import matplotlib.pyplot as plt


class Figure2(Figure):
    _required_datasets = None

    def __init__(self, *args, **kwargs):
        super(Figure2, self).__init__(*args, **kwargs)

    @is_subfigure(label="A")
    def generate_A(self):
        print("foo")

    @is_subfigure(label="A")
    @plots
    def plot_A(self):
        fig, ax = plt.subplots(1)
        return fig, ax
