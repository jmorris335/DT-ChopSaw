import plotly.express as px

class Plotter:
    def __init__(self, data):
        self.data = data
        self.fig = px.scatter_3d(self.data)
        self.fig.show()

if __name__ == "__main__":
    data = [[1,1,1], [2,2,2], [1,2,3]]
    p = Plotter(data)