import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing


class Model:
    def __init__(self, datafilename="../data.csv", thetafilename="./theta"):
        self.datafilename = datafilename
        self.thetafilename = thetafilename
        self.theta1, self.theta0 = self._read_theta()
        self.xs, self.ys = self._read_data()

    def train(self, alpha=1, epoch=100):
        self._normalize_data()
        for _ in range(epoch):
            next_theta0 = self.theta0 - alpha * self._partial_theta0()
            next_theta1 = self.theta1 - alpha * self._partial_theta1()
            self.theta0 = next_theta0
            self.theta1 = next_theta1

    def write_theta(self):
        with open(self.thetafilename, "w") as file:
            file.write("{},{}".format(str(self.theta1), str(self.theta0)))

    def hypothesis(self, x):
        return x * self.theta1 + self.theta0

    def cost(self):
        return (1 / (2 * len(self.xs))) * sum([(self.hypothesis(x) - y) ** 2
                                               for x, y in zip(self.xs, self.ys)])

    def plot(self, plot_data=True, plot_model=True):
        self.fig, self.ax = plt.subplots()
        if plot_data:
            self._plot_data()
        if plot_model:
            self._plot_model()
        plt.show()

    def _plot_data(self):
        self.ax.scatter(self.xs, self.ys)

    def _plot_model(self):
        line_xs = [self.xs.min(), self.xs.max()]
        line_ys = [self.hypothesis(x) for x in line_xs]
        self.ax.plot(line_xs, line_ys)

    def _partial_theta1(self):
        return sum([(self.hypothesis(x) - y) * x
                    for x, y in zip(self.xs, self.ys)]) / len(self.xs)

    def _partial_theta0(self):
        return sum([self.hypothesis(x) - y
                    for x, y in zip(self.xs, self.ys)]) / len(self.xs)

    def _normalize_data(self):
        self.xs, self.ys = sklearn.preprocessing.normalize([self.xs, self.ys])

    def _read_theta(self):
        try:
            with open(self.thetafilename, "r") as file:
                strs = file.read().strip().split(",")
                if len(strs) != 2:
                    raise "wrong theta file format"
                return float(strs[0]), float(strs[1])
        except IOError:
            print(self.thetafilename, "do not exist")

    def _read_data(self):
        try:
            data = np.genfromtxt(self.datafilename, delimiter=",")[1:]
            return data[:, 0], data[:, 1]
        except IOError:
            print(self.datafilename, "do not exist")


if __name__ == "__main__":
    m = Model()
    m.train()
    m.write_theta()
