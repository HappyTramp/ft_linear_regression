class Model:
    def __init__(self, filename='../data.csv'):
        self.datafile = filename

    def train(self):
        pass

    def partial_theta1(self):
        pass

    def partial_theta0(self):
        pass

    def gradient_descent(self):
        pass

    def read_data(self):
        pass

    def normalize_data(self):
        pass

    def write_theta(self):
        pass


if __name__ == "__main__":
    m = Model()
    m.read_data()
    m.normalize_data()
    m.train()
    m.write_theta()
