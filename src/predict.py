class Predictor:
    def __init__(self, filename='theta'):
        self.filename = filename
        self.theta1, self.theta0 = self.read_theta()

    def make_prediction(self, x):
        return x * self.theta1 + self.theta0

    def read_theta(self):
        try:
            with open(self.filename, 'r') as file:
                strs = file.read().strip().split(",")
                if len(strs) != 2:
                    raise "wrong theta file format"
                return int(strs[0]), int(strs[1])
        except IOError:
            print(self.filename, "do not exist")

if __name__ == "__main__":
    p = Predictor()
    x = int(input("Enter a mileage: "))
    print("The predicted price for this mileage is", p.make_prediction(x))
