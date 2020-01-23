from model import Model


if __name__ == "__main__":
    m = Model(thetafilename="./theta")
    print("Cost:", m.cost())

