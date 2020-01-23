from model import Model

def predict_input(m):
    while True:
        try:
            x = int(input("Enter a mileage: "))
        except ValueError:
            print("Bad input, you should enter a number")
        else:
            break
    print("The predicted price for this mileage is", m.hypothesis(x))

if __name__ == "__main__":
    m = Model(thetafilename="./theta")
    predict_input(m)
