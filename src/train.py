#!/usr/bin/env python3.7

from model import Model


if __name__ == "__main__":
    m = Model(datafilename="../data.csv", thetafilename="./theta")
    m.train()
