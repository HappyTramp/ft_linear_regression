#!/usr/bin/env python3.7

import sys
import argparse

from model import Model
import predict

class CommandLineInterface:
    def __init__(self):
        self.model = Model()

    def parse_args(self):
        parser = argparse.ArgumentParser(
            prog="ft_linear_regression_cli",
            description="CLI to interact with the ft_linear_regression project"
        )
        subparsers = parser.add_subparsers(dest="subparser_name")
        parser_train = subparsers.add_parser("train", help="train the model")
        parser_train.set_defaults(func=self._train)
        parser_train.add_argument("-a --alpha", type=float, default=1.0,
                                  dest="alpha", help="learning rate")
        parser_train.add_argument("-e --epoch", type=int, default=100,
                                  dest="epoch", help="number of iterations")
        parser_predict = subparsers.add_parser("predict", help="make a predict")
        parser_predict.set_defaults(func=self._predict)
        parser_predict.add_argument("-x", type=int,
                                    help="mileage for which the prediction will be made")
        parser_cost = subparsers.add_parser("cost", help="print model cost")
        parser_cost.set_defaults(func=self._cost)
        parser_plot = subparsers.add_parser("plot", help="plot data and model")
        parser_plot.set_defaults(func=self._plot)
        parser_plot.add_argument("-d --data", help="only plot data",
                                 action="store_true", dest="plot_data")
        parser_plot.add_argument("-m --model", help="only plot model",
                                 action="store_true", dest="plot_model")
        self.args = parser.parse_args(sys.argv[1:])

    def _train(self):
        self.model.train(self.args.alpha, self.args.epoch)
        self.model.write_theta()

    def _predict(self):
        if self.args.x is not None:
            print(self.model.make_prediction(self.args.x))
        else:
            predict.predict_input(self.model)

    def _cost(self):
        print("Cost:", self.model.cost())

    def _plot(self):
        if not self.args.plot_data and not self.args.plot_model:
            self.model.plot()
        else:
            self.model.plot(self.args.plot_data, self.args.plot_model)

    def exec_args(self):
        if self.args.subparser_name is None:
            print("{} --help for more information".format(sys.argv[0]))
            return
        self.args.func()


if __name__ == "__main__":
    cli = CommandLineInterface()
    cli.parse_args()
    cli.exec_args()
