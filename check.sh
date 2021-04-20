#!/bin/sh

cuda-memcheck --leak-check full ./lrgpu_mbsgd -t ./data/test_iris.csv ./data/train_iris.csv
