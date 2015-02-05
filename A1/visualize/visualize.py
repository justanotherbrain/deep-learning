__author__ = 'sam.royston'
import os
from parse_file import get_sessions
from matplotlib import pyplot as plt
import numpy as np
import sys


data = get_sessions(os.path.dirname(os.getcwd()) + '/experiments')

def plot_session(session):
    """
    plot performance of each number for a given session
    """
    plt.figure(figsize=(16,10))
    plt.title(session["date"] + " -- " + pretty(session["params"]))
    for i,digit in enumerate(session["test"].keys()):
        line_type = ['--' , '-.' ][i%2]
        line_width = 1
        if digit is 'avg':
            line_type = '-'
            line_width = 2
        x = np.array(xrange(0,len(session["test"][digit])))
        y = np.array(session["test"][digit])
        plt.plot(x, y, label = digit, ls=line_type, lw = line_width)
    plt.legend(loc='best')
    plt.show()

def pretty(params):
    return  params["loss"] + ", " + params["optimization"] + ", " + params["model"]

def plot_averages(all_data):
    """
    plot the average performance for each set of params
    """
    plt.figure(figsize=(20,10))
    for session in all_data:
        x = np.array(xrange(0,len(session["test"]["avg"])))
        y_clip = np.clip(np.array(session["test"]["avg"]), 0.85,1.0)
        y = np.array(session["test"]["avg"])
        print session["params"]
        plt.subplot(1, 2, 1)
        plt.plot(x, y_clip, label = pretty(session["params"]))
        plt.subplot(1, 2, 2)
        plt.legend(loc='best')
        plt.plot(x, y, label = pretty(session["params"]))
    plt.show()

def read_args():
    """
    interpret command line args
    """
    if len(sys.argv) is 1:
        plot_session(data[10])
    elif len(sys.argv) is 2:
        try:
            if sys.argv[1] == "AVGS":
                plot_averages(data)
            else:
                plot_session(data[int(sys.argv[1])])
        except:
            print "argument error"
    elif len(sys.argv) is 3:
        try:
            start = int(sys.argv[1])
            finish = int(sys.argv[2])
            for d in data[start:finish]:
                plot_session(d)
        except:
            print "argument error"
    else:
        print "too many args"

read_args()
