#!/usr/bin/env python

import argparse
import matplotlib.pyplot as mpl
import os
import sys

import bob.core
import bob.measure


def scorefile(s):
    scores = []
    with open(s, 'r') as fp:
        for line in fp:
            line.rstrip('\n')
            scores.append(float(line))
    return scores


def parse_args():
    parser = argparse.ArgumentParser('EER computation using bob')
    parser.add_argument('truescores', default=sys.stdin,
                        type=scorefile, help="A File containing all the true scores")
    parser.add_argument('impostscores', default=sys.stdin,
                        type=scorefile, help="A File containing all the impostor scores")
    parser.add_argument('out', nargs="?", help="The EER, FAR and FFR output file ( default is stdout)",
                        default=sys.stdout, type=argparse.FileType('w'))
    parser.add_argument('-det', '--plotdet', type=str,
                        default=False, help="Plots the det curve")
    return parser.parse_args()


def plot_det_curves(scores_real, scores_attacks, plot_filename, label):
    '''
    Function: plot_det_curves
    Summary: Plot, using matplotlib, Detection Error Tradeoff (DET) curve for pre-computed scores.
    Examples: InsertHere
    Attributes:
        @param (scores_real):scores for real data
        @param (scores_attacks):scores for attacks
        @param (plot_filename):name of the file with plot (pdf)
        @param (label):what goes into the legend of the plot
    Returns: None, Just plots
    '''
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(plot_filename)
    fig = mpl.figure()
    ax1 = mpl.subplot(111)

    # nagative scores is first parameter and positives is second parameter
    bob.measure.plot.det(scores_attacks, scores_real, 100,
                         color='blue', linestyle='-', label=label, linewidth=2)

    bob.measure.plot.det_axis([0.1, 99, 0.1, 99])
    mpl.xlabel('FRR (%)')
    mpl.ylabel('FAR (%)')
    mpl.title("A Detection Error Tradeoff (DET) curve for '%s'" % label)
    mpl.legend()
    mpl.grid()
    pp.savefig()
    pp.close()


def main():
    args = parse_args()

    eer_threshold = bob.measure.bob.measure.eer_threshold(
        args.impostscores, args.truescores)
    far, frr = bob.measure.farfrr(
        args.impostscores, args.truescores, eer_threshold)
    args.out.write("EER = %.2f%%, FAR = %.2f, FRR=%.2f, Threshold = %e\n" % (
        (far + frr) / 2 * 100, far, frr, eer_threshold))
    if args.plotdet:
        name, ending = os.path.splitext(args.plotdet)
        if ending == None or ending == "":
            ending = ".pdf"
        plotname = name + ending
        plot_det_curves(
            args.truescores, args.impostscores, plotname, 'Development set')


if __name__ == '__main__':
    main()
