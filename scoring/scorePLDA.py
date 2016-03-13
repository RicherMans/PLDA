#!/usr/bin/env python
import numpy as np
import os
import logging as log
try:
    import sys
    sys.path.insert(0, os.path.realpath(__file__))
    import htkpython.htkfeature as htkfeature
except:
    import htkfeature
try:
    from liblda import PLDA
except:
    # Check if the user maybe has the library in the current dir
    sys.path.insert(0, os.path.realpath(__file__))
    log.warn("liblda cannot be found in the pythonpath.")
    log.warn("Looking for liblda in the current filepath!.")
    try:
        import liblda.liblda as liblda
        from liblda import PLDA
    except:
        log.error(
            "Could not find liblda! Either install it directly on the machine or copy the library into the folder liblda of this path!")
        raise
import sys
import argparse
from collections import defaultdict
import marshal
from extractdvector import *


def float_zeroone(value):
    float_repr = float(value)
    if float_repr < 0 or float_repr > 1:
        raise argparse.ArgumentTypeError('Value has to be between 0 and 1')
    return float_repr

# Parses Mlf file


def mlffile(f):
    tests = defaultdict(list)
    with open(f, 'r') as mlfpointer:
        #skip the #!MLF!#
        next(mlfpointer)
        for line in mlfpointer:
            line = line.rstrip('\n')
            if line.startswith("\""):
                withoutlab = line.split(".")[0]
                withoutslashes = withoutlab.split("/")[1]
                # Split the model and utt
                modeltotestutt = withoutslashes.split("-")
                targetmdl = modeltotestutt[0]
                # In the case that there is some utterance having another -
                testutt = "-".join(modeltotestutt[1:])
                # Get the next line which identifies the target model
                enrolemodel = next(mlfpointer).rstrip('\n')
                tests[enrolemodel].append([testutt, targetmdl])
    return tests

# Imported from dvector
methods = {
    'mean': extractdvectormean,
    'max': extractdvectormax,
    'var': extractdvectorvar
}


def parse_args():
    parser = argparse.ArgumentParser(
        'Scores the enrole models against the testutterances')
    parser.add_argument('bkgdata', type=str,
                        help="Background data which is used to train the Model. Also a folder can be passed!")
    parser.add_argument(
        'inputdata', type=str, help='Input dir or a file specifying all the utterances')
    parser.add_argument(
        'testdata', type=str, help='Input dir or a file specifying the test utterances')
    parser.add_argument(
        'testmlf', type=mlffile, help='test.mlf file to get the tests. Model and utterance are separated by "-"! E.g. F001-F001_session1_utt1.')
    parser.add_argument(
        'scoreoutfile', default=sys.stdout, type=argparse.FileType('w'), nargs="?", metavar="STDOUT")
    parser.add_argument('-z', '--znorm', type=str,
                        help="Does Z-Norm with the given dataset.Z-Norm generally improves the performance")
    parser.add_argument(
        '--zutt', help="Number of znorm utterances, default is all the bkg size", type=int)
    parser.add_argument(
        '-e', '--extractionmethod', choices=methods, default='mean', help='The method which should be used to extract dvectors'
    )
    parser.add_argument('-del', '--delimiter', type=str,
                        help='If we extract the features from the given data, we use the delimiter (default : %(default)s) to obtain the splits.',
                        default="_")
    parser.add_argument('--smoothing', type=float_zeroone,
                        help="Smoothing factor during the PLDA transformation. default %(default)s", default=1.0)
    parser.add_argument(
        '--iters', type=int, help="Number of iterations for the PLDA estimation, default is %(default)s", default=10)
    parser.add_argument(
        '-id', '--indices', help="The indices of the given splits which are used to determinate the speaker labels! default is rsr %(default)s", nargs="+", default=[0, 2], type=int)
    parser.add_argument('-d', '--debug', help="Sets the debug level. A value of 10 represents debug. The lower the value, the more output. Default is INFO",
                        type=int, default=log.INFO)
    return parser.parse_args()


def checkmarshalled(files):
    '''
    Function: checkmarshalled
    Summary: Checks if the given files in a list are marshalled or not by simply opening them.
    Examples: checkmarshalled([file1,file2,file3])
    Attributes:
        @param (files):List of opened files (open('rb'))
    Returns: A list of the given opened files if sucessful, otherwise none
    '''
    marshalledfiles = []
    for f in files:
        try:
            marshalledfiles.append(marshal.load(f))
        except:
            return
    return marshalledfiles


def main():
    args = parse_args()
    log.basicConfig(
        level=args.debug, format='%(asctime)s %(levelname)s %(message)s', datefmt='%d/%m %H:%M:%S')

    marshalformat = False
    # Check if the given data is in marshal format
    with open(args.bkgdata, 'rb') as bkg, open(args.inputdata, 'rb') as enrol, open(args.testdata, 'rb') as testd:
        # If marshalled data is given, it was preprocessed using dvectors
        bkgdata, inputdata, testdata = checkmarshalled([bkg, enrol, testd])

        # Check if marshal format is given, so that we do not need to reextract
        # the data
        if bkgdata and inputdata and testdata:
            marshalformat = True
        enroldvectors = inputdata
        bkgdvectors = bkgdata
        testdvectors = testdata

        # Get the labels for the speakers
        enrollabels = []
        bkglabels = []
        testlabels = []
        for spk, v in enroldvectors.iteritems():
            enrollabels.extend([spk for i in xrange(len(v))])
        for spk, v in bkgdvectors.iteritems():
            bkglabels.extend([spk for i in xrange(len(v))])
        for spk, v in testlabels.iteritems():
            testlabels.extend([spk for i in xrange(len(v))])

    if not marshalformat:
        # Note that I just dont know hot to add these extra parameters ( delim and indices)
        # To the argparser, therefore we just use strings and call the method
        # later
        bkgdata = parseinputfiletomodels(
            args.bkgdata, args.delimiter, args.indices, test=True)
        enroldata = parseinputfiletomodels(
            args.inputdata, args.delimiter, args.indices)
        testdata = parseinputfiletomodels(
            args.testdata, args.delimiter, args.indices, test=True)

        extractmethod = methods[args.extractionmethod]

        # Extraction of the dvectors
        log.info("Extracting dvectors for enrolment data")
        enroldvectors, enrollabels = extractvectors(enroldata, extractmethod)
        log.info("Extracting dvectors for background data")
        bkgdvectors, bkglabels = extractvectors(bkgdata, extractmethod)
        log.info("Extracting dvectors for test data")
        testdvectors, testlabels = extractvectors(testdata, extractmethod)

    # Debugging information
    log.debug("Enrol dvectors have dimension (%i,%i) and overall %i labels" % (
        enroldvectors.shape[0], bkgdvectors.shape[1], len(enrollabels)))
    log.debug("Background dvectors have dimension (%i,%i)" %
              (bkgdvectors.shape[0], bkgdvectors.shape[1]))

    # Transform the string labels to unique integer id's
    enrolspktonum = {
        spk: num for num, spk in enumerate(np.unique(enrollabels))}
    enrollabelsenum = np.array([enrolspktonum[i]
                                for i in enrollabels], dtype='uint')

    bkgspktonum = {spk: num for num, spk in enumerate(np.unique(bkglabels))}
    bkglabelsenum = np.array([bkgspktonum[i] for i in bkglabels], dtype='uint')

    testspktonum = {spk: num for num, spk in enumerate(np.unique(testlabels))}
    testlabelsenum = np.array([testspktonum[i]
                               for i in testlabels], dtype='uint')

    # Beginning PLDA estimation
    plda = PLDA()

    log.debug("Background labels are of type %s, enrolment labels of type %s " % (
        type(bkglabelsenum), type(enrollabelsenum)))
    log.info("Starting to estimate PLDA model for background data")

    plda.fit(bkgdvectors, bkglabelsenum, args.iters)

    # Transforming the enrolment and test datasets to the PLDA space

    log.info("Transforming enrolment data vector to PLDA space")
    enroltransform = plda.transform(enroldvectors, enrollabelsenum)
    log.info("Transforming test data vector to PLDA space")
    testtransform = plda.transform(testdvectors, testlabelsenum)

    # Perform znorm mean/var estimation.
    # We extract for the znorm dataset the dvectors and then estimate mean/var of the dataset using
    # the scores obtained by scoring the enrolment model against the
    # "impostor" znorm dataset utterances
    if args.znorm:
        log.debug("Running Z-Norm")
        znormdata = parseinputfiletomodels(
            args.znorm, args.delimiter, args.indices, test=True)
        log.info("Extracting z-norm data dvectors")
        znormdvectors, znormlabels = extractvectors(znormdata, extractmethod)
        log.info("Estimating z-norm")
        plda.norm(znormdvectors, enroltransform, args.zutt)

    errors = 0
    log.info("Beginning scoring")
    for enrolemodel, vals in args.testmlf.iteritems():
        if enrolemodel not in enrolspktonum:
            errors += 1
            log.warn("Enrolemodel %s not found in the labels" % (enrolemodel))
            continue
        enrolspk = enrolspktonum[enrolemodel]
        for testutt, targetmdl in vals:
            if testutt not in testspktonum:
                log.warn("Utterance %s not found in the testset" % (testutt))
                errors += 1
                continue
            # Enrolmodel is a string given by the mlf file. We transform that
            # string to the integer classid with enrolspktonum
            score = plda.score(enrolspk, enroltransform[
                               enrolspk], testtransform[testspktonum[testutt]])
            args.scoreoutfile.write(
                "{} {}-{} {:.3f}\n".format(targetmdl, enrolemodel, testutt, score))
    if errors > 0:
        log.warn(
            "Overall %i errors occured during the testing phase!" % (errors))
    log.info("Scoring done! Scores can be seen in the file %s" %
             (args.scoreoutfile.name))

if __name__ == "__main__":
    main()
