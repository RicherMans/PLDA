#!/usr/bin/env python
import numpy as np
import sys
import argparse
# import pywt
import os
import logging as log
from collections import defaultdict
import itertools
import marshal
import cPickle
from extractdvector import *
try:
    sys.path.insert(0, os.path.realpath(__file__))
    import htkpython.htkfeature as htkfeature
except:
    import htkfeature
try:
    # Search in the usual path for Scikit Library
    from sklearn.lda import LDA
except:
    from liblda import LDA
    # Check if the user maybe has the library in the current dir
    sys.path.insert(0, os.path.realpath(__file__))
    log.warn("liblda cannot be found in the pythonpath.")
    log.warn("Looking for liblda in the current filepath!.")
    try:
        import liblda.liblda as liblda
        from liblda import LDA
    except:
        log.error(
            "Could not find liblda! Either install it directly on the machine or copy the library into the folder liblda of this path! (e.g. liblda/libplda.so )")
        raise


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


def drawProgressBar(percent, barLen=20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent))
    sys.stdout.flush()


def checkBinary(filenames):
    '''
    Function: checkBinary
    Summary: Checks if filesnames are binary marshal or cpickle dumps
    Examples:
    Attributes:
        @param (filenames):A list of filenames
    Returns: Tuple of bkg,enrol,test data if they exist
    '''
    ret = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            log.debug("Check if file %s is in CPickle Format" % (filename))
            curret = checkCPickle(f)
            if not curret:
                log.debug("Checking if file %s is in Marshal Format" %
                          (filename))
                curret = checkmarshalled(f)
                if not curret:
                    return
            ret.append(curret)
    return ret


def test_ref(f):
    tests = defaultdict(list)
    with open(f, 'r') as testpointer:
        for line in testpointer:
            line = line.rstrip('\n')
            targetmodel, enrol_testutt = line.split()[:2]
            modeltotestutt = enrol_testutt.split("-")
            enrolemdl = modeltotestutt[0]
            testutt = '-'.join(modeltotestutt[1:])
            tests[targetmodel].append([testutt, enrolemdl])
    return tests

methods = {
    'mean': extractdvectormean,
    'var': extractdvectorvar,
    'max': extractdvectormax,
}


def parse_args():
    parser = argparse.ArgumentParser(
        'Scores the enrole models against the testutterances')
    parser.add_argument(
        'inputdata', type=str, help='Input dir or a file specifying all the utterances for enrolment')
    parser.add_argument(
        'testutts', type=str, help='Input dir or a file specifying the utterances')
    parser.add_argument(
        'testref', type=str, help='test.mlf/test_ref file to get the tests. Model and utterance are separated by "-"! ')
    parser.add_argument(
        'scoreoutfile', default=sys.stdout, type=argparse.FileType('w', 50485760), nargs="?", metavar="STDOUT")
    parser.add_argument('-d', '--debug', default=log.INFO, metavar='DEBUGLEVEL',
                        help="Sets the debug level. A level of 10 represents DEBUG. Higher levels are 20 = INFO (default), 30 = WARN", type=int)
    parser.add_argument('-del', '--delimiter', type=str,
                        help='If we extract the features from the given data, we use the delimiter (default : %(default)s) to obtain the splits.',
                        default="_")
    parser.add_argument(
        '-mlf', help='Uses mlf file as the label file', action="store_true", default=False)
    parser.add_argument(
        '-b', '--binary', help="Specify if the given input is binary ( either marshalled or cPickle)", action='store_true', default=False)
    parser.add_argument(
        '-id', '--indices', help="The indices of the given splits which are used to determinate the speaker labels! default is rsr %(default)s", nargs="+", type=int, default=[0, 2])
    parser.add_argument(
        '-e', '--extractionmethod', choices=methods, default='mean', help='The method which should be used to extract dvectors, default: %(default)s'
    )
    return parser.parse_args()
args = parse_args()
log.basicConfig(
    level=args.debug, format='%(asctime)s %(levelname)s %(message)s', datefmt='%d/%m %H:%M:%S')

extractmethod = methods[args.extractionmethod]


def checkmarshalled(marshalfile):
    '''
    Function: checkmarshalled
    Summary: Checks if the given files in a list are marshalled or not by simply opening them.
    Examples: checkmarshalled(file1)
    Attributes:
        @param (files):List of opened files (open('rb'))
    Returns: A list of the given opened files if sucessful, otherwise none
    '''
    try:
        return marshal.load(marshalfile)
    except:
        return


def checkCPickle(cpicklefile):
    try:
        return cPickle.load(cpicklefile)
    except:
        return


def parsepaths(paths):
    mdict = {}
    for path in paths:
        bname, ext = os.path.splitext(os.path.basename(path))
        mdict[bname] = extractmethod(path)
    return mdict


def main():
    lda = LDA(solver='svd')

    # Will be filled as dict
    testreferences = None
    if args.mlf:
        log.info("Parsing input label file %s as mlf file" % (args.testref))
        testreferences = mlffile(args.testref)
    else:
        log.info(
            "Parsing input label file %s as test_ref file( TARGETMDL ENROLEMODEL-TESTUTT LABEL )" % (args.testref))
        testreferences = test_ref(args.testref)
    # Check if the given data is in marshal format or cPickle
    if args.binary:
        log.info("Try to read input as a binary file")
        inputdata, testutts = checkBinary([args.inputdata, args.testutts])
        datadim = len(inputdata.values()[0])
        dvectors = np.zeros((len(inputdata.keys()), datadim))
        labels = []
        log.info("Getting dvectors from input data")
        for i, (spk, v) in enumerate(inputdata.iteritems()):
            dvectors[i] = v
            labels.append(getspkmodel(spk, args.delimiter, args.indices))
        log.debug("Data which was loaded in (%s) has %i labels and %i dvectors" %
                  args.inputdata, (len(labels), len(dvectors)))
        testtofeature = testutts

    else:
        log.info("Given data is either a folder or a filelist. Trying to read")
        inputdata = parseinputfiletomodels(
            args.inputdata, args.delimiter, args.indices)
        testtofeature = parsepaths(readFeats(args.testutts))
        labels = []
        dvectors = []
        log.info("Extracting dvectors for input data")
        for spk, v in inputdata.iteritems():
            dvectors.extend(itertools.imap(extractmethod, v))
            labels.extend([spk for i in xrange(len(v))])
    spktonum = {spk: num for num, spk in enumerate(np.unique(labels))}
    dvectors = np.array(dvectors)
    labelsnum = np.array([spktonum[i] for i in labels])

    log.debug("Overall we have %i labels" % (len(labelsnum)))
    log.debug("Number of speakers: %i" % (len(spktonum.keys())))
    log.debug("Dvector size is (%i,%i)" %
              (dvectors.shape[0], dvectors.shape[1]))

    log.info("Fitting LDA model")
    lda.fit(dvectors, labelsnum)

    errors = 0
    log.info("Starting test")
    for enrolemodel, vals in testreferences.iteritems():
        if enrolemodel not in spktonum:
            errors += 1
            log.warn("Enrolemodel %s not found in the labels" % (enrolemodel))
            continue
        curspk = spktonum[enrolemodel]
        for testutt, targetmdl in vals:
            if testutt not in testtofeature:
                log.warn("Utterance %s not found in the testset" % (testutt))
                errors += 1
                continue
            # LDA currently needs a twodimensional imput vector
            testdvector = testtofeature[testutt][np.newaxis, :]
            score = lda.predict_log_proba(testdvector)[0]
            # score is a list of all the scores ( not only one ). we need to
            # find the current speakers score
            finalscore = score[curspk]
            args.scoreoutfile.write(
                "{} {}-{} {:.3f}\n".format(enrolemodel, targetmdl, testutt, finalscore))
    if errors > 0:
        log.warn(
            "Overall %i happened while processing the testutterances. The scores may not be complete" % (errors))
    log.info("LDA estimation done, output file is: %s. Output file has the following structure: TARGETMODEL ENROLEMODEL-TESTUTT" %
             (args.scoreoutfile.name))

if __name__ == '__main__':
    main()
