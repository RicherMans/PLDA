#!/usr/bin/env python
import numpy as np
import sys
import argparse
# import pywt
import os
import logging as log
from collections import defaultdict
import itertools
import sys
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


def readFeats(value):

    if os.path.isfile(value):
        return open(value, 'r').read().splitlines()
    else:
        return readDir(value)


def readDir(input_dir):
    '''
    Reads from the given Inputdir recursively down and returns all files in the directories.
    Be careful since there is not a check if the file is of a specific type!
    '''
    foundfiles = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if os.path.isfile(os.path.join(root, f)):
                foundfiles.append(os.path.abspath(os.path.join(root, f)))
    return foundfiles


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
                tests[enrolemodel].append([testutt,targetmdl])
    return tests


def parseinputfiletomodels(filepath, delim, ids, test=False):
    '''
    Function: parseinputfiletomodels
    Summary: Parses the given filepath into a dict of speakers and its uttrances
    Examples: parseinputfiletomodels('bkg.scp','_',[0,2])
    Attributes:
        @param (filepath):Path to the dataset. Dataset consists of absolute files
        @param (delim):Delimited in how to extract the speaker from the filename
        @param (ids):After splitting the filename using delim, the indices which parts are taken to be the speaker
        @param (test):If true, test will result in using the whole utterance name as speaker model
    Returns: Dict which keys are the speaker and values are a list of utterances
    '''
    lines = readFeats(filepath)
    speakertoutts = defaultdict(list)
    for line in lines:
        line = line.rstrip("\n")
        fname = line.split("/")[-1]
        fname, ext = os.path.splitext(fname)
        splits = fname.split(delim)
        # If we have no test option, we split the filename with the give id's
        speakerid = delim.join([splits[id] for id in ids])
        if test:
            speakerid = delim.join(splits)
        speakertoutts[speakerid].append(line)
    return speakertoutts


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


def getnormalizedvector(utt):
    '''
    Function: getnormalizedvector
    Summary: Reads in the utterance given as utt and returns a length normalized vector
    Examples: getnormalizedvector('myfeat.plp')
    Attributes:
        @param (utt):Path to the utterance which needs to be read
    Returns: A numpy array
    '''
    feat = np.array(htkfeature.read(utt)[0])
    denom = np.linalg.norm(feat, axis=1)
    return feat / denom[:, np.newaxis]


def extractdvectormax(utt):
    # Average over the saples
    return np.max(getnormalizedvector(utt), axis=0)


def extractdvectormean(utt):
    # Average over the saples
    return np.mean(getnormalizedvector(utt), axis=0)


def extractdvectorvar(utt):
    # Average over the saples over the feature dim
    # normalized has dimensions (n_samples,featdim)
    # We Do not use np.diag(np.cov()), because somehow memory overflows with it
    return np.var(getnormalizedvector(utt), axis=0)


def extractdvectormeanvar(utt):
    # Just stack the vectors of mean and variance onto each other
    dvector = np.hstack((extractdvectormean(utt), extractdvectorvar(utt)))
    # dvector= dvector.reshape(1,dvector.shape[0])
    return dvector

methods = {
    'mean': extractdvectormean,
    'var': extractdvectorvar,
    'max': extractdvectormax,
    'meanvar': extractdvectormeanvar
}


def parse_args():
    parser = argparse.ArgumentParser(
        'Scores the enrole models against the testutterances')
    parser.add_argument(
        'inputdata', type=str, help='Input dir or a file specifying all the utterances for enrolment')
    parser.add_argument(
        'testutts', type=readFeats, help='Input dir or a file specifying the utterances')
    parser.add_argument(
        'testmlf', type=mlffile, help='test.mlf file to get the tests. Model and utterance are separated by "-"! ')
    parser.add_argument(
        'scoreoutfile', default=sys.stdout, type=argparse.FileType('w'), nargs="?", metavar="STDOUT")
    parser.add_argument('-d', '--debug', default=log.INFO, metavar='DEBUGLEVEL',
                        help="Sets the debug level. A level of 10 represents DEBUG. Higher levels are 20 = INFO (default), 30 = WARN", type=int)
    parser.add_argument('-del', '--delimiter', type=str,
                        help='If we extract the features from the given data, we use the delimiter (default : %(default)s) to obtain the splits.',
                        default="_")
    parser.add_argument(
        '-id', '--indices', help="The indices of the given splits which are used to determinate the speaker labels! default is rsr %(default)s", nargs="+", type=int, default=[0, 2])
    parser.add_argument(
        '-e', '--extractionmethod', choices=methods, default='mean', help='The method which should be used to extract dvectors, default: %(default)s'
    )
    return parser.parse_args()
args = parse_args()

extractmethod = methods[args.extractionmethod]


def parsepaths(paths):
    mdict = {}
    for path in paths:
        bname, ext = os.path.splitext(os.path.basename(path))
        mdict[bname] = extractmethod(path)
    return mdict


def main():
    inputdata = parseinputfiletomodels(
        args.inputdata, args.delimiter, args.indices)
    testtofeature = parsepaths(args.testutts)
    log.basicConfig(
        level=args.debug, format='%(asctime)s %(levelname)s %(message)s', datefmt='%d/%m %H:%M:%S')
    lda = LDA(solver='svd')
    labels = []
    dvectors = []
    log.info("Extracting dvectors for input data")
    for spk, v in inputdata.iteritems():
        dvectors.extend(itertools.imap(extractmethod, v))
        labels.extend([spk for i in xrange(len(v))])
    spktonum = {spk: num for num, spk in enumerate(np.unique(labels))}
    dvectors = np.array(dvectors)
    labelsnum = np.array([spktonum[i] for i in labels])
    log.debug("Overall we have %i labels"%(len(labelsnum)))
    log.debug("Number of speakers: %i"%(len(spktonum.keys())))
    log.debug("Dvector size is (%i,%i)" %
            (dvectors.shape[0], dvectors.shape[1]))

    log.info("Fitting LDA model")
    lda.fit(dvectors, labelsnum)

    errors = 0
    log.info("Starting test")
    for enrolemodel, vals in args.testmlf.iteritems():
        if enrolemodel not in spktonum:
            errors += 1
            log.warn("Enrolemodel %s not found in the labels" % (enrolemodel))
            continue
        curspk = spktonum[enrolemodel]
        for testutt, targetmdl in vals:
            if testutt not in testtofeature:
                log.warn("\nUtterance %s not found in the testset" % (testutt))
                errors += 1
                continue
            testdvector = testtofeature[testutt][np.newaxis, :]
            score = lda.predict_log_proba(testdvector)[0]
            finalscore = score[curspk]
            args.scoreoutfile.write(
                "{} {}-{} {:.3f}\n".format(targetmdl, enrolemodel, testutt, finalscore))
    if errors > 0:
        log.warn(
            "Overall %i happened while processing the testutterances. The scores may not be complete" % (errors))
    log.info("LDA estimation done, output file is: %s. Output file has the following structure: TARGETMODEL ENROLEMODEL-TESTUTT" %
             (args.scoreoutfile.name))

if __name__ == '__main__':
    main()
