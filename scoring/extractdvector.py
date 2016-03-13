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

import sys
import argparse
from collections import defaultdict
import itertools
from marshal import dump


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
    # Average over the samples
    return np.max(getnormalizedvector(utt), axis=0)


def extractdvectormean(utt):
    # Average over the samples
    return np.mean(getnormalizedvector(utt), axis=0)


def extractdvectorvar(utt):
    # Average over the saples over the feature dim
    # normalized has dimensions (n_samples,featdim)
    # We Do not use np.diag(np.cov()), because somehow memory overflows with it
    return np.var(getnormalizedvector(utt), axis=0)


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


def readFeats(value):

    if os.path.isfile(value):
        return open(value, 'r').read().splitlines()
    else:
        return readDir(value)


def parseinputfiletomodels(filepath, delim, ids):
    '''
    Function: parseinputfiletomodels
    Summary: Parses the given filepath into a dict of speakers and its uttrances
    Examples: parseinputfiletomodels('bkg.scp','_',[0,2])
    Attributes:
        @param (filepath):Path to the dataset. Dataset consists of absolute files
        @param (delim):Delimited in how to extract the speaker from the filename
        @param (ids):After splitting the filename using delim, the indices which parts are taken to be the speaker. If none is given we take the whole utterance
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
        if ids:
            speakerid = delim.join([splits[id] for id in ids])
        else:
            speakerid = delim.join(splits)
        speakertoutts[speakerid].append(line)
    return speakertoutts

methods = {
    'mean': extractdvectormean,
    'max': extractdvectormax,
    'var': extractdvectorvar
}


def parse_args():
    parser = argparse.ArgumentParser(
        'Scores the enrole models against the testutterances')
    parser.add_argument('inputdata', type=str,
                        help="The data which is used to extract dvectors from. Also a folder can be passed!")
    parser.add_argument(
        'outputdvectors', type=argparse.FileType('wb'), metavar="OutputFILE")
    parser.add_argument(
        '-e', '--extractionmethod', choices=methods, default='mean', help='The method which should be used to extract dvectors'
    )
    parser.add_argument('-del', '--delimiter', type=str,
                        help='If we extract the features from the given data, we use the delimiter (default : %(default)s) to obtain the splits.',
                        default="_")
    parser.add_argument('-d', '--debug', help="Sets the debug level. A value of 10 represents debug. The lower the value, the more output. Default is INFO",
                        type=int, default=log.INFO)
    return parser.parse_args()


def extractvectors(datadict, extractmethod):
    dvectors = []
    labels = []
    for spk, v in datadict.iteritems():
        dvectors.extend(itertools.imap(extractmethod, v))
        labels.extend([spk for i in xrange(len(v))])
    log.debug("After extraction, we have %i dvectors and %i labels." %
              (len(dvectors), len(labels)))
    return np.array(dvectors), np.array(labels)

def extractspktovectors(datadict,extractmethod):
    for spk, v in datadict.iteritems():
        datadict[spk] = np.array(itertools.imap(extractmethod, v))
    return datadict




def main():
    args = parse_args()
    log.basicConfig(
        level=args.debug, format='%(asctime)s %(levelname)s %(message)s', datefmt='%d/%m %H:%M:%S')

    # extract the dvectors for each utterance!
    inputdata = parseinputfiletomodels(
        args.inputdata, args.delimiter, ids=None)
    log.info("Input data consists of %i speakers." % (len(inputdata.keys())))
    extractmethod = methods[args.extractionmethod]
    log.info(
        "Extracting dvectors [%s] for the input data" % (args.extractionmethod))
    dvectors = extractspktovectors(inputdata, extractmethod)
    # Dvectors is a dict with "utt":[dvector] elements
    dump(dvectors, args.outputdvectors)

    log.info("Extraction done!")


if __name__ == '__main__':
    main()
