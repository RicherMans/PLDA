# PLDA
An LDA/PLDA estimator using KALDI in python for speaker verification tasks

## Installation ##

Make sure that you have KALDI compiled and installed. Further make sure that KALDI was compiled using the option --shared, during ./configure (e.g. ./configure --shared).
Moreover the included ATLAS within KALDI is sufficient that PLDA works. If any compilation errors happen it's most likely that not all of the ATLAS libraries was installed successfully.

Moreover to find KALDI correctly, please run:

```bash
export KALDI_ROOT=/your/path/to/root
```

if your ATLAS is installed in a different directory please set the variable ATLAS_DIR e.g.

```bash
export ATLAS_DIR=/your/atlas/dir
```

Then just run:
```bash
git clone https://github.com/RicherMans/PLDA
cd PLDA
mkdir build && cd build && cmake ../ && make
```

Per default cmake is installing the python package into your /usr/lib directory. If this is not wised, pass the option -DUSER=ON to cmake to install the packages only for the current user

Voila, the python library is copied to your local users installation path.

## Usage ##

Generally we use this script to do LDA/PLDA scoring. First we need to fit a model using LDA/PLDA.

For LDA:
```python
from liblda import LDA
lda = LDA()
n_samples=500, featdim = 200
X=np.random.rand(n_samples,featdim)
Y=np.random.randint(0,2,n_samples).astype('uint') # for binary labels

lda.fit(X,Y)
```

For PLDA:
```python
from liblda import PLDA
plda = PLDA()

n_samples=500, featdim = 200

X=np.random.rand(n_samples,featdim)
Y=np.random.randint(0,2,n_samples).astype('uint') # for binary labels

plda.fit(X,Y)
```
Note that fitting the model in the LDA case is done using enrolment data, while for PLDA we use background data ( which can be any data).

PLDA fit does also accept one extra argument:

```python
#Transform the features first to a given target dimension. Default is keeping the dimension
targetdim=10
plda.fit(X,Y,targetdim)
```

LDA can then after fitting be used to directly score any incoming utterance using predict_log_proba(SAMPLE)

```python
pred = np.random.rand(featdim)[np.newaxis,...]
scores = lda.predict_log_proba(pred)
```
the predict_log_proba method returns a list where each element in the last represents the likelihood for the indexed class.

For PLDA one can also do standard normalization methods such as z-norm (other norms are not implemented yet). For this case, simply transform your enrolment vectors (labeled as ENROL_X,ENROL_Y) into the PLDA space and then normalize them using any other data ( but do not use the background data from .fit() ).
Generally it is recommended to have an held out set to do this estimation. The normalization procedure will then estimate the mean and variance of the scores of the enrolment models against the held out set (Otherdata).

```python
ENROL_X=np.random.rand(n_samples,featdim)
ENROL_Y=np.arange(n_samples,dtype='uint')
#Smoothing factor does increase the performance. Its a value between 0 and 1. #Does affect the covariance matrix. Optional!
smoothing=0.5

#Transform the features first to a given target dimension. Default is keeping the dimension
targetdim=10

transformed_vectors = plda.transform(ENROL_X,ENROL_Y,targetdim,smoothing)

Otherdata = np.random.rand(m_samples,featdim)
plda.norm(Otherdata,transformed_vectors)
```
Note that if targetdim is given, all future plda.transform() calls also need targetdim as a valid parameter

And finally one can score any model against a utterance by:

```python
Models_X=np.random.rand(n_samples,featdim)
Models_Y=np.arange(n_samples,dtype='uint')
transformed_vectors = plda.transform(Models_X,Models_Y)

testutt_x=np.random.rand(n_samples,featdim)
testutt_y=np.arange(n_samples,dtype='uint')

transformedtest_vectors=plda.transform(testutt_x,testutt_y)

for model,modelvec in transformed_vectors.iteritems():
  for testutt,testvec in transformedtest_vectors.iteritems():
    #model is an integer which represents the current class, need for normalization
    #modelvec is a tuple consisting of (samplesize,datavector)
    #testvec is a tuple consisting of (samplesize,datavector)
    score=plda.score(model,modelvec,testvec)
```
Note that the modelid is necessary only if one wants to normalize using z-norm.
