# PLDA
An LDA/PLDA estimator using KALDI in python for speaker verification tasks

## Installation ##

Make sure that you have KALDI compiled and installed. Further make sure that KALDI was compiled using the shared option --shared. 
Then just run:
```bash
git clone https://github.com/RicherMans/PLDA
cd PLDA
mkdir build && cd build **&& cmake ../ && make
```

Voila, the python library is copied to your local users installation path.

## Usage ##

Generally we use this script to do LDA/PLDA scoring. First we need to fit a model using LDA/PLDA.

For LDA:
```bash
from liblda import LDA
lda = LDA()
X=np.random.rand(n_samples,featdim)
Y=np.array(n_samples)

lda.fit(X,Y)

```

For PLDA:
```bash
from liblda import PLDA
plda = PLDA()

X=np.random.rand(n_samples,featdim)
Y=np.array(n_samples)

plda.fit(X,Y)
```

LDA can then after fitting be used to directly score any incoming utterance using predict_log_prob(SAMPLE)

```bash
pred = np.random.rand(featdim)
scores = lda.predict_log_prob(pred)
```
the predict_log_prob method returns a list where each element in the last represents the likelihood for the indexced class.

For PLDA one can also do standard normalization methods such as z-norm

```bash
X_znorm=np.random.rand(n_samples,featdim)
Y_znorm=np.array(n_samples)
plda.norm(X_znorm,Y_znorm)
```

And finally one can score any model against a utterance by:

```bash
model = np.random.rand(featdim)
modelid = 1
testval = np.random.rand(featdim)
plda.score(model,modelid,testval)
```
