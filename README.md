# PLDA
An LDA/PLDA estimator using KALDI in python for speaker verification tasks

## Installation ##

Make sure that you have KALDI compiled and installed. Further make sure that KALDI was compiled using the shared option --shared. 
Then just run:
```bash
git clone https://github.com/RicherMans/PLDA
cd PLDA
mkdir build && cd build && cmake ../ && make
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
Note that fitting the model in the LDA case is done using enrolment data, while for PLDA we use background data ( any data).
LDA can then after fitting be used to directly score any incoming utterance using predict_log_prob(SAMPLE)

```bash
pred = np.random.rand(featdim)
scores = lda.predict_log_prob(pred)
```
the predict_log_prob method returns a list where each element in the last represents the likelihood for the indexced class.

For PLDA one can also do standard normalization methods such as z-norm (other norms are not implemented yet). For this case, simply transform your enrolment vectors (labeld as Models_X,Models_Y) into the PLDA space and then normalize them using any other data.

```python
Models_X=np.random.rand(n_samples,featdim)
Models_Y=np.arange(n_samples)
transformed_vectors = plda.transform(Models_X,Models_Y)

Otherdata = np.random.rand(m_samples,featdim)
plda.norm(Otherdata,transformed_vectors)
```

And finally one can score any model against a utterance by:

```python
Models_X=np.random.rand(n_samples,featdim)
Models_Y=np.arange(n_samples)
transformed_vectors = plda.transform(Models_X,Models_Y)

testutt_x=np.random.rand(n_samples,featdim)
testutt_y=np.arange(n_samples)

transformedtest_vectors=plda.transform(testutt_x,testutt_y)

model = transformed_vectors
modelid = 1
testval = transformedtest_vectors
plda.score(model,modelid,testval)
```
Note that the modelid is necessary only if one wants to normalize using z-norm.
