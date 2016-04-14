import numpy as np

import liblda

size = 100
featdim = 10

x = np.random.normal(size=(size, featdim))
y3 = np.array([i % 3 for i in xrange(size)])
y2 = np.array([i % 2 for i in xrange(size)])
lda3 = liblda.LDA()
lda3.fit(x, y3)
lda2 = liblda.LDA()
lda2.fit(x, y2)

t = np.random.normal(size=(size, featdim))
lda3.predict_log_proba(t)
lda2.predict_log_proba(t)
