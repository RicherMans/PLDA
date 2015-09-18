#include <Python.h>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>
#include <algorithm>
//numpy library
#include "numpy/arrayobject.h"
#include <cassert>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

#include "ivector/plda.h"
#include "kaldi-utils.hpp"

namespace kaldi{


    static PyObject* py_fitplda(PyObject* self,PyObject *args){

        //From the given data as features with dimensions (nsamples,featdim ) and labels as (nsamples,)
        //we estimate the statistics.
        //The labels values indicate which label is given for each sample in nsamples
        PyArrayObject* py_inputfeats;
        PyArrayObject* py_labels;
        std::string outputfile;
        if (! PyArg_ParseTuple( args, "O!O!s", &PyArray_Type,&py_inputfeats,&PyArray_Type,&py_labels )) return NULL;

        auto n_samples=py_inputfeats->dimensions[0];
        auto featdim =py_inputfeats->dimensions[1];

        assert(py_labels->dimensions[0]==py_inputfeats->dimensions[0]);

        const Matrix<BaseFloat> &inputfeats = pyarraytomatrix<double>(py_inputfeats);

        long *labels = pyvector_to_type<long>(py_labels);

        for (auto i = 0u; i < n_samples; i++) {
            std::cerr << labels[i] << std::endl;
        }

        PldaStats plda_stats;
    }

    static PyMethodDef libpldaModule_methods[]={
        {"fitplda",py_fitplda,METH_VARARGS},
        {NULL,NULL,0,NULL}
    };

    extern "C" void initlibplda()
    {
        (void) Py_InitModule("libplda",libpldaModule_methods);

        import_array();
    }

}
