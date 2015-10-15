#include <Python.h>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>
#include <algorithm>
#include <unordered_map>
//numpy library
#include "numpy/arrayobject.h"
// T_INT and all other datatypes for the class
#include "structmember.h"
#include <cassert>



#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

#include "chtk.h"
#include "ivector/plda.h"
#include "kaldi-utils.hpp"

namespace kaldi{


    typedef struct {
        PyObject_HEAD
        Plda plda;
        PldaEstimationConfig estconfig;
        PldaConfig config;
        std::unordered_map<long,double> meanz,stdvz;
    } MPlda;

    struct Stats{
        uint32_t size;
        Vector<double> data;
    };


    static PyObject * MPlda_fit(MPlda* self, PyObject *args, PyObject * kwds){
        SetVerboseLevel(0);
        //From the given data as features with dimensions (nsamples,featdim ) and labels as (nsamples,)
        //we estimate the statistics.
        //The labels values indicate which label is given for each sample in nsamples
        PyArrayObject* py_inputfeats;
        PyArrayObject* py_labels;
        // Default number of iterations is 10
        uint32_t iters=10;
        if (! PyArg_ParseTuple( args, "O!O!|O!O!k", &PyArray_Type,&py_inputfeats,&PyArray_Type,&py_labels,&iters)) return NULL;

        auto n_samples=py_inputfeats->dimensions[0];
        auto featdim =py_inputfeats->dimensions[1];

        assert(py_labels->dimensions[0]==py_inputfeats->dimensions[0]);

        PldaStats stats;

        const Matrix<double> &inputfeats = pyarraytomatrix<double>(py_inputfeats);

        const long *labels = pyvector_to_type<long>(py_labels);

        std::set<long> u_labels;
        for (auto sample = 0u; sample < n_samples; ++sample) {
            u_labels.insert(labels[sample]);
        }
        auto num_speakers = u_labels.size();
        std::vector<MatrixIndexT> indices[num_speakers];
        for(auto n=0u ; n < n_samples;n++){
            indices[labels[n]].push_back(n);
        }

        for(auto spk=0u; spk < num_speakers;spk ++){
            Matrix<double> tmp(indices[spk].size(),featdim);
            tmp.CopyRows(inputfeats,indices[spk]);
            stats.AddSamples(1.0/indices[spk].size(),tmp);
        }

        stats.Sort();

        self->estconfig.num_em_iters = iters;
        PldaEstimator estimator(stats);

        // Plda plda;
        estimator.Estimate(self->estconfig,&(self->plda));
        return Py_BuildValue("");
    }

    static PyObject* Mplda_transform(MPlda* self,PyObject* args,PyObject* kwads){

        float smoothfactor=1.0;
        PyArrayObject* py_inpututts;
        PyArrayObject* py_labels;
        uint32_t targetdim = 0;
        if (! PyArg_ParseTuple( args, "O!O!|O!O!(k|f)", &PyArray_Type,&py_inpututts,&PyArray_Type,&py_labels,&targetdim,&smoothfactor)) return NULL;
        std::map<uint32_t,Stats> speakertoutts;
        PyObject *retdict = PyDict_New();

        const Matrix<double> &inputfeats = pyarraytomatrix<double>(py_inpututts);
        auto n_samples=py_inpututts->dimensions[0];
        auto featdim =py_inpututts->dimensions[1];
        // If targetdim is given, we transform the vectors to targetdim
        if (targetdim>0){
            featdim = targetdim;
        }
        // Labels are strings!
        if (py_labels->descr->kind == 'S'){
            std::string err("Labels need to be numpy array of ints, not strings!");
            return PyErr_Format(PyExc_ValueError,err.c_str());
        }
        assert(py_labels->dimensions[0]==py_inpututts->dimensions[0]);

        long *labels = pyvector_to_type<long>(py_labels);

        // Get the unique labels in the number of labels, which are the speakers
        std::set<long> u_labels;
        for (auto sample = 0u; sample < n_samples; ++sample) {
            u_labels.insert(labels[sample]);
        }

        for(auto spk = 0u; spk < n_samples;spk++){
            if(speakertoutts.count(labels[spk])==0){
                Stats stat;
                stat.size=0;
                stat.data.Resize(featdim);
                speakertoutts.insert(std::make_pair(labels[spk],stat));
            }
            speakertoutts[labels[spk]].size += 1;
            speakertoutts[labels[spk]].data.AddVec(1.0,inputfeats.Row(spk));
        }
        // Smooth within class covariance if smooth factor is given
        if (smoothfactor != 1.0){
            self->plda.SmoothWithinClassCovariance(smoothfactor);
        }

        for(std::map<uint32_t,Stats>::iterator it=speakertoutts.begin();it!=speakertoutts.end();it++){
            auto samplesize=it->second.size;
            assert(samplesize>0);
            // Scale the vector by its samplesize to normalize it
            it->second.data.Scale(1.0/samplesize);
            // Need to allocate on heap otherwise we can't return it to python ( will be deleted after this function ends)
            Vector<double> *transformed=new Vector<double>(featdim);
            self->plda.TransformIvector(self->config,it->second.data,transformed);
            // We use the labels given in the labels array
            PyObject* spkid = PyInt_FromSize_t(it->first);
            npy_intp dimensions[1]  = {transformed->Dim()};
            PyArrayObject* py_transformed = (PyArrayObject* )PyArray_SimpleNewFromData(1,dimensions,NPY_DOUBLE,transformed->Data());
            // Let python free the memory of this Vector if necessary
            py_transformed->flags |= NPY_ARRAY_OWNDATA;
            PyObject* tup = PyTuple_New(2);
            PyObject* py_samplesize = PyInt_FromSize_t(samplesize);
            Py_INCREF(py_samplesize);
            Py_INCREF(spkid);
            PyTuple_SetItem(tup,0,py_samplesize);
            PyTuple_SetItem(tup,1,(PyObject*) py_transformed);
            // Set the value in the map as the transformed value
            PyDict_SetItem(retdict,spkid,tup);
            //Decrement all non returned variables
            Py_DECREF(py_samplesize);
            Py_DECREF(spkid);
            Py_DECREF(tup);
        }
        return Py_BuildValue("O",retdict);


    }

    static PyObject* MPlda_norm(MPlda* self,PyObject* args,PyObject *kwargs){
        PyArrayObject* py_bkgdata;
        PyObject* py_spktoutt;
        uint32_t numutts=0;
        if(!PyArg_ParseTuple(args,"O!O!|O!O!k",&PyArray_Type,&py_bkgdata,&PyDict_Type,&py_spktoutt,&numutts)) return NULL;

        const Matrix<double> &bkgdata = pyarraytomatrix<double>(py_bkgdata);
        // matrows represent every row in the matrix bkgdata
        std::vector<uint32_t> matrows(bkgdata.NumRows());

        //We want to shuffle the indices of the matrix
        std::iota(matrows.begin(),matrows.end(),0);

        std::random_shuffle(matrows.begin(),matrows.end());

        if (numutts==0){
            numutts = bkgdata.NumRows();
        }
        // Accumulated scores for every given speakermodel
        std::unordered_map<long,std::vector<double>> scores;

        // Workaround, somehow there is a bug when inserting the first item in the global variable, no clue why
        self->meanz.reserve(numutts);
        self->stdvz.reserve(numutts);
        for (auto i=0u; i < numutts; ++i) {
            auto rowindex = matrows[i];
            Vector<double> transformed(self->plda.Dim());
            self->plda.TransformIvector(self->config,bkgdata.Row(rowindex),&transformed);

            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while(PyDict_Next(py_spktoutt, &pos, &key, &value)){
                if(! PyInt_Check(key)) return NULL;
                if(! PyTuple_Check(value)) return NULL;

                long k = PyInt_AsLong(key);
                // The values are a tuple of (samplesize,DATA), here we dont need the samplesize
                const Vector<double> &repr = pyarraytovector<double>((PyArrayObject* )PyTuple_GetItem(value,1));
                double score = self->plda.LogLikelihoodRatio(transformed,1,repr);
                scores[k].push_back(score);
            }
        }
        for(std::unordered_map<long,std::vector<double> >::const_iterator it=scores.begin();it!=scores.end();it++){
            double sum = std::accumulate( it->second.begin(), it->second.end(), 0.0);
            assert(it->second.size()>0);
            double mean = sum/it->second.size();
            self->meanz[it->first] = std::move(mean);
            // self->meanz.insert(std::make_pair(it->first,mean));
            double sqsum = std::accumulate(it->second.begin(),it->second.end(),0.0,[&](const double &a,const double &b){
                    return a+((b-mean) * (b-mean));
                    });
            sqsum /= it->second.size();
            self->stdvz.insert(std::make_pair(it->first,sqrt(sqsum)));
        }

    }

    static PyObject* MPlda_score(MPlda * self, PyObject* args, PyObject* kwds){
        PyObject* py_enrolemodel;
        PyObject* py_testutt;
        uint32_t enrolemodelid=-1;
        if(!PyArg_ParseTuple(args,"kO!O!",&enrolemodelid,&PyTuple_Type,&py_enrolemodel,&PyTuple_Type,&py_testutt)) return NULL;
        long samplesize = PyInt_AsLong(PyTuple_GetItem(py_enrolemodel,0));
        const Vector<double> &enrolemodel = pyarraytovector<double>((PyArrayObject *)PyTuple_GetItem(py_enrolemodel,1));
        const Vector<double> &testutt = pyarraytovector<double>((PyArrayObject *)PyTuple_GetItem(py_testutt,1));
        double score = self->plda.LogLikelihoodRatio(enrolemodel,samplesize,testutt);

        // Cant normalize if we have not seen this enrolemodel
        if (self->meanz.size()==0 || self->meanz.count(enrolemodelid)==0){
            return Py_BuildValue("f",score);
        }
        // Do t-z norm
        score = (score - (self->meanz[enrolemodelid]))/self->stdvz[enrolemodelid];
        return Py_BuildValue("f",score);
    }


    static PyMethodDef Plda_methods[] = {
        {"fit", (PyCFunction)MPlda_fit, METH_VARARGS,
         "Fit the model with parameters X,Y, numiters\n"
         "X = (nsamples,featdim) and Y = (nsamples,) , numiters = number of iterations for PLDA estimation"
        },
        {"transform",(PyCFunction)Mplda_transform,METH_VARARGS,
        "Transforms the given parameters X into the PLDA subspace. "
        },
        {"norm",(PyCFunction)MPlda_norm,METH_VARARGS,
        "Does Normalization on the scores, using Z norm currently"
        },
        {"score",(PyCFunction)MPlda_score,METH_VARARGS,
        "Scores a given enrolement model, which was adapted using transform against an test utterance. Both enrolment and test utterance need to be adapted first"
        },
        {NULL}  /* Sentinel */
    };

    static PyObject *
    Plda_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {
        MPlda *self;

        self = (MPlda *)type->tp_alloc(type, 0);

        return (PyObject *)self;
    }

    static void MPLDA_dealloc(MPlda* self)
    {
        self->ob_type->tp_free((PyObject*)self);
    }


    static PyTypeObject MPlda_Type = {
        PyObject_HEAD_INIT(NULL)
        0,                         /*ob_size*/
        "liblda.Plda",             /*tp_name*/
        sizeof(MPlda),             /*tp_basicsize*/
        0,                         /*tp_itemsize*/
        (destructor)MPLDA_dealloc,                         /*tp_dealloc*/
        0,                         /*tp_print*/
        0,                         /*tp_getattr*/
        0,                         /*tp_setattr*/
        0,                         /*tp_compare*/
        0,                         /*tp_repr*/
        0,                         /*tp_as_number*/
        0,                         /*tp_as_sequence*/
        0,                         /*tp_as_mapping*/
        0,                         /*tp_hash */
        0,                         /*tp_call*/
        0,                         /*tp_str*/
        0,                         /*tp_getattro*/
        0,                         /*tp_setattro*/
        0,                         /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        "Plda object",           /* tp_doc */
        0,                     /* tp_traverse */
        0,                     /* tp_clear */
        0,                     /* tp_richcompare */
        0,                     /* tp_weaklistoffset */
        0,                     /* tp_iter */
        0,                     /* tp_iternext */
        Plda_methods,             /* tp_methods */
        0,                /* tp_members */
        0,           /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        0,      /* tp_init */
        0,                         /* tp_alloc */
        Plda_new,                 /* tp_new */
    };


    static PyMethodDef libpldaModule_methods1[] = {
        {NULL}
    };


    extern "C" {
        #ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
        #define PyMODINIT_FUNC void
        #endif
        PyMODINIT_FUNC initlibplda(){
        // (void) Py_InitModule("libplda",libpldaModule_methods);
        if (PyType_Ready(&MPlda_Type) < 0)
            return;
        MPlda_Type.tp_new = PyType_GenericNew;
        PyObject *m = Py_InitModule3("libplda", libpldaModule_methods1,
                       "Example module that creates an extension type.");
        if (m == NULL)
            return;
        import_array();

        Py_INCREF(&MPlda_Type);
        PyModule_AddObject(m, "MPlda", (PyObject *)&MPlda_Type);

        }
    }

}
