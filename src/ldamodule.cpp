#include <Python.h>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>
#include "chtk.h"
#include <algorithm>
#include <set>
//numpy library
#include "numpy/arrayobject.h"
#include "structmember.h"
#include <cassert>
//Also imports the KALDI headers
// #include "LDA.hpp"
#include "transform/lda-estimate.h"
#include "kaldi-utils.hpp"

namespace kaldi{

    struct LdaEst:public LdaEstimate{

        void getclassmean(Matrix<double> *out_class_mean) const{
            int32 num_class = this->NumClasses(), dim = Dim();
            out_class_mean->Resize(num_class,dim);
            out_class_mean->CopyFromMat(first_acc_);
              for (int32 c = 0; c < num_class; c++) {
                if (zero_acc_(c) != 0.0) {
                  out_class_mean->Row(c).Scale(1.0 / zero_acc_(c));
                }
            }
        }
        void getstats(SpMatrix<double> *total_covar,
                SpMatrix<double> *between_covar,
                Vector<double> *total_mean,
                double *sum){
            this->GetStats(total_covar,between_covar,total_mean,sum);
        }
    };


    struct header{

        // Need these statistics for doing LSQR solving of LDA

        void init(int32 num_classes,int32 featdim){
            // workaround to initalize the est member, since this struct is never inittialized
            // initest();
            est->Init(num_classes,featdim);
        }

        void estimate(const LdaEstimateOptions &opts,
                Matrix<BaseFloat> *M,
                Matrix<BaseFloat> *Mfull = NULL) const {
            est->Estimate(opts,M,Mfull);
        }

        void getstats(SpMatrix<double> *total_covar,
                SpMatrix<double> *between_covar,
                Vector<double> *total_mean,
                double *sum){
            est->getstats(total_covar,between_covar,total_mean,sum);
        }

        void accumulate(const VectorBase<BaseFloat> &data, int32 class_id, BaseFloat weight = 1.0){
            est->Accumulate(data,class_id,weight);
        }

        void getclassmean(Matrix<double> *out_class_mean) const{
            est->getclassmean(out_class_mean);
        }

        Matrix<BaseFloat> *transmat;

        PyObject_HEAD
        LdaEstimateOptions opts;

        LdaEst *est;
    };
    typedef struct header MLDA;

    Matrix<BaseFloat> transform(const Matrix<BaseFloat> &feat,const Matrix<BaseFloat> &trans){
        int32 transform_rows = trans.NumRows(),
              transform_cols = trans.NumCols(),
              feat_dim = feat.NumCols();
        Matrix<BaseFloat> feat_out(feat.NumRows(), transform_rows);
        if (transform_cols == feat_dim) {
            feat_out.AddMatMat(1.0, feat, kNoTrans, trans, kTrans, 0.0);
        } else if (transform_cols == feat_dim + 1) {
            // append the implicit 1.0 to the input features.
            SubMatrix<BaseFloat> linear_part(trans, 0, transform_rows, 0, feat_dim);
            feat_out.AddMatMat(1.0, feat, kNoTrans, linear_part, kTrans, 0.0);
            Vector<BaseFloat> offset(transform_rows);
            offset.CopyColFromMat(trans, feat_dim);
            feat_out.AddVecToRows(1.0, offset);
        }else{
            std::stringstream errmsg;
            errmsg << "Feature sizes do not match!\n" << "Feature dim:" <<feat_dim << "\t\tTransform dim: "<<transform_cols;
            throw std::runtime_error(errmsg.str());
        }
        return feat_out;
    }

//Estimates the transform matrix for LDA
static PyObject* py_estimate(MLDA *self, PyObject* args,PyObject* kwds){

    int targetdim;
    if (! PyArg_ParseTuple( args, "i",&targetdim )) return NULL;

    //Accumulation finsihed, now we process the transformation matrix
    LdaEstimateOptions opts;
    opts.dim = targetdim;
    //the lda transformation matrix, it can be used to do dimensionality reduction
    Matrix<BaseFloat> lda_mat;
    self->estimate(opts,&lda_mat);
    //Transformation is stored in the lda_mat variable. We return a numpy array to the python script
    npy_intp dimensions[2]={lda_mat.NumRows(),lda_mat.NumCols()};

    auto mat_size = lda_mat.NumRows() * lda_mat.NumCols();

    //This seems to be rather stupid, but the problem is that kaldi stores its arrays, which are in
    //the ->Data() pointer with an offset for the rows and cols. Therefore we cant directly copy the
    //content of kaldi, rather than we need to store the result in a new array
    float *retarr = new float[mat_size];
    for (auto i = 0; i < lda_mat.NumRows(); ++i) {
        auto curind = i*lda_mat.NumCols();
        std::copy(lda_mat.RowData(i),lda_mat.RowData(i)+lda_mat.NumCols(),retarr+curind);
    }
    //Init a new python object with 2 dimensions and the datapointer
    PyArrayObject* lda_result_mat = (PyArrayObject* )PyArray_SimpleNewFromData(2,dimensions,NPY_FLOAT,retarr);
    //Usually python does only store a reference on the given pointer and does not own the data
    //With this flag, we tell him to own the data and deallocate the data with the PyObject
    lda_result_mat->flags |= NPY_ARRAY_OWNDATA;


    return PyArray_Return(lda_result_mat);
}


static PyObject* py_fitldafromdata(MLDA *self, PyObject* args,PyObject* kwds){
//     //From the given data as features with dimensions (nsamples,featdim ) and labels as (nsamples,)
//     //we estimate the statistics.
//     //The labels values indicate which label is given for each sample in nsamples
    PyArrayObject* py_inputfeats;
    PyArrayObject* py_labels;

    if (! PyArg_ParseTuple( args, "O!O!", &PyArray_Type,&py_inputfeats,&PyArray_Type,&py_labels)) return NULL;

    auto n_samples=py_inputfeats->dimensions[0];
    auto featdim =py_inputfeats->dimensions[1];

    assert(py_labels->dimensions[0]==py_inputfeats->dimensions[0]);

    // Assume that the given data array is a double but we need to cast it to a float
    const Matrix<BaseFloat> &inputfeats = pyarraytomatrix<BaseFloat,double>(py_inputfeats);

    long *labels = pyvector_to_type<long>(py_labels);
    std::set<long> u_labels;
    for (auto sample = 0u; sample < n_samples; ++sample) {
        u_labels.insert(labels[sample]);
    }
    auto num_speakers = u_labels.size();

    //Init the LDA model, with the number of speakers as classes
    self->init(num_speakers,inputfeats.NumCols());
    for (auto samplenum = 0u; samplenum < n_samples;samplenum++) {
        SubVector<BaseFloat> feat(inputfeats,samplenum);
        //Accumulate the feature and the corresponding label
        self->accumulate(feat,labels[samplenum]);
        PyErr_CheckSignals();
    }
    return Py_BuildValue("");
}

static PyObject* py_predictldafromarray(MLDA *self, PyObject* args,PyObject* kwds){
    PyArrayObject *pytrans;
    PyArrayObject* pyinputfeat;

    if (! PyArg_ParseTuple( args, "O!O!", &PyArray_Type,&pyinputfeat,&PyArray_Type,&pytrans)) return NULL;

    const Matrix<BaseFloat>& trans = pyarraytomatrix<BaseFloat>(pytrans);
    const Matrix<BaseFloat>& feat = pyarraytomatrix<BaseFloat,double>(pyinputfeat);

    const Matrix<BaseFloat>& feat_out = transform(feat,trans);

    float *transformedarr = new float[feat_out.NumRows()*feat_out.NumCols()];
    k_matrix_to_array(feat_out,transformedarr);

    //Transformation is stored in the lda_mat variable. We return a numpy array to the python script
    npy_intp dimensions[2]={feat_out.NumRows(),feat_out.NumCols()};

    PyArrayObject* result = (PyArrayObject* )PyArray_SimpleNewFromData(2,dimensions,NPY_FLOAT,transformedarr);
    //Usually python does only store a reference on the given pointer and does not own the data
    //With this flag, we tell him to own the data and deallocate the data with the PyObject
    result->flags |= NPY_ARRAY_OWNDATA;

    return PyArray_Return(result);
}


static PyObject* py_predictldafromutterance(MLDA *self, PyObject* args, PyObject* kwgs){
    const char* featurefilename;
    //pytrans is the transition matrix, stored as a numpy array
    PyArrayObject* pytrans;
    if (! PyArg_ParseTuple( args, "sO!", &featurefilename,&PyArray_Type,&pytrans)) return NULL;


    const Matrix<BaseFloat> &trans = pyarraytomatrix<float>(pytrans);

    const Matrix<BaseFloat> &feat = readFeatureFromChar(featurefilename);
    const Matrix<BaseFloat> &feat_out = transform(feat,trans);

    float *retarr = new float[feat_out.NumRows()*feat_out.NumCols()];
    k_matrix_to_array(feat_out,retarr);

    //Transformation is stored in the lda_mat variable. We return a numpy array to the python script
    npy_intp dimensions[2]={feat_out.NumRows(),feat_out.NumCols()};

    PyArrayObject* result = (PyArrayObject* )PyArray_SimpleNewFromData(2,dimensions,NPY_FLOAT,retarr);
    //Usually python does only store a reference on the given pointer and does not own the data
    //With this flag, we tell him to own the data and deallocate the data with the PyObject
    result->flags |= NPY_ARRAY_OWNDATA;

    return PyArray_Return(result);

}

//Returns a tuple (mean,total_covar,between covar,n_samples) of all the statistics accumulated by fit()
//The mean represents the class mean
//total covar represents the covariance matrix within the classes
//Between covar represents the covariance matrix between the classes
//n_samples if the amount of samples for all data
static PyObject* py_getstats(MLDA* self,PyObject* args,PyObject* kwargs){

    double count;
    SpMatrix<double> total_covar, bc_covar;
    Vector<double> total_mean;
    //gets the statistics from the accumulated lda inputs
    self->getstats(&total_covar, &bc_covar, &total_mean, &count);

    double *total_covar_ret = new double[total_covar.NumRows()*total_covar.NumCols()];
    k_matrix_to_array(total_covar,total_covar_ret);

    double *total_mean_ret = new double[total_mean.Dim()];
    k_vector_to_array(total_mean,total_mean_ret);

    double *bc_ret = new double[bc_covar.NumRows()*bc_covar.NumCols()];
    k_matrix_to_array(bc_covar,bc_ret);

    npy_intp dimensions_covar[2] = {total_covar.NumRows(),total_covar.NumCols()};

    npy_intp dimensions_bc[2] = {bc_covar.NumRows(),bc_covar.NumCols()};

    npy_intp dimensions_mean[1] = {total_mean.Dim()};

    PyArrayObject* tot_covar = (PyArrayObject* )PyArray_SimpleNewFromData(2,dimensions_covar,NPY_DOUBLE,total_covar_ret);
    //Usually python does only store a reference on the given pointer and does not own the data
    //With this flag, we tell him to own the data and deallocate the data with the PyObject
    tot_covar->flags |= NPY_ARRAY_OWNDATA;

    PyArrayObject* tot_mean = (PyArrayObject*) PyArray_SimpleNewFromData(1,dimensions_mean,NPY_DOUBLE,total_mean_ret);

    tot_mean->flags |= NPY_ARRAY_OWNDATA;

    PyArrayObject* tot_bc = (PyArrayObject* )PyArray_SimpleNewFromData(2,dimensions_bc,NPY_DOUBLE,bc_ret);
    //Usually python does only store a reference on the given pointer and does not own the data
    //With this flag, we tell him to own the data and deallocate the data with the PyObject
    tot_bc->flags |= NPY_ARRAY_OWNDATA;

    return Py_BuildValue("(OOOd)",tot_mean,tot_covar,tot_bc,count);
}


static PyObject* py_getclassmean(MLDA *self,PyObject* args,PyObject* kwargs){

    Matrix<double> classmeans;
    self->getclassmean(&classmeans);

    double *classmeans_ret = new double[classmeans.NumRows()*classmeans.NumCols()];

    k_matrix_to_array<double>(classmeans,classmeans_ret);

    npy_intp dimensions_mean[2] = {classmeans.NumRows(),classmeans.NumCols()};

    PyArrayObject* class_mean = (PyArrayObject*) PyArray_SimpleNewFromData(2,dimensions_mean,NPY_DOUBLE,classmeans_ret);

    class_mean->flags |= NPY_ARRAY_OWNDATA;
    return  PyArray_Return(class_mean);

}

/*
 *  * Bind Python function names to our C functions
 *   */
static PyMethodDef lda_methods[] = {
    //Depreacated, this function uses the utterances given as string paths
    // {"fitlda",(PyCFunction)py_fitlda,METH_VARARGS|METH_KEYWORDS,"filelist,targetdim,transform are the parameters!"},
    {"fit", (PyCFunction)py_fitldafromdata,METH_VARARGS},
    {"predictldafromutterance",(PyCFunction)py_predictldafromutterance,METH_VARARGS},
    {"predictldafromarray",(PyCFunction)py_predictldafromarray,METH_VARARGS},
    {"_getstats",(PyCFunction)py_getstats,METH_VARARGS},
    {"_getclassmeans",(PyCFunction)py_getclassmean,METH_VARARGS},
    {"estimate",(PyCFunction)py_estimate,METH_VARARGS},
    {NULL, NULL,0,NULL}
};


static PyMethodDef ldamodule_methods[]={
    {NULL}
};

static void
    LDA_dealloc(MLDA* self)
    {
        self->ob_type->tp_free((PyObject*)self);
    }


static PyObject *
    LDA_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {
        MLDA *self;
        self = (MLDA *)type->tp_alloc(type, 0);

        return (PyObject *)self;
    }

static int
LDA_init(MLDA *self, PyObject *args, PyObject *kwds)
{
    // self->transmat = new Matrix<BaseFloat>(1,2);
    // std::cout << *(self->transmat) <<std::endl;
    self->est = new LdaEst();
    return 0;
}


static PyTypeObject MLDA_type = {
        PyObject_HEAD_INIT(NULL)
        0,                         /*ob_size*/
        "liblda.MLDA",             /*tp_name*/
        sizeof(MLDA),             /*tp_basicsize*/
        0,                         /*tp_itemsize*/
        0,                /*tp_dealloc*/
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
        "LDA object",           /* tp_doc */
        0,                     /* tp_traverse */
        0,                     /* tp_clear */
        0,                     /* tp_richcompare */
        0,                     /* tp_weaklistoffset */
        0,                     /* tp_iter */
        0,                     /* tp_iternext */
        lda_methods,             /* tp_methods */
        0,                /* tp_members */
        0,           /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)LDA_init,      /* tp_init */
        0,                         /* tp_alloc */
        LDA_new,                 /* tp_new */
    };




/*
 *  * Python calls this to let us initialize our module
 *   */
extern "C"{
#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initliblda()
{
    //For the numpy code, import_array needs to be called otherwise all functions are not defined of nump
    //leading to a segfault whenever any is called
    import_array();

    if (PyType_Ready(&MLDA_type) < 0)
        return;
    MLDA_type.tp_new = PyType_GenericNew;
    PyObject *m = Py_InitModule3("liblda", ldamodule_methods,
                   "Example module that creates an extension type.");
    if (m == NULL)
        return;

    Py_INCREF(&MLDA_type);
    PyModule_AddObject(m, "MLDA", (PyObject *)&MLDA_type);
}
}
}
