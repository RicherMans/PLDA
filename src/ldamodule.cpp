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
#include <cassert>
//Also imports the KALDI headers
#include "LDA.hpp"

namespace kaldi{

template<typename T>
T* pyvector_to_type(PyArrayObject* arrayin){
    return (T*) arrayin->data;
}

template <std::size_t N>
struct type_of_size
{
    typedef char type[N];
};

template<typename T,std::size_t Size>
typename type_of_size<Size>::type& sizeof_array_help(T(&)[Size]);

//Define the sizeof function as a constant expression. We do not need to do this neceessarily
#define sizeof_array(arr) sizeof(sizeof_array_help(arr))


Matrix<BaseFloat> readFeatureFromChar(const char* featurefilename){
     try{
        //Reads in the feature from the given file. We do not extend the feature ( hence the paremter 0 )
        chtk::htkarray feature = chtk::htk_load(std::string(featurefilename),0);
        //Samplesize is in bytes, so we need to divide it by 4 to get the actual dimension
        auto featdim = feature.samplesize/4;
        //feat is just an temporarary vector , which is refilled for every sample
        Vector<BaseFloat> feat(featdim);
        std::vector<float> nsamples= feature.as_vec<float>();
        Matrix<BaseFloat> mat(nsamples.size()/featdim,featdim);
        //Copy the data of every sample into the Vector so that Kaldi can use it
        for (auto k = 0u; k < nsamples.size(); k+=featdim) {
            std::copy(nsamples.begin()+k,nsamples.begin()+k+featdim,feat.Data());
            mat.CopyRowFromVec(feat,k/featdim);
        }
        return mat;
    }catch(const std::exception &e){
        std::cerr << e.what();
        throw e;
    }

}
Matrix<BaseFloat> readFeatureFromPyString(PyObject* pyfeaturefilename){
    try{
        PyString_Check(pyfeaturefilename);
        const char* featurefilename = PyString_AsString(pyfeaturefilename);
        //Reads in the feature from the given file. We do not extend the feature ( hence the paremter 0 )
        return readFeatureFromChar(featurefilename);
    }catch(const std::exception &e){
        std::cerr << e.what();
        throw e;
    }
}


template<typename T>
void k_matrix_to_array(const MatrixBase<T>& inpmat,T* retarr){
    //Copies the vector of inpmat into the array retarr. Please verify that the sizes match!
    for (auto i = 0; i < inpmat.NumRows(); ++i) {
        auto curind = i*inpmat.NumCols();
        std::copy(inpmat.RowData(i),inpmat.RowData(i)+inpmat.NumCols(),retarr+curind);
    }
}
//Packed matrix overload, packed matrix are triangular matrices
template<typename T>
void k_matrix_to_array(const PackedMatrix<T>& inpmat,T* retarr){
    //Copies the vector of inpmat into the array retarr. Please verify that the sizes match!
    const T* rawdata = inpmat.Data();
    //THe number of elements in the lower diagonal
    int32 n_elements = inpmat.NumCols()*(inpmat.NumCols()+1)/2;
    //ind is the index of the maximum element of the current row
    auto ind = 0u;
    //the current row index, counts throught the rows
    auto rowind = 0u;
    //the index where the last row did start
    auto laststartind = 0u;
    auto colind = 0u;
    for (auto k = 0; k < n_elements; ++k) {
        if (k>ind){
            ind = (2*k)-laststartind;
            //std::cerr << ind << " " << " " << k << std::endl;
            laststartind = k;
            rowind +=1;
            colind = 0;
        }

        retarr[rowind*inpmat.NumCols()+colind] = rawdata[k];
        retarr[colind*inpmat.NumCols()+rowind] = rawdata[k];
        colind += 1;
    }
}

template<typename T>
void k_vector_to_array(const VectorBase<T>& inputvec,T* retarr){
    const T* rawdata = inputvec.Data();

    auto n_elements=inputvec.Dim();
    std::copy(rawdata,rawdata+n_elements,retarr);


}


template<typename T>
Matrix<BaseFloat> pyarraytomatrix(PyArrayObject* pytrans){
    auto dim1 = pytrans->dimensions[0];
    auto dim2 = pytrans->dimensions[1];
    Matrix<BaseFloat> trans(dim1,dim2);
    T *arr = pyvector_to_type<T>(pytrans);
    for (auto i = 0; i < dim1; i++) {
        auto beginind = i*dim2;
        auto endind = (i+1)*dim2;
        std::copy(arr+beginind,arr+endind,trans.RowData(i));
    }
    return trans;
}



template<typename T>
Vector<T> pyarraytovector(PyArrayObject* pytrans){
    auto dim1 = pytrans->dimensions[0];
    Vector<T> out(dim1);
    T *arr = pyvector_to_type<T>(pytrans);
    std::copy(arr,arr+dim1,out.Data());
    return out;
}

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
static PyObject* py_estimate(PyObject* self,PyObject* args){

    int targetdim;
    if (! PyArg_ParseTuple( args, "i",&targetdim )) return NULL;

    LDA *lda = LDA::getInstance();
    //Accumulation finsihed, now we process the transformation matrix
    LdaEstimateOptions opts;
    opts.dim = targetdim;
    //the lda transformation matrix, it can be used to do dimensionality reduction
    Matrix<BaseFloat> lda_mat;
    lda->estimate(opts,&lda_mat);
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


static PyObject* py_fitldafromdata(PyObject* self,PyObject* args){
    //From the given data as features with dimensions (nsamples,featdim ) and labels as (nsamples,)
    //we estimate the statistics.
    //The labels values indicate which label is given for each sample in nsamples
    PyArrayObject* py_inputfeats;
    PyArrayObject* py_labels;

    if (! PyArg_ParseTuple( args, "O!O!", &PyArray_Type,&py_inputfeats,&PyArray_Type,&py_labels)) return NULL;

    auto n_samples=py_inputfeats->dimensions[0];
    auto featdim =py_inputfeats->dimensions[1];

    assert(py_labels->dimensions[0]==py_inputfeats->dimensions[0]);

    const Matrix<BaseFloat> &inputfeats = pyarraytomatrix<double>(py_inputfeats);

    long *labels = pyvector_to_type<long>(py_labels);
    std::set<long> u_labels;
    for (auto sample = 0u; sample < n_samples; ++sample) {
        u_labels.insert(labels[sample]);
    }
    auto num_speakers = u_labels.size();

    LDA* lda = LDA::getInstance();
    //Init the LDA model, with the number of speakers as classes
    lda->init(num_speakers,inputfeats.NumCols());
    for (auto samplenum = 0; samplenum < n_samples;samplenum++) {
        SubVector<BaseFloat> feat(inputfeats,samplenum);
        //Accumulate the feature and the corresponding label
        lda->accumulate(feat,labels[samplenum]);
        PyErr_CheckSignals();
    }

    return Py_BuildValue("");
}

static PyObject* py_fitlda(PyObject* self,PyObject* args,PyObject* kwargs){
    bool estimate_transform;

    char* kwds[] ={
        "filelist",
        "targetdim",
        "transform",
        NULL
    };
    PyObject* dict;
    //Target dimension which is passed in the args
    int targetdim;
    /* the O! parses for a Python object (listObj) checked
     *    to be of type PyList_Type */
    if (! PyArg_ParseTupleAndKeywords( args,kwargs, "O!i|i", kwds,&PyDict_Type, &dict,&targetdim,&estimate_transform)) return NULL;
    //Check if the given argument is really a dict
    PyDict_Check(dict);

    auto numspeakers = PyDict_Size(dict);
    //equal to dict.items()
    PyObject* items = PyDict_Items(dict);
    //Variable to check if the sizes of the utterance for every key are consistent
    auto check_utts=0;
    LDA *lda = LDA::getInstance();
    //The bins used to estimate the priors
    //The problem is that we cannot use stack allocated arrays, since they get destoryed after the function returns;
    std::vector<uint32_t> *bins = new std::vector<uint32_t>;
    for (auto spkid = 0u; spkid < numspeakers; ++spkid) {
        PyObject* item = PyList_GetItem(items,spkid);
        PyObject* spk=PyTuple_GetItem(item,0);
        PyObject* values=PyTuple_GetItem(item,1);
        PyList_Check(values);
        auto num_utts=PyList_Size(values);
        if (num_utts < 0){
            std::string err("Values need to be a list of features(string)!");
            return PyErr_Format(PyExc_ValueError,err.c_str());
        }
        //We assume that we have key value pairs, where the values are a list of strings representing
        //The utterances for the speaker
        for (auto j = 0; j < num_utts; ++j) {
            PyObject *utt = PyList_GetItem(values,j);
            //Feats stores in row major its data. In every row there is one feature vector.
            const Matrix<BaseFloat> &feats = readFeatureFromPyString(utt);
            //init lda at the first iteration
            if(spkid==0){
                lda->init(numspeakers,feats.NumCols());
            }
            for (auto matind = 0; matind < feats.NumRows(); ++matind) {
                SubVector<BaseFloat> feat(feats,matind);
                //accumulate the LDA statistics for later use
                lda->accumulate(feat,spkid);
                //We return for every speakers sample the corresponding bin
                bins->push_back(spkid);
            }
        }
        //Cancel the operation of keyboardinterrupt is called
        PyErr_CheckSignals();
    }

    //The result which we are going to return. Its a numpy array with lda_mat dimensions
    PyArrayObject* lda_result_mat;
    if (estimate_transform ==true){
        //Accumulation finsihed, now we process the transformation matrix
        LdaEstimateOptions opts;
        opts.dim = targetdim;
        //the lda transformation matrix, it can be used to do dimensionality reduction
        Matrix<BaseFloat> lda_mat;
        lda->estimate(opts,&lda_mat);
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
        lda_result_mat = (PyArrayObject* )PyArray_SimpleNewFromData(2,dimensions,NPY_FLOAT,retarr);
        //Usually python does only store a reference on the given pointer and does not own the data
        //With this flag, we tell him to own the data and deallocate the data with the PyObject
        lda_result_mat->flags |= NPY_ARRAY_OWNDATA;
    }
    else{
//TODO:FIX THAT, currently we just return an empty array when the function finishes
        npy_intp dims[1]={};
        lda_result_mat= (PyArrayObject*)PyArray_EMPTY(1,dims,NPY_INT,0);
    }
    npy_intp bins_dims[1] = {bins->size()};
    PyArrayObject* bins_result = (PyArrayObject* )PyArray_SimpleNewFromData(1,bins_dims,NPY_UINT32,bins->data());
    bins_result->flags |= NPY_ARRAY_OWNDATA;

    return Py_BuildValue("(OO)",lda_result_mat,bins_result);
}


static PyObject* py_predictldafromarray(PyObject* self,PyObject* args){
    PyArrayObject *pytrans;
    PyArrayObject* pyinputfeat;

    if (! PyArg_ParseTuple( args, "O!O!", &PyArray_Type,&pyinputfeat,&PyArray_Type,&pytrans)) return NULL;

    const Matrix<BaseFloat>& trans = pyarraytomatrix<float>(pytrans);
    const Matrix<BaseFloat>& feat = pyarraytomatrix<double>(pyinputfeat);


    const Matrix<BaseFloat>& feat_out = transform(feat,trans);

    //std::cerr << feat_out << std::endl;
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


static PyObject* py_predictldafromutterance(PyObject* self, PyObject* args){
    const char* featurefilename;
    //pytrans is the transition matrix, stored as a numpy array
    PyArrayObject* pytrans;
    if (! PyArg_ParseTuple( args, "sO!", &featurefilename,&PyArray_Type,&pytrans)) return NULL;


    const Matrix<BaseFloat> &trans = pyarraytomatrix<float>(pytrans);

    const Matrix<BaseFloat> &feat = readFeatureFromChar(featurefilename);
    const Matrix<BaseFloat> &feat_out = transform(feat,trans);
    PyErr_CheckSignals();
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
static PyObject* py_getstats(PyObject* self,PyObject* args){

    LDA* lda = LDA::getInstance();
    double count;
    SpMatrix<double> total_covar, bc_covar;
    Vector<double> total_mean;
    //gets the statistics from the accumulated lda inputs
    lda->getstats(&total_covar, &bc_covar, &total_mean, &count);

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


static PyObject* py_getclassmean(PyObject* self,PyObject* args){
    LDA *lda = LDA::getInstance();
    Matrix<double> classmeans;
    lda->getclassmean(&classmeans);

    double *classmeans_ret = new double[classmeans.NumRows()*classmeans.NumCols()];

    k_matrix_to_array(classmeans,classmeans_ret);

    npy_intp dimensions_mean[2] = {classmeans.NumRows(),classmeans.NumCols()};

    PyArrayObject* class_mean = (PyArrayObject*) PyArray_SimpleNewFromData(2,dimensions_mean,NPY_DOUBLE,classmeans_ret);

    class_mean->flags |= NPY_ARRAY_OWNDATA;
    return  PyArray_Return(class_mean);

}



/*
 *  * Bind Python function names to our C functions
 *   */
static PyMethodDef libldaModule_methods[] = {
    {"fitlda",(PyCFunction)py_fitlda,METH_VARARGS|METH_KEYWORDS,"filelist,targetdim,transform are the parameters!"},
    {"fitldafromdata",py_fitldafromdata,METH_VARARGS},
    {"predictldafromutterance",py_predictldafromutterance,METH_VARARGS},
    {"predictldafromarray",py_predictldafromarray,METH_VARARGS},
    {"getstats",py_getstats,METH_VARARGS},
    {"getclassmeans",py_getclassmean,METH_VARARGS},
    {"estimate",py_estimate,METH_VARARGS},
    {NULL, NULL,0,NULL}
};

/*
 *  * Python calls this to let us initialize our module
 *   */
extern "C" void initliblda()
{
    (void) Py_InitModule("liblda", libldaModule_methods);
    //For the numpy code, import_array needs to be called otherwise all functions are not defined of nump
    //leading to a segfault whenever any is called
    import_array();
}
}
