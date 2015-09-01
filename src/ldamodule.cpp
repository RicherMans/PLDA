#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include "transform/lda-estimate.h"
#include "chtk.h"
//numpy library
#include "numpy/arrayobject.h"
#include <cassert>

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
void k_matrix_to_array(const Matrix<BaseFloat>& inpmat,T* retarr){
    //Copies the vector of inpmat into the array retarr. Please verify that the sizes match!
    for (auto i = 0; i < inpmat.NumRows(); ++i) {
        auto curind = i*inpmat.NumCols();
        std::copy(inpmat.RowData(i),inpmat.RowData(i)+inpmat.NumCols(),retarr+curind);
    }
}

Matrix<BaseFloat> pyarraytomatrix(PyArrayObject* pytrans){
    auto dim1 = pytrans->dimensions[0];
    auto dim2 = pytrans->dimensions[1];
    Matrix<BaseFloat> trans(dim1,dim2);
    float *arr = pyvector_to_type<float>(pytrans);
    for (auto i = 0; i < dim1; i++) {
        auto beginind = i*dim2;
        auto endind = (i+1)*dim2;
        std::copy(arr+beginind,arr+endind,trans.RowData(i));
    }
    return trans;
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
        std::string err("Feature sizes do not match!");
        throw std::runtime_error(err);
    }
    return feat_out;
}

static PyObject* py_fitlda(PyObject* self,PyObject* args){
    PyObject* dict;
    //Target dimension which is passed in the args
    int targetdim;
    /* the O! parses for a Python object (listObj) checked
     *    to be of type PyList_Type */
    if (! PyArg_ParseTuple( args, "O!i", &PyDict_Type, &dict,&targetdim)) return NULL;
    //Check if the given argument is really a dict
    PyDict_Check(dict);
    auto numspeakers = PyDict_Size(dict);
    //equal to dict.items()
    PyObject* items = PyDict_Items(dict);
    //Variable to check if the sizes of the utterance for every key are consistent
    auto check_utts=0;
    LdaEstimate lda;
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
            if(lda.Dim()==0){
                lda.Init(numspeakers,feats.NumCols());
            }
            for (auto matind = 0; matind < feats.NumRows(); ++matind) {
                SubVector<BaseFloat> feat(feats,matind);
                lda.Accumulate(feat,spkid);
            }
        }
        //Cancel the operation of keyboardinterrupt is called
        PyErr_CheckSignals();
    }
    //Accumulation finsihed, now we process the transformation matrix
    LdaEstimateOptions opts;
    opts.dim = targetdim;
    //The problem is that we cannot use stack allocated arrays, since they get destoryed after the function returns;
    Matrix<BaseFloat> lda_mat;
    lda.Estimate(opts,&lda_mat);
    //Transformation is stored in the lda_mat variable. We return a numpy array to the python script
    npy_intp dimensions[2]={lda_mat.NumRows(),lda_mat.NumCols()};
    //The result which we are going to return. Its a numpy array with lda_mat dimensions
    PyArrayObject* result;

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
    result = (PyArrayObject* )PyArray_SimpleNewFromData(2,dimensions,NPY_FLOAT,retarr);
    //Usually python does only store a reference on the given pointer and does not own the data
    //With this flag, we tell him to own the data and deallocate the data with the PyObject
    result->flags |= NPY_ARRAY_OWNDATA;
    //result = (PyArrayObject *)PyArray_FromDimsAndData(2, dimensions, PyArray_FLOAT,reinterpret_cast<char *> (lda_mat.Data()));
    return PyArray_Return(result);
}


static PyObject* py_predictldafromarray(PyObject* self,PyObject* args){
    PyArrayObject *pytrans;
    PyArrayObject* pyinputfeat;

    if (! PyArg_ParseTuple( args, "O!O!", &PyArray_Type,&pyinputfeat,&PyArray_Type,&pytrans)) return NULL;

    const Matrix<BaseFloat>& trans = pyarraytomatrix(pytrans);
    const Matrix<BaseFloat>& feat = pyarraytomatrix(pyinputfeat);

    const Matrix<BaseFloat>& feat_out = transform(feat,trans);

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


    const Matrix<BaseFloat> &trans = pyarraytomatrix(pytrans);

    const Matrix<BaseFloat> &feat = readFeatureFromChar(featurefilename);

    const Matrix<BaseFloat> &feat_out = transform(feat,trans);
    //int32 transform_rows = trans.NumRows(),
          //transform_cols = trans.NumCols(),
          //feat_dim = feat.NumCols();
    //Matrix<BaseFloat> feat_out(feat.NumRows(), transform_rows);
    //if (transform_cols == feat_dim) {
        //feat_out.AddMatMat(1.0, feat, kNoTrans, trans, kTrans, 0.0);
    //} else if (transform_cols == feat_dim + 1) {
         //append the implicit 1.0 to the input features.
        //SubMatrix<BaseFloat> linear_part(trans, 0, transform_rows, 0, feat_dim);
        //feat_out.AddMatMat(1.0, feat, kNoTrans, linear_part, kTrans, 0.0);
        //Vector<BaseFloat> offset(transform_rows);
        //offset.CopyColFromMat(trans, feat_dim);
        //feat_out.AddVecToRows(1.0, offset);
    //}else{
        //std::string err("Feature sizes do not match!");
        //return PyErr_Format(PyExc_ValueError,err.c_str());
    //}
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



/*
 *  * Bind Python function names to our C functions
 *   */
static PyMethodDef libldaModule_methods[] = {
    {"fitlda",py_fitlda,METH_VARARGS},
    {"predictldafromutterance",py_predictldafromutterance,METH_VARARGS},
    {"predictldafromarray",py_predictldafromarray,METH_VARARGS},
    {NULL, NULL}
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
