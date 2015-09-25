#ifndef KALDI_UTILS_HPP_
#define KALDI_UTILS_HPP_

#include "chtk.h"
#include <Python.h>
#include "numpy/arrayobject.h"
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"


namespace kaldi{
template<typename T>
T* pyvector_to_type(PyArrayObject* arrayin){
    return (T*) arrayin->data;
}

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
}
#endif
