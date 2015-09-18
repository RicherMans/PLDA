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
T* pyvector_to_type(PyArrayObject* arrayin);

template <std::size_t N>
struct type_of_size
{
    typedef char type[N];
};

template<typename T,std::size_t Size>
typename type_of_size<Size>::type& sizeof_array_help(T(&)[Size]);

//Define the sizeof function as a constant expression. We do not need to do this neceessarily
#define sizeof_array(arr) sizeof(sizeof_array_help(arr))


Matrix<BaseFloat> readFeatureFromChar(const char* featurefilename);
Matrix<BaseFloat> readFeatureFromPyString(PyObject* pyfeaturefilename);


template<typename T>
void k_matrix_to_array(const MatrixBase<T>& inpmat,T* retarr);

//Packed matrix overload, packed matrix are triangular matrices
template<typename T>
void k_matrix_to_array(const PackedMatrix<T>& inpmat,T* retarr);

template<typename T>
void k_vector_to_array(const VectorBase<T>& inputvec,T* retarr);


template<typename T>
Matrix<BaseFloat> pyarraytomatrix(PyArrayObject* pytrans);

template<typename T>
Vector<T> pyarraytovector(PyArrayObject* pytrans);


}
#endif

