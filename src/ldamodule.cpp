#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include "transform/lda-estimate.h"

namespace kaldi{


template<typename T>
std::map<std::string,T> pythondict_to_map(){

    }

static PyObject* py_fitlda(PyObject* self,PyObject* args){
    PyObject* dict;
    /* the O! parses for a Python object (listObj) checked
     *    to be of type PyList_Type */
    if (! PyArg_ParseTuple( args, "O!", &PyDict_Type, &dict)) return NULL;
    //Check if the given argument is really a dict
    PyDict_Check(dict);
    auto numitems = PyDict_Size(dict);
    //equal to dict.items()
    PyObject* items = PyDict_Items(dict);
    //Variable to check if the sizes of the utterance for every key are consistent
    auto check_utts=0;
    for (auto i = 0u; i < numitems; ++i) {
        PyObject* item = PyList_GetItem(items,i);
        PyObject* key=PyTuple_GetItem(item,0);
        PyObject* values=PyTuple_GetItem(item,1);
        PyList_Check(values);
        auto num_utts=PyList_Size(values);
        if (i==0) {
            check_utts = num_utts;
        } else {
            if (check_utts != num_utts) {
                return PyErr_Format(PyExc_ValueError,"Size of lists inconsistent!");
            }
        }
        Vector<BaseFloat> v(10);
        if (num_utts < 0){
            std::string err("Values need to be a list of floats!");
            return PyErr_Format(PyExc_ValueError,err.c_str());
        }
        for (auto j = 0; j < num_utts; ++j) {
            PyObject *utt = PyList_GetItem(values,j);
            double curvalue = PyFloat_AsDouble(utt);
            BaseFloat f = (BaseFloat) curvalue;
        }
    }
    return Py_BuildValue("");
}

/*
 *  * Bind Python function names to our C functions
 *   */
static PyMethodDef libldaModule_methods[] = {
    {"fitlda",py_fitlda,METH_VARARGS},
    {NULL, NULL}
};

/*
 *  * Python calls this to let us initialize our module
 *   */
extern "C" void initliblda()
{
    (void) Py_InitModule("liblda", libldaModule_methods);

}
}
