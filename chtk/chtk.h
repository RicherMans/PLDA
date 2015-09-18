#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include<string>
#include<stdexcept>
#include<sstream>
#include<vector>
#include<cstdio>
#include<typeinfo>
#include<iostream>
#include<cassert>
#include<zlib.h>
#include<map>
#include<memory>
#include<stdint.h>

namespace chtk{

    // Struct representing a typical HTK header with plus data
	struct htkarray{
		htkarray(size_t _nsamples, size_t _sample_period, size_t _samplesize, size_t _parmkind, size_t _frm_ext):
			nsamples(_nsamples), sample_period(_sample_period), samplesize(_samplesize), parmkind(_parmkind), frm_ext(_frm_ext)
		{
			data_holder = std::shared_ptr<std::vector<char>>(
				new std::vector<char>(nsamples*samplesize*(2*frm_ext+1)));
		}

		htkarray():nsamples(0), sample_period(0), samplesize(0), parmkind(0), frm_ext(0) {}

        template<typename T>
        T* data() {
        	return reinterpret_cast<T*>(&(*data_holder)[0]);
        }


        template<typename T>
        std::vector<T> as_vec() {
    		const T* p = data<T>();
	        return std::vector<T>(p, p+(nsamples*samplesize/4));
        }

		std::shared_ptr<std::vector<char>> data_holder;


		size_t frm_ext;
		size_t nsamples;
		size_t sample_period;
		size_t samplesize;
		size_t parmkind;
	};

    struct htkheader{
        int nsamples;
        int sample_period;
        short samplesize;
        short parmkind;
    };

	htkarray htk_load(std::string fname,int FRM_EXT);

    float swapfloatendian( const float inFloat );

    void swapfloatendian( char* inFloat );

    // Loading the header from a given filename
    htkheader load_header(std::string fname);

    // The reads in the header and returns it to the load_header(string) function
    htkheader load_header(std::ifstream &fstream);

}
#endif

