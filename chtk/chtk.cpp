#include"chtk.h"
#include<complex>
#include<cstdlib>
#include<algorithm>
#include<cstring>
#include<iomanip>
#include<stdint.h>
#include<netinet/in.h>
#include <fstream>
#include <vector>


void chtk::swapfloatendian( char* inFloat )
{

   std::swap(inFloat[0],inFloat[3]);
   std::swap(inFloat[1],inFloat[2]);
   // swap the bytes into a temporary buffer

}

float chtk::swapfloatendian( const float inFloat )
{
   float retVal;
   char *floatToConvert = ( char* ) & inFloat;
   char *returnFloat = ( char* ) & retVal;

   // swap the bytes into a temporary buffer
   returnFloat[0] = floatToConvert[3];
   returnFloat[1] = floatToConvert[2];
   returnFloat[2] = floatToConvert[1];
   returnFloat[3] = floatToConvert[0];

   return retVal;
}


chtk::htkarray chtk::htk_load(std::string fname, int FRM_EXT){


    std::ifstream inp;
    inp.open(fname,std::ios::binary);
    if (!inp.is_open()){
        std::string exept= "File " + fname + " cannot be opened !\n";
        throw std::runtime_error(exept.c_str());
    }

    chtk::htkheader header = load_header(inp);
    // Reading in the header file of any htk binary file


    // Generate the returning htk representation
    chtk::htkarray retarr(ntohl(header.nsamples),ntohl(header.sample_period),ntohs(header.samplesize),ntohs(header.parmkind),FRM_EXT);
    // Prepare vector to read in data
    std::vector<std::vector<char>> buf(retarr.nsamples);
    for ( auto i=0u ; i < retarr.nsamples; i++) {
        // buf[i] = new char[arr->samplesize];
        std::vector<char> samplebuf(retarr.samplesize/sizeof(char));
        inp.read(reinterpret_cast<char*>(samplebuf.data()),retarr.samplesize);

        // std::cout <<"samplesize for sample " << i << " is " << (arr)->samplesize/4 << std::endl;
        for(auto j = 0u ; j < samplebuf.size();j+=4){
            // Swapping the elements from big endian to little endian

            std::swap(samplebuf.at(j+3),samplebuf.at(j));
            std::swap(samplebuf.at(j+2),samplebuf.at(j+1));
        }
        buf.at(i) = samplebuf;
    }

    auto offset=0;
    // Append the extended features
    for ( auto i=0; i< retarr.nsamples; i++){
        for (auto j= -FRM_EXT;j < FRM_EXT+1;j++){
            int tmpidx= i+j>0?(i+j):0;
            // Check if the extended index is in range, if it is use the extended index
            // otherwise use the last sample
            int index= (tmpidx>retarr.nsamples-1)?retarr.nsamples-1:tmpidx;
            // Copy the current extended frame into the current buffer
            std::copy(buf.at(index).begin(),
                      buf.at(index).end(),
                      retarr.data_holder.get()->begin()+offset
                      );
            offset+=retarr.samplesize;
        }
    }
    return retarr;
}

chtk::htkheader chtk::load_header(std::ifstream &inp){
    chtk::htkheader header;
    inp.read((char*)&header,sizeof(htkheader));
    return header;
}

chtk::htkheader chtk::load_header (std::string fname){
    std::ifstream inp;
    inp.open(fname,std::ios::binary);
    if (!inp.is_open()){
        std::string exept= "File " + fname + " cannot be opened !\n";
        throw std::runtime_error(exept.c_str());
    }
    htkheader header = load_header(inp);
    header.nsamples = ntohl(header.nsamples);
    header.sample_period = ntohl(header.sample_period);
    header.samplesize =ntohs(header.samplesize);
    header.parmkind  = ntohs(header.parmkind);
    return header;

}
