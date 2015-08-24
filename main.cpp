#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <fstream>
#include <regex>
#include "CovarianceStats.hpp"
#include "chtk.h"
#include "Python.h"


bool is_file_exist(std::string fileName)
{
    std::ifstream infile(fileName);
    return infile.good();

}


namespace kaldi{
std::map<std::string, Vector<BaseFloat> *> readfeatures(std::string filename){
    std::ifstream infile(filename);
    std::string line;

    while(std::getline(infile,line)){
        //read in the htk feature
        if(! is_file_exist(line)){
            std::cerr << "Error. File "<< line << " does not exist !";
            exit(1);
        }
        chtk::htk_load(line,1);
    }
}
}

int main(int argc, char *argv[])
{
    using namespace kaldi;
    typedef kaldi::int32 int32;
    try{
        const char *usage ="Computes LDA for a given feature file. Feature file should be in the format:\n"
            "./lda scpfilelist mlffile output";
        ParseOptions po(usage);
        int32 lda_dim = 100; // Dimension we reduce to
        BaseFloat total_covariance_factor = 0.0;
        bool binary = true;

        po.Register("dim", &lda_dim, "Dimension we keep with the LDA transform");
        po.Register("total-covariance-factor", &total_covariance_factor,
                "If this is 0.0 we normalize to make the within-class covariance "
                "unit; if 1.0, the total covariance; if between, we normalize "
                "an interpolated matrix.");
        po.Register("binary", &binary, "Write output in binary mode");
        po.Read(argc, argv);

        //if(po.NumArgs() != 3){
            //po.PrintUsage();
            //exit(1);
        //}
        std::string feature_specifier=po.GetArg(1);
        std::map<std::string, Vector<BaseFloat> *> utt2feature;
        std::map<std::string, std::vector<std::string> > spk2utt;


        //dimension of the dataset
        int32 dim=10;

        Matrix<BaseFloat> lda_mat(lda_dim, dim + 1); // LDA matrix without the offset term.
        SubMatrix<BaseFloat> linear_part(lda_mat, 0, lda_dim, 0, dim);

        return 0;
    }catch(const std::exception &e){
        std::cerr << e.what();
        return -1;
    }

}

