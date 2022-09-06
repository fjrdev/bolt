#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "quantize/profile_amm.hpp"
#include "immintrin.h"

// unit tests magic (keep this uncommented even if not doing catch run,
// // or else test_* files will have missing symbols)
// #define CATCH_CONFIG_RUNNER
//
// #ifdef BLAZE
//     #include "test/external/catch.hpp"
//     #else
//        // #include "catch.hpp"
//        #endif
//
using namespace std;

void get_splits_v1(mithral_amm_task<float> *task, int ncodebooks);
void get_splits_v2(mithral_amm_task<float> *task, int ncodebooks);
void get_split_dims(mithral_amm_task<float> *task, int ncodebooks);
void get_encode_scales(mithral_amm_task<float> *task, int ncodebooks);
void get_encode_offsets(mithral_amm_task<float> *task, int ncodebooks);
void get_centroids(mithral_amm_task<float> *task, int ncodebooks, int D);
void get_mat_x(mithral_amm_task<float> *task, int N, int D);
void get_mat_q(mithral_amm_task<float> *task, int D, int M);
void print_mat_w(mithral_amm_task<float> *task, int N, int M);
void get_luts(mithral_amm_task<float> *task, int N, int ncodebooks);

int main() {

        vector<int> tmp_vec;

        // read init-data into tmp_vec
        // [0] :> N
        // [1] :> D
        // [2] :> M
        // [3] :> ncodebooks
        ifstream file;
        file.open("../../experiments/python/init_params.txt");
        string init_param;
        for (int i = 0; i < 4; i++) {
                getline(file, init_param);
                tmp_vec.push_back(stoi(init_param));
        }

        mithral_amm_task<float> task(tmp_vec[0], tmp_vec[1], tmp_vec[2], tmp_vec[3], -1);

        // load split values
        get_splits_v1(&task, tmp_vec[3]);

        // load split dims
        get_split_dims(&task, tmp_vec[3]);

        // load scale values
        get_encode_scales(&task, tmp_vec[3]);

        // load offsets
        get_encode_offsets(&task, tmp_vec[3]);

        // load centroids
        get_centroids(&task, tmp_vec[3], tmp_vec[1]);

        // load matrix X
        get_mat_x(&task, tmp_vec[0], tmp_vec[1]);

        // load matrix Q
        get_mat_q(&task, tmp_vec[1], tmp_vec[2]);

        // load luts
        // get_luts(&task, tmp_vec[0], tmp_vec[3]);

        // print matrix W (np.matmul(X, Q))
        print_mat_w(&task, tmp_vec[0], tmp_vec[2]);

        // run
        task.run_matmul();

        // output       
        cout << "mithral output" << "\n" << task.output() << endl;

        return 0;
}


void get_splits_v1(mithral_amm_task<float> *task, int ncodebooks) {

        ifstream split;
        split.open("../../experiments/python/split_vals.txt");
        string splits;

        // use 4 columns for one codebook (first col: one splitval, second col: 2 splitvals, third col: 4 splitvals, fourth col: 8 splitvals)
        for (int i = 0; i < ncodebooks*4; i++) {
                for (int y = 0; y < 16; y++) {
                        getline(split, splits);
                        task->splitvals(y, i) = stoi(splits);
                }
        }

}


void get_splits_v2(mithral_amm_task<float> *task, int ncodebooks) {

        ifstream split;
        split.open("../../experiments/python/split_vals_v2.txt");
        string splits;

        // load all 15 split values of one codebook into one column and zero-pad the remaining 3 columns
        for (int i = 0; i < ncodebooks*4; i+=4) {
                for (int y = 0; y < 15; y++) {
                        getline(split, splits);
                        task->splitvals(y, i) = stoi(splits);
                }
                // zero-pad last split value of column
                task->splitvals(15, i) = 0;

                // zero-pad remaining 3 columns of codebook
                for (int z = i+1; z < i+4; z++) {
                        for (int y = 0; y < 16; y++) {
                                task->splitvals(y, z) = 0;
                        }
                }
        }
}


void get_split_dims(mithral_amm_task<float> *task, int ncodebooks) {

        ifstream dim_file;
        dim_file.open("../../experiments/python/split_dims.txt");
        string dim;

        for (int i = 0; i < ncodebooks * 4; i++) {
                getline(dim_file, dim);
                task->splitdims(i) = stoi(dim);
        }
}


void get_encode_scales(mithral_amm_task<float> *task, int ncodebooks) {

        ifstream scale_file;
        scale_file.open("../../experiments/python/scaleby.txt");
        string scales;

        for (int i = 0; i < ncodebooks * 4; i++) {
                getline(scale_file, scales);
                task->encode_scales(i) = stoi(scales);
        }
}


void get_encode_offsets(mithral_amm_task<float> *task, int ncodebooks) {

        ifstream offset_file;
        offset_file.open("../../experiments/python/offset.txt");
        string offsets;

        for (int i = 0; i < ncodebooks * 4; i++) {
                getline(offset_file, offsets);
                task->encode_offsets(i) = stoi(offsets);
        }
}


void get_centroids(mithral_amm_task<float> *task, int ncodebooks, int D) {

        ifstream centroids_file;
        centroids_file.open("../../experiments/python/centroids.txt");
        string centroids;

        // load centroids line-wise, not column-wise

        for (int i = 0; i < ncodebooks * 16; i++) {
                for (int y = 0; y < D; y++) {
                        getline(centroids_file, centroids);
                        task->centroids(i, y) = stoi(centroids);
                }
        }
}


void get_mat_x(mithral_amm_task<float> *task, int N, int D) {

        ifstream x_file;
        x_file.open("../../experiments/python/mat_x.txt");
        string x_vals;

        for (int i = 0; i < D; i++) {
                for (int y = 0; y < N; y++) {
                        getline(x_file, x_vals);
                        task->X(y, i) = stoi(x_vals);
                }
        }
}


void get_mat_q(mithral_amm_task<float> *task, int D, int M) {

        ifstream q_file;
        q_file.open("../../experiments/python/mat_q.txt");
        string q_vals;

        for (int i = 0; i < M; i++) {
                for (int y = 0; y < D; y++) {
                        getline(q_file, q_vals);
                        task->Q(y, i) = stoi(q_vals);
                }
        }
}


void print_mat_w(mithral_amm_task<float> *task, int N, int M) {

        ifstream w_file;
        w_file.open("../../experiments/python/mat_w.txt");
        string w_vals;

        cout << "----- correct mult -----" << endl;

        for (int i = 0; i < M; i++) {
                for (int y = 0; y < N; y++) {
                        getline(w_file, w_vals);
                        cout << w_vals << endl;
                }
        }
        cout << "-----" << endl;
}


void get_luts(mithral_amm_task<float> *task, int N, int ncodebooks) {

        ifstream lut_file;
        lut_file.open("../../experiments/python/luts.txt");
        string luts;

        for (int i = 0; i < ncodebooks*16; i++) {
                for (int y = 0; y < N; y++) {
                        getline(lut_file, luts);
                        task->amm.luts(y, i);
                }
        }
}