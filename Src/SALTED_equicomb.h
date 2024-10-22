#pragma once

#include "convenience.h"
#include <complex>
#include "SALTED_math.h"

//Sparse implementation if sparsification is enabled
void equicomb(int natoms, int nrad1, int nrad2,
    const cvec4& v1,
    const cvec4& v2,
    const vec& w3j,
    const ivec2& llvec, const int& lam,
    const cvec2& c2r, const int& featsize,
    const int& nfps, const std::vector<int64_t>& vfps,
    vec& p);

//Normal implementation
void equicomb(int natoms, int nang1, int nang2, int nrad1, int nrad2,
    cvec4& v1,
    cvec4& v2,
    vec& w3j,
    ivec2& llvec, int lam,
    cvec2& c2r, int featsize,
    vec& p);

void equicomb_vec_multiply(const int natoms,
    const int nrad1, const int nrad2,
    const int nang1, const int nang2,
    const cvec4& v1,
    const cvec4& v2,
    const ivec2& max_lam_per_l1l2,
    const int& lam_max,
    //std::unordered_map<int, cvec>& p,
    //std::unordered_map<int, ivec>& feats_per_l1l2);
    cvec2& p,
    ivec2& feats_per_l1l2);

void get_angular_indexes_symmetric_per_lambda(const int lam_max, const int nang1, const int nang2,
                                            ivec2& highes_lam_per_l1l2, std::vector<ivec2>& l1l2_per_lam);

ivec2 calc_llvec(const int nang1, const int nang2, const int lam);