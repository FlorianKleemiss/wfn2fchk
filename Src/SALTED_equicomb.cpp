#include "SALTED_equicomb.h"

#if has_RAS
#include "cblas.h"
#endif

// BE AWARE, THAT V2 IS ALREADY ASSUMED TO BE CONJUGATED!!!!!
void equicomb(int natoms, int nrad1, int nrad2,
              const cvec4 &v1,
              const cvec4 &v2,
              const vec &w3j,
              const ivec2 &llvec, const int &lam,
              const cvec2 &c2r, const int &featsize,
              const int &nfps, const std::vector<int64_t> &vfps,
              vec &p)
{

    const int l21 = 2 * lam + 1;
    const int llmax = (int)llvec[0].size();

    // Initialize p with zeros
    p.assign(natoms * l21 * nfps, 0.0);
    const vec f_vec(featsize, 0.0);

    // Declare variables at the beginning
    int iat, n1, n2, il, imu, im1, im2, i, j, ifeat, iwig, l1, l2, mu, m1, m2;
    double inner, normfact;
    const cdouble null(0.0, 0.0);
#ifdef _OPENMP
    omp_lock_t l;
    omp_init_lock(&l);
#endif
    ProgressBar pb(natoms);
#pragma omp parallel for private(iat, n1, n2, il, imu, im1, im2, i, j, ifeat, iwig, l1, l2, mu, m1, m2, inner, normfact) default(none) shared(pb, l, natoms, nrad1, nrad2, v1, v2, w3j, llmax, llvec, lam, c2r, nfps, vfps, p, featsize, l21, null, f_vec, std::cout)
    for (iat = 0; iat < natoms; ++iat)
    {
        vec2 ptemp(l21, f_vec);
        cvec pcmplx(l21, null);
        vec preal(l21, 0.0);
        inner = 0.0;

        ifeat = 0;
        for (n1 = 0; n1 < nrad1; ++n1)
        {
            for (n2 = 0; n2 < nrad2; ++n2)
            {
                iwig = 0;
                for (il = 0; il < llmax; ++il)
                {
                    l1 = llvec[0][il];
                    l2 = llvec[1][il];

                    //cvec2 *v1_ptr = (cvec2 *)&v1[l1][iat];
                    //cvec2 *v2_ptr = (cvec2 *)&v2[l2][iat];

                    cvec *v1_ptr = (cvec *)&v1[iat][n1][l1];
                    cvec *v2_ptr = (cvec *)&v2[iat][n2][l2];

                    fill(pcmplx.begin(), pcmplx.end(), null);

                    for (imu = 0; imu < l21; ++imu)
                    {
                        mu = imu - lam;
                        for (im1 = 0; im1 < 2 * l1 + 1; ++im1)
                        {
                            m1 = im1 - l1;
                            m2 = m1 - mu;
                            if (abs(m2) <= l2)
                            {
                                im2 = m2 + l2;
                                //pcmplx[imu] += w3j[iwig] * (*v1_ptr)[im1][n1] * (*v2_ptr)[im2][n2];
                                pcmplx[imu] += w3j[iwig] * (*v1_ptr)[im1] * (*v2_ptr)[im2];
                                iwig++;
                            }
                        }
                    }

                    fill(preal.begin(), preal.end(), 0.0);
                    for (i = 0; i < l21; ++i)
                    {
                        for (j = 0; j < l21; ++j)
                        {
                            preal[i] += real(c2r[i][j] * pcmplx[j]);
                        }
                        inner += preal[i] * preal[i];
                        ptemp[i][ifeat] = preal[i];
                    }
                    ifeat++;
                }
            }
        }

        normfact = sqrt(inner);
        int offset = iat * l21 * nfps;
        for (int n = 0; n < nfps; ++n)
        {
            for (imu = 0; imu < l21; ++imu)
            {
                p[offset + (imu * nfps)] = ptemp[imu][vfps[n]] / normfact;
            }
            offset++;
        }
#ifdef _OPENMP
        omp_set_lock(&l);
#endif
        pb.update(std::cout);
#ifdef _OPENMP
        omp_unset_lock(&l);
#endif
    }
#ifdef _OPENMP
    omp_destroy_lock(&l);
#endif
}



void equicomb(int natoms, int nang1, int nang2, int nrad1, int nrad2,
              cvec4 &v1,
              cvec4 &v2,
              vec &w3j,
              ivec2 &llvec, int lam,
              cvec2 &c2r, int featsize,
              vec &p)
{

    const int l21 = 2 * lam + 1;
    const int llmax = (int)llvec[0].size();
    // Initialize p with zeros
    p.assign(natoms * (2 * lam + 1) * featsize, 0.0);
    const vec f_vec(featsize, 0.0);

    // Declare variables at the beginning
    int iat, n1, n2, il, imu, im1, im2, i, j, ifeat, iwig, l1, l2, mu, m1, m2;
    double inner, normfact;
    const cdouble null(0.0, 0.0);

//#pragma omp parallel for private(iat, n1, n2, il, imu, im1, im2, i, j, ifeat, iwig, l1, l2, mu, m1, m2, inner, normfact) default(none) shared(natoms, nang1, nang2, nrad1, nrad2, v1, v2, w3j, llmax, llvec, lam, c2r, p, featsize, null)
    for (iat = 0; iat < natoms; ++iat)
    {
        vec2 ptemp(l21, f_vec);
        cvec pcmplx(l21, null);
        vec preal(l21, 0.0);
        inner = 0.0;

        ifeat = 0;
        for (n1 = 0; n1 < nrad1; ++n1)
        {
            for (n2 = 0; n2 < nrad2; ++n2)
            {
                iwig = 0;
                for (il = 0; il < llmax; ++il)
                {
                    l1 = llvec[0][il];
                    l2 = llvec[1][il];

                    //cvec2 *v1_ptr = (cvec2 *)&v1[l1][iat];
                    //cvec2 *v2_ptr = (cvec2 *)&v2[l2][iat];

                    cvec* v1_ptr = (cvec*)&v1[iat][n1][l1];
                    cvec* v2_ptr = (cvec*)&v2[iat][n2][l2];

                    fill(pcmplx.begin(), pcmplx.end(), null);

                    for (imu = 0; imu < l21; ++imu)
                    {
                        mu = imu - lam;
                        for (im1 = 0; im1 < 2 * l1 + 1; ++im1)
                        {
                            m1 = im1 - l1;
                            m2 = m1 - mu;
                            if (abs(m2) <= l2)
                            {
                                im2 = m2 + l2;
                                //pcmplx[imu] += w3j[iwig] * (*v1_ptr)[im1][n1] * (*v2_ptr)[im2][n2];
                                pcmplx[imu] += w3j[iwig] * (*v1_ptr)[im1] * (*v2_ptr)[im2];
                                iwig++;
                            }
                        }
                    }

                    fill(preal.begin(), preal.end(), 0.0);
                    for (i = 0; i < l21; ++i)
                    {
                        for (j = 0; j < l21; ++j)
                        {
                            preal[i] += real(c2r[i][j] * pcmplx[j]);
                        }
                        inner += preal[i] * preal[i];
                        ptemp[i][ifeat] = preal[i];
                    }
                    ifeat++;
                }
            }
        }

        int offset = iat * l21 * featsize;
        normfact = sqrt(inner);
        std::cout << normfact << std::endl;
        for (ifeat = 0; ifeat < featsize; ++ifeat)
        {
            for (imu = 0; imu < l21; ++imu)
            {
                p[offset + (imu * featsize)] = ptemp[imu][ifeat] / normfact;
            }
            offset++;
        }
    }
}



// Returns the l values for each combination of l1 and l2
void get_angular_indexes_symmetric_per_lambda(const int lam_max, const int nang1, const int nang2, ivec2& highes_lam_per_l1l2, std::vector<ivec2>& l1l2_per_lam) {
    // Iterate through lam_max to 0
    for (int lam = lam_max; lam >= 0; --lam) {
        for (int l1 = 0; l1 <= nang1; ++l1) {
            for (int l2 = 0; l2 <= nang2; ++l2) {
                // Keep only even combination to enforce inversion symmetry
                if ((lam + l1 + l2) % 2 == 0) {
                    // Enforce triangular inequality
                    if ((std::abs(l2 - lam) <= l1) && (l1 <= (l2 + lam))) {
                        if (highes_lam_per_l1l2[l1][l2] == -1) {
                            highes_lam_per_l1l2[l1][l2] = lam;   // Assign the lam value
                        }
                        l1l2_per_lam[lam].push_back({ l1, l2 });  // Assign the l1 and l2 values
                    }
                }
            }
        }
    }
}

//Computes only the necessary products of v1 and v2
void equicomb_vec_multiply(const int natoms,
    const int nrad1, const int nrad2,
    const int nang1, const int nang2,
    const cvec4& v1,
    const cvec4& v2,
    const ivec2& max_lam_per_l1l2,
    const int& lam_max,
    cvec2& p_vec,
    ivec2& feats_per_l1l2_vec)
{
    const cdouble null(0.0, 0.0);

    // Declare variables at the beginning
    int l1, l2, lam, iat, n1, n2, ifeat;
    int imu, im1, im2, mu, m1, m2;
    int total_features = natoms * nrad1 * nrad2 * (nang1 + 1) * (nang2 + 1);

    const int radfactor1 = (nang2 + 1) * (nang1 + 1);
    const int radfactor2 = radfactor1 * nrad2;
    const int radfactor3 = radfactor2 * nrad1;

    // Initialize the vectors
    p_vec.resize(total_features);
    feats_per_l1l2_vec.resize((nang1 + 1) * (nang2 + 1));

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    std::vector<ivec2> local_feats_per_l1l2(nthreads, ivec2((nang1 + 1) * (nang2 + 1)));
#endif

    ProgressBar pb(total_features);

#pragma omp parallel for default(none) \
    private(ifeat, l1, l2, lam, n1, n2, iat, imu, im1, im2, mu, m1, m2) \
    shared(natoms, nrad1, nrad2, radfactor1, radfactor2, radfactor3, v1, v2, max_lam_per_l1l2, p_vec, feats_per_l1l2_vec, null, std::cout, pb \
    , nang1, nang2 \
    , nthreads \
    , local_feats_per_l1l2)
    for (ifeat = 0; ifeat < total_features; ++ifeat)
    {
        l2 = ifeat % (nang2 + 1);
        l1 = (ifeat / (nang2 + 1)) % (nang1 + 1);

        // Only calculate the necessary combinations
        lam = max_lam_per_l1l2[l1][l2];
        if (lam == -1) {
            continue;
        }

        n2 = (ifeat / radfactor1) % nrad2;
        n1 = (ifeat / radfactor2) % nrad1;
        iat = (ifeat / radfactor3) % natoms;

		cvec pcmplx((2 * lam + 1) * (2 * nang1 + 1),null);
        for (imu = 0; imu < 2 * lam + 1; ++imu) {
            mu = imu - lam;
            for (im1 = 0; im1 < 2 * l1 + 1; ++im1) {
                m1 = im1 - l1;
                m2 = m1 - mu;
                if (abs(m2) <= l2) {
                    im2 = m2 + l2;
                    pcmplx[imu * (2 * l1 + 1) + im1] = v1[iat][n1][l1][im1] * v2[iat][n2][l2][im2];
                }
            }
        }

        p_vec[ifeat] = std::move(pcmplx);

#ifdef _OPENMP
        int tid = omp_get_thread_num();
        local_feats_per_l1l2[tid][l1 * (nang2 + 1) + l2].push_back(ifeat);
#else
        feats_per_l1l2_vec[l1 * (nang2 + 1) + l2].push_back(ifeat);
#endif

#pragma omp critical
        {
            pb.update(std::cout);
        }
    }

#ifdef _OPENMP
    // Merge thread-local feats_per_l1l2 into the global feats_per_l1l2_vec
    for (int tid = 0; tid < nthreads; ++tid) {
        for (size_t idx = 0; idx < feats_per_l1l2_vec.size(); ++idx) {
            feats_per_l1l2_vec[idx].insert(
                feats_per_l1l2_vec[idx].end(),
                local_feats_per_l1l2[tid][idx].begin(),
                local_feats_per_l1l2[tid][idx].end()
            );
        }
    }
#endif
}



//Calculate the neccecary combinations of l1 and l2 for a given lam
ivec2 calc_llvec(const int nang1, const int nang2, const int lam) {
    ivec2 llvec((nang1 + 1) * (nang2 + 1), ivec(2));
    int indx = 0;
    for (int l1 = 0; l1 < nang1 + 1; l1++)
    {
        for (int l2 = 0; l2 < nang2 + 1; l2++)
        {
            // keep only even combination to enforce inversion symmetry
            if ((lam + l1 + l2) % 2 == 0)
            {
                if (abs(l2 - lam) <= l1 && l1 <= (l2 + lam))
                {
                    llvec[indx] = { l1, l2 };
                    indx++;
                }
            }
        }
    }
	llvec.resize(indx);
    return llvec;
}
