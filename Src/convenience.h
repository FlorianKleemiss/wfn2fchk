#pragma once
#define WIN32_LEAN_AND_MEAN
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#ifdef __cplusplus__
#include <cstdlib>
#else
#include <stdlib.h>
#endif
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <regex>
#include <set>
#include <map>
#include <string>
#include <stdexcept>
#include <sstream>
#include <typeinfo>
#include <vector>

// Here are the system specific libaries
#ifdef _WIN32
#include <direct.h>
#define GetCurrentDir _getcwd(NULL, 0)
#include <io.h>
#else
#define GetCurrentDir getcwd
#include <unistd.h>
#include <cfloat>
#include <sys/wait.h>
#include <termios.h>
#include <cstring>
#endif

// Pre-definition of classes included later
class WFN;
class cell;
class atom;

typedef std::complex<double> cdouble;
typedef std::vector<double> vec;
typedef std::vector<int> ivec;
typedef std::vector<cdouble> cvec;
typedef std::chrono::high_resolution_clock::time_point time_point;

inline double vec_sum(const vec& in)
{
    double res = 0.0;
    for (int i = 0; i < in.size(); i++)
        res += in[i];
    return res;
}

inline cdouble vec_sum(const cvec& in)
{
    cdouble res = 0.0;
    for (int i = 0; i < in.size(); i++)
        res += in[i];
    return res;
}

inline const std::complex<double> c_one(0, 1.0);

std::string help_message();
std::string NoSpherA2_message();
std::string build_date();

namespace constants
{
#include <limits>

    double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
    {
        return curr == prev
            ? curr
            : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }

    /*
     * Constexpr version of the square root
     * Return value:
     *   - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
     *   - Otherwise, returns NaN
     * Taken from https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr
     */
    double constexpr sqrt(double x)
    {
        return x >= 0 && x < std::numeric_limits<double>::infinity()
            ? sqrtNewtonRaphson(x, x, 0)
            : std::numeric_limits<double>::quiet_NaN();
    }
    // Constants for later use
    constexpr double SQRT2 = sqrt(2.0);
    constexpr double SQRT3 = sqrt(3.0);
    constexpr double SQRT5 = sqrt(5.0);
    constexpr double INV_SQRT2 = 1.0 / SQRT2;
    constexpr double INV_SQRT3 = 1.0 / SQRT3;
    constexpr double INV_SQRT5 = 1.0 / SQRT5;
    constexpr int hardness = 3;
    constexpr double cutoff = 1.0e-20;
    constexpr double PI = 3.14159265358979323846;
    constexpr double INV_PI = 1.0 / PI;
    constexpr double PI_2 = PI / 2.0;
    constexpr double TWO_PI = 2 * PI;
    constexpr double FOUR_PI = 4 * PI;
    constexpr double C0 = SQRT2 * FOUR_PI;
    const double sqr_pi = sqrt(PI);
    constexpr double PI2 = PI * PI;
    constexpr double PI3 = PI2 * PI;
    constexpr double PI_180 = PI / 180.0;
    const double TG32 = tgamma(3.0 / 2.0);
    constexpr double ED_fact = 0.023934;
    constexpr int max_LT = 33;
    constexpr int MAG = 5810;
    //                       3,     5     7,    9,    11,   13,   15,   17
    //                      19,    21
    constexpr int lebedev_table[33] = { 6, 14, 26, 38, 50, 74, 86, 110,
                                       146, 170, 194, 230, 266, 302, 350, 434,
                                       590, 770, 974, 1202, 1454, 1730, 2030, 2354,
                                       2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810 };
    constexpr long long int ft[21]{ 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000 };
    constexpr double alpha_coef = 0.1616204596739954813316614;
    constexpr double c_13 = 1.0 / 3.0;
    constexpr double c_43 = 4.0 / 3.0;
    constexpr double c_38 = 3.0 / 8.0;
    constexpr double c_16 = 1.0 / 6.0;
    constexpr double c_23 = 2.0 / 3.0;
    constexpr double c_53 = 5.0 / 3.0;
    constexpr double c_1_21 = 1.0 / 21.0;
    constexpr double c_1_30 = 1.0 / 30.0;
    constexpr double c_1_35 = 1.0 / 35.0;
    constexpr double c_1_15 = 1.0 / 15.0;
    constexpr double c_3_40 = 3.0 / 40.0;
    constexpr double c_1_79 = 1.0 / 79.0;
    constexpr double c_4_105 = 4.0 / 105.0;
    constexpr double c_1_105 = 1.0 / 105.0;
    constexpr double c_9_280 = 9.0 / 280.0;
    constexpr double c_m43 = -4.0 / 3.0;
    constexpr double c_m53 = -5.0 / 3.0;
    constexpr double barnsbohr = 2.80028520539078E+7;
    constexpr double fine_struct = 7.2973525693E-3;
    constexpr double inv_fine_struct = 1 / fine_struct;
    constexpr double fine_pi = inv_fine_struct / TWO_PI / PI;
    constexpr double inv_fine_mod = inv_fine_struct / FOUR_PI;
    constexpr double keV_per_hartree = 0.027211386245988;
    constexpr double angstrom2eV = 1.23984193 * 10000;
    constexpr double angstrom2keV = 12.3984193;
    constexpr double f_to_mu = 4208.031548;
    constexpr double barns_to_electrons = 1.43110541E-8;
    constexpr double a0 = 0.529177210903E-10;              // in m
    constexpr double h = 6.62607015E-34 / 1.602176634E-19; // in eV*s
    constexpr double Ryd_ener = 13.6056923;                // in eV
    constexpr double alpha = 0.0072973525693;              // Sommerfeld fine structure constant
    constexpr double el_mass = 9.1093837015E-31;           // in kg
    constexpr double el_charge = 1.602176634E-19;          // in C
    constexpr double speed_of_light = 2.99792458E8;        // m/s
    constexpr double null = 0.0;
		constexpr cdouble cnull = cdouble(0.0, 0.0);

    const double ctelf = 10 * pow(2, -2.0 / 3.0) * pow(3, c_m53) * pow(PI, -c_43);
    constexpr double c_1_4p = sqrt(1.0 / (FOUR_PI));
    constexpr double c_3_4p = sqrt(3.0 / (FOUR_PI));
    constexpr double c_5_16p = sqrt(5.0 / (16.0 * PI));
    constexpr double c_7_16p = sqrt(7.0 / (16.0 * PI));
    constexpr double c_9_256p = sqrt(9.0 / (256.0 * PI));
    constexpr double c_11_256p = sqrt(11.0 / (256.0 * PI));
    constexpr double c_13_1024p = sqrt(13.0 / (1024.0 * PI));
    constexpr double c_15_4p = sqrt(15.0 / (FOUR_PI));
    constexpr double c_15_16p = sqrt(15.0 / (16.0 * PI));
    constexpr double c_21_32p = sqrt(21.0 / (32.0 * PI));
    constexpr double c_35_32p = sqrt(35.0 / (32.0 * PI));
    constexpr double c_45_16p = sqrt(45.0 / (16.0 * PI));
    constexpr double c_45_32p = sqrt(45.0 / (32.0 * PI));
    constexpr double c_45_64p = sqrt(45.0 / (64.0 * PI));
    constexpr double c_105_4p = sqrt(105.0 / (FOUR_PI));
    constexpr double c_105_16p = sqrt(105.0 / (16.0 * PI));
    constexpr double c_165_256p = sqrt(165.0 / (256.0 * PI));
    constexpr double c_273_256p = sqrt(273.0 / (256.0 * PI));
    constexpr double c_315_16p = sqrt(315.0 / (16.0 * PI));
    constexpr double c_315_32p = sqrt(315.0 / (32.0 * PI));
    constexpr double c_315_256p = sqrt(315.0 / (256.0 * PI));
    constexpr double c_385_512p = sqrt(385.0 / (512.0 * PI));
    constexpr double c_693_2048p = sqrt(693.0 / (2048.0 * PI));
    constexpr double c_1155_64p = sqrt(1155.0 / (64.0 * PI));
    constexpr double c_3465_256p = sqrt(3465.0 / (256.0 * PI));

    inline const long long int ft_fun(const int& nr)
    {
        if (nr >= 0 && nr <= 20)
            return ft[nr];
        else if (nr < 0)
            return 0;
        else
            return ft_fun(nr - 1) * nr;
    }

    constexpr double bohr2ang(const double& inp)
    {
        return inp * 0.529177249;
    }

    constexpr double bohr2ang_p(const double& inp, const int& p)
    {
        if (p == 0)
            return 1.0;
        else if (p == 1)
            return bohr2ang(inp);
        else
            return bohr2ang_p(bohr2ang(inp), p - 1);
    }

    constexpr double ang2bohr(const double& inp)
    {
        return inp / 0.529177249;
    }
    inline const double ang2bohr_p(const double& inp, const int& p)
    {
        if (p == 0)
            return 1.0;
        else if (p == 1)
            return ang2bohr(inp);
        else
            return ang2bohr_p(ang2bohr(inp), p - 1);
    }

    constexpr double cubic_ang2bohr(const double& inp)
    {
        return inp / (0.529177249 * 0.529177249 * 0.529177249);
    }

    constexpr double cubic_bohr2ang(const double& inp)
    {
        return inp * (0.529177249 * 0.529177249 * 0.529177249);
    }

    //------------------general functions for easy use of terminal input--------------------
    constexpr double bragg_angstrom[114]{
        0.00, // DUMMY LINE
        0.35, 0.35,
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45,
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.00,
        2.20, 1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 1.10,
        2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, 1.55, 1.45, 1.45, 1.40, 1.40, 1.40,
        2.60, 2.15, 1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.55, 1.45, 1.35, 1.30, 1.30, 1.35, 1.35, 1.35, 1.50, 1.90, 1.75, 1.60, 1.90, 1.50, 1.50,
        2.80, 2.35, 2.15, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.00, 1.95, 1.95, 1.95, 1.95, 1.95, 1.95 };

    // Covalent Radii according to the CSD
    constexpr double covalent_radii[114]{
        0.0,
        0.23, 1.5,
        1.28, 0.96, 0.83, 0.68, 0.68, 0.68, 0.64, 1.5,
        1.66, 1.41, 1.21, 1.2, 1.05, 1.02, 0.99, 1.51,
        2.03, 1.76, 1.7, 1.6, 1.53, 1.39, 1.61, 1.52, 1.26, 1.24, 1.32, 1.22, 1.22, 1.17, 1.21, 1.22, 1.21, 1.5,
        2.2, 1.95, 1.9, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.54, 1.42, 1.39, 1.39, 1.47, 1.4, 1.5,
        2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.9, 1.87, 1.87, 1.75, 1.7, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.4, 1.21, 1.5,
        2.6, 2.21, 2.15, 2.06, 2.00, 1.96, 1.9, 1.87, 1.8, 1.69, 1.54, 1.83, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 };

    // Integer atom masses
    constexpr unsigned int integer_masses[]{
        1, 4,
        7, 9, 11, 12, 14, 16, 19, 20,
        23, 24, 27, 28, 31, 32, 35, 40,
        39, 40, 45, 48, 51, 52, 55, 56, 59, 58, 63, 64, 69, 74, 75, 80, 79, 84,
        85, 87, 88, 91, 92, 96, 98, 101, 103, 106, 108, 112, 115, 119, 122, 128, 127, 131,
        132, 137, 139, 140, 141, 144, 145, 150, 152, 157, 159, 163, 165, 167, 169, 173, 175, 178, 181, 184, 186, 190, 192, 195, 197, 201, 204, 207, 209, 209, 210, 222 };

    constexpr double real_masses[]{
        1.0079, 4.0026,
        6.941, 9.0122, 10.811, 12.011, 14.007, 15.999, 18.998, 20.18,
        22.99, 24.305, 26.986, 28.086, 30.974, 32.065, 35.453, 39.948,
        39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.64, 74.922, 78.96, 79.904, 83.798,
        85.468, 87.62, 88.906, 91.224, 92.906, 95.96, 97.90, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.6, 126.9, 131.29,
        132.91, 137.33, 139.91, 140.12, 140.91, 144.24, 144.9, 150.36, 151.96, 157.25, 158.93, 162.5, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2, 208.98, 208.9, 209.9, 222.0 };


    const std::map<std::string, std::string> SPACE_GROUPS_HM = { {"1", "P1"},  {"2", "P-1"},  
        {"3:b", "P121"},  {"3:c", "P112"},  {"3:a", "P211"},  {"4:b", "P1211"},  {"4:c", "P1121"},  
        {"4:a", "P2111"},  {"5:b1", "C121"},  {"5:b2", "A121"},  {"5:b3", "I121"},  {"5:c1", "A112"},  
        {"5:c2", "B112"},  {"5:c3", "I112"},  {"5:a1", "B211"},  {"5:a2", "C211"},  {"5:a3", "I211"},  
        {"6:b", "P1m1"},  {"6:c", "P11m"},  {"6:a", "Pm11"},  {"7:b1", "P1c1"},  {"7:b2", "P1n1"},  
        {"7:b3", "P1a1"},  {"7:c1", "P11a"},  {"7:c2", "P11n"},  {"7:c3", "P11b"},  {"7:a1", "Pb11"},  
        {"7:a2", "Pn11"},  {"7:a3", "Pc11"},  {"8:b1", "C1m1"},  {"8:b2", "A1m1"},  {"8:b3", "I1m1"},  
        {"8:c1", "A11m"},  {"8:c2", "B11m"},  {"8:c3", "I11m"},  {"8:a1", "Bm11"},  {"8:a2", "Cm11"},  
        {"8:a3", "Im11"},  {"9:b1", "C1c1"},  {"9:b2", "A1n1"},  {"9:b3", "I1a1"},  {"9:-b1", "A1a1"},  
        {"9:-b2", "C1n1"},  {"9:-b3", "I1c1"},  {"9:c1", "A11a"},  {"9:c2", "B11n"},  {"9:c3", "I11b"},  
        {"9:-c1", "B11b"},  {"9:-c2", "A11n"},  {"9:-c3", "I11a"},  {"9:a1", "Bb11"},  {"9:a2", "Cn11"},  
        {"9:a3", "Ic11"},  {"9:-a1", "Cc11"},  {"9:-a2", "Bn11"},  {"9:-a3", "Ib11"},  {"10:b", "P12/m1"},  
        {"10:c", "P112/m"},  {"10:a", "P2/m11"},  {"11:b", "P121/m1"},  {"11:c", "P1121/m"},  {"11:a", "P21/m11"},  
        {"12:b1", "C12/m1"},  {"12:b2", "A12/m1"},  {"12:b3", "I12/m1"},  {"12:c1", "A112/m"},  {"12:c2", "B112/m"},  
        {"12:c3", "I112/m"},  {"12:a1", "B2/m11"},  {"12:a2", "C2/m11"},  {"12:a3", "I2/m11"},  {"13:b1", "P12/c1"},  
        {"13:b2", "P12/n1"},  {"13:b3", "P12/a1"},  {"13:c1", "P112/a"},  {"13:c2", "P112/n"},  {"13:c3", "P112/b"},  
        {"13:a1", "P2/b11"},  {"13:a2", "P2/n11"},  {"13:a3", "P2/c11"},  {"14:b1", "P121/c1"},  {"14:b2", "P121/n1"},  
        {"14:b3", "P121/a1"},  {"14:c1", "P1121/a"},  {"14:c2", "P1121/n"},  {"14:c3", "P1121/b"},  {"14:a1", "P21/b11"},  
        {"14:a2", "P21/n11"},  {"14:a3", "P21/c11"},  {"15:b1", "C12/c1"},  {"15:b2", "A12/n1"},  {"15:b3", "I12/a1"},  
        {"15:-b1", "A12/a1"},  {"15:-b2", "C12/n1"},  {"15:-b3", "I12/c1"},  {"15:c1", "A112/a"},  {"15:c2", "B112/n"},  
        {"15:c3", "I112/b"},  {"15:-c1", "B112/b"},  {"15:-c2", "A112/n"},  {"15:-c3", "I112/a"},  {"15:a1", "B2/b11"},  
        {"15:a2", "C2/n11"},  {"15:a3", "I2/c11"},  {"15:-a1", "C2/c11"},  {"15:-a2", "B2/n11"},  {"15:-a3", "I2/b11"},  
        {"16", "P222"},  {"17", "P2221"},  {"17:cab", "P2122"},  {"17:bca", "P2212"},  {"18", "P21212"},  {"18:cab", "P22121"}, 
        {"18:bca", "P21221"},  {"19", "P212121"},  {"20", "C2221"},  {"20:cab", "A2122"},  {"20:bca", "B2212"},  {"21", "C222"},  
        {"21:cab", "A222"},  {"21:bca", "B222"},  {"22", "F222"},  {"23", "I222"},  {"24", "I212121"},  {"25", "Pmm2"},  
        {"25:cab", "P2mm"},  {"25:bca", "Pm2m"},  {"26", "Pmc21"},  {"26:ba-", "Pcm21"},  {"26:cab", "P21ma"},  
        {"26:-cb", "P21am"},  {"26:bca", "Pb21m"},  {"26:a-c", "Pm21b"},  {"27", "Pcc2"},  {"27:cab", "P2aa"}, 
        {"27:bca", "Pb2b"},  {"28", "Pma2"},  {"28:ba-", "Pbm2"},  {"28:cab", "P2mb"},  {"28:-cb", "P2cm"},  
        {"28:bca", "Pc2m"},  {"28:a-c", "Pm2a"},  {"29", "Pca21"},  {"29:ba-", "Pbc21"},  {"29:cab", "P21ab"},  
        {"29:-cb", "P21ca"},  {"29:bca", "Pc21b"},  {"29:a-c", "Pb21a"},  {"30", "Pnc2"},  {"30:ba-", "Pcn2"},  
        {"30:cab", "P2na"},  {"30:-cb", "P2an"},  {"30:bca", "Pb2n"},  {"30:a-c", "Pn2b"},  {"31", "Pmn21"},  
        {"31:ba-", "Pnm21"},  {"31:cab", "P21mn"},  {"31:-cb", "P21nm"},  {"31:bca", "Pn21m"},  {"31:a-c", "Pm21n"},  
        {"32", "Pba2"},  {"32:cab", "P2cb"},  {"32:bca", "Pc2a"},  {"33", "Pna21"},  {"33:ba-", "Pbn21"},  
        {"33:cab", "P21nb"},  {"33:-cb", "P21cn"},  {"33:bca", "Pc21n"},  {"33:a-c", "Pn21a"},  {"34", "Pnn2"},  
        {"34:cab", "P2nn"},  {"34:bca", "Pn2n"},  {"35", "Cmm2"},  {"35:cab", "A2mm"},  {"35:bca", "Bm2m"},  
        {"36", "Cmc21"},  {"36:ba-", "Ccm21"},  {"36:cab", "A21ma"},  {"36:-cb", "A21am"},  {"36:bca", "Bb21m"},  
        {"36:a-c", "Bm21b"},  {"37", "Ccc2"},  {"37:cab", "A2aa"},  {"37:bca", "Bb2b"},  {"38", "Amm2"},  
        {"38:ba-", "Bmm2"},  {"38:cab", "B2mm"},  {"38:-cb", "C2mm"},  {"38:bca", "Cm2m"},  {"38:a-c", "Am2m"},  
        {"39", "Abm2"},  {"39:ba-", "Bma2"},  {"39:cab", "B2cm"},  {"39:-cb", "C2mb"},  {"39:bca", "Cm2a"},  
        {"39:a-c", "Ac2m"},  {"40", "Ama2"},  {"40:ba-", "Bbm2"},  {"40:cab", "B2mb"},  {"40:-cb", "C2cm"},  
        {"40:bca", "Cc2m"},  {"40:a-c", "Am2a"},  {"41", "Aba2"},  {"41:ba-", "Bba2"},  {"41:cab", "B2cb"},  
        {"41:-cb", "C2cb"},  {"41:bca", "Cc2a"},  {"41:a-c", "Ac2a"},  {"42", "Fmm2"},  {"42:cab", "F2mm"},  
        {"42:bca", "Fm2m"},  {"43", "Fdd2"},  {"43:cab", "F2dd"},  {"43:bca", "Fd2d"},  {"44", "Imm2"},  
        {"44:cab", "I2mm"},  {"44:bca", "Im2m"},  {"45", "Iba2"},  {"45:cab", "I2cb"},  {"45:bca", "Ic2a"},  
        {"46", "Ima2"},  {"46:ba-", "Ibm2"},  {"46:cab", "I2mb"},  {"46:-cb", "I2cm"},  {"46:bca", "Ic2m"},  
        {"46:a-c", "Im2a"},  {"47", "Pmmm"},  {"48:1", "Pnnn:1"},  {"48:2", "Pnnn:2"},  {"49", "Pccm"},  
        {"49:cab", "Pmaa"},  {"49:bca", "Pbmb"},  {"50:1", "Pban:1"},  {"50:2", "Pban:2"},  {"50:1ca", "Pncb:1"},  
        {"50:2ca", "Pncb:2"},  {"50:1bc", "Pcna:1"},  {"50:2bc", "Pcna:2"},  {"51", "Pmma"},  {"51:ba-", "Pmmb"},  
        {"51:cab", "Pbmm"},  {"51:-cb", "Pcmm"},  {"51:bca", "Pmcm"},  {"51:a-c", "Pmam"},  {"52", "Pnna"},  
        {"52:ba-", "Pnnb"},  {"52:cab", "Pbnn"},  {"52:-cb", "Pcnn"},  {"52:bca", "Pncn"},  {"52:a-c", "Pnan"},  
        {"53", "Pmna"},  {"53:ba-", "Pnmb"},  {"53:cab", "Pbmn"},  {"53:-cb", "Pcnm"},  {"53:bca", "Pncm"},  
        {"53:a-c", "Pman"},  {"54", "Pcca"},  {"54:ba-", "Pccb"},  {"54:cab", "Pbaa"},  {"54:-cb", "Pcaa"},  
        {"54:bca", "Pbcb"},  {"54:a-c", "Pbab"},  {"55", "Pbam"},  {"55:cab", "Pmcb"},  {"55:bca", "Pcma"},  
        {"56", "Pccn"},  {"56:cab", "Pnaa"},  {"56:bca", "Pbnb"},  {"57", "Pbcm"},  {"57:ba-", "Pcam"},  
        {"57:cab", "Pmca"},  {"57:-cb", "Pmab"},  {"57:bca", "Pbma"},  {"57:a-c", "Pcmb"},  {"58", "Pnnm"},  
        {"58:cab", "Pmnn"},  {"58:bca", "Pnmn"},  {"59:1", "Pmmn:1"},  {"59:2", "Pmmn:2"},  {"59:1ca", "Pnmm:1"},  
        {"59:2ca", "Pnmm:2"},  {"59:1bc", "Pmnm:1"},  {"59:2bc", "Pmnm:2"},  {"60", "Pbcn"},  {"60:ba-", "Pcan"},  
        {"60:cab", "Pnca"},  {"60:-cb", "Pnab"},  {"60:bca", "Pbna"},  {"60:a-c", "Pcnb"},  {"61", "Pbca"},  
        {"61:ba-", "Pcab"},  {"62", "Pnma"},  {"62:ba-", "Pmnb"},  {"62:cab", "Pbnm"},  {"62:-cb", "Pcmn"},  
        {"62:bca", "Pmcn"},  {"62:a-c", "Pnam"},  {"63", "Cmcm"},  {"63:ba-", "Ccmm"},  {"63:cab", "Amma"},  
        {"63:-cb", "Amam"},  {"63:bca", "Bbmm"},  {"63:a-c", "Bmmb"},  {"64", "Cmca"},  {"64:ba-", "Ccmb"},  
        {"64:cab", "Abma"},  {"64:-cb", "Acam"},  {"64:bca", "Bbcm"},  {"64:a-c", "Bmab"},  {"65", "Cmmm"},  
        {"65:cab", "Ammm"},  {"65:bca", "Bmmm"},  {"66", "Cccm"},  {"66:cab", "Amaa"},  {"66:bca", "Bbmb"},  
        {"67", "Cmma"},  {"67:ba-", "Cmmb"},  {"67:cab", "Abmm"},  {"67:-cb", "Acmm"},  {"67:bca", "Bmcm"},  
        {"67:a-c", "Bmam"},  {"68:1", "Ccca:1"},  {"68:2", "Ccca:2"},  {"68:1ba", "cCccb:1"},  {"68:2ba", "cCccb:2"},  
        {"68:1ca", "Abaa:1"},  {"68:2ca", "Abaa:2"},  {"68:1-c", "aAcaa:1"},  {"68:2-c", "aAcaa:2"},  {"68:1bc", "Bbcb:1"},  
        {"68:2bc", "Bbcb:2"},  {"68:1a-", "bBbab:1"},  {"68:2a-", "bBbab:2"},  {"69", "Fmmm"},  {"70:1", "Fddd:1"},  
        {"70:2", "Fddd:2"},  {"71", "Immm"},  {"72", "Ibam"},  {"72:cab", "Imcb"},  {"72:bca", "Icma"},  {"73", "Ibca"},  
        {"73:ba-", "Icab"},  {"74", "Imma"},  {"74:ba-", "Immb"},  {"74:cab", "Ibmm"},  {"74:-cb", "Icmm"},  
        {"74:bca", "Imcm"},  {"74:a-c", "Imam"},  {"75", "P4"},  {"76", "P41"},  {"77", "P42"},  {"78", "P43"},  
        {"79", "I4"},  {"80", "I41"},  {"81", "P-4"},  {"82", "I-4"},  {"83", "P4/m"},  {"84", "P42/m"},  
        {"85:1", "P4/n:1"},  {"85:2", "P4/n:2"},  {"86:1", "P42/n:1"},  {"86:2", "P42/n:2"},  {"87", "I4/m"},  
        {"88:1", "I41/a:1"},  {"88:2", "I41/a:2"},  {"89", "P422"},  {"90", "P4212"},  {"91", "P4122"},  
        {"92", "P41212"},  {"93", "P4222"},  {"94", "P42212"},  {"95", "P4322"},  {"96", "P43212"},  
        {"97", "I422"},  {"98", "I4122"},  {"99", "P4mm"},  {"100", "P4bm"},  {"101", "P42cm"},  {"102", "P42nm"},  
        {"103", "P4cc"},  {"104", "P4nc"},  {"105", "P42mc"},  {"106", "P42bc"},  {"107", "I4mm"},  {"108", "I4cm"},  
        {"109", "I41md"},  {"110", "I41cd"},  {"111", "P-42m"},  {"112", "P-42c"},  {"113", "P-421m"},  
        {"114", "P-421c"},  {"115", "P-4m2"},  {"116", "P-4c2"},  {"117", "P-4b2"},  {"118", "P-4n2"},  
        {"119", "I-4m2"},  {"120", "I-4c2"},  {"121", "I-42m"},  {"122", "I-42d"},  {"123", "P4/mmm"},  
        {"124", "P4/mcc"},  {"125:1", "P4/nbm:1"},  {"125:2", "P4/nbm:2"},  {"126:1", "P4/nnc:1"},  
        {"126:2", "P4/nnc:2"},  {"127", "P4/mbm"},  {"128", "P4/mnc"},  {"129:1", "P4/nmm:1"},  
        {"129:2", "P4/nmm:2"},  {"130:1", "P4/ncc:1"},  {"130:2", "P4/ncc:2"},  {"131", "P42/mmc"},  
        {"132", "P42/mcm"},  {"133:1", "P42/nbc:1"},  {"133:2", "P42/nbc:2"},  {"134:1", "P42/nnm:1"},  
        {"134:2", "P42/nnm:2"},  {"135", "P42/mbc"},  {"136", "P42/mnm"},  {"137:1", "P42/nmc:1"},  
        {"137:2", "P42/nmc:2"},  {"138:1", "P42/ncm:1"},  {"138:2", "P42/ncm:2"},  {"139", "I4/mmm"},  
        {"140", "I4/mcm"},  {"141:1", "I41/amd:1"},  {"141:2", "I41/amd:2"},  {"142:1", "I41/acd:1"},  
        {"142:2", "I41/acd:2"},  {"143", "P3"},  {"144", "P31"},  {"145", "P32"},  {"146:H", "R3:H"},  
        {"146:R", "R3:R"},  {"147", "P-3"},  {"148:H", "R-3:H"},  {"148:R", "R-3:R"},  {"149", "P312"},  
        {"150", "P321"},  {"151", "P3112"},  {"152", "P3121"},  {"153", "P3212"},  {"154", "P3221"},  
        {"155:H", "R32:H"},  {"155:R", "R32:R"},  {"156", "P3m1"},  {"157", "P31m"},  {"158", "P3c1"},  
        {"159", "P31c"},  {"160:H", "R3m:H"},  {"160:R", "R3m:R"},  {"161:H", "R3c:H"},  {"161:R", "R3c:R"},  
        {"162", "P-31m"},  {"163", "P-31c"},  {"164", "P-3m1"},  {"165", "P-3c1"},  {"166:H", "R-3m:H"},  
        {"166:R", "R-3m:R"},  {"167:H", "R-3c:H"},  {"167:R", "R-3c:R"},  {"168", "P6"},  {"169", "P61"},  
        {"170", "P65"},  {"171", "P62"},  {"172", "P64"},  {"173", "P63"},  {"174", "P-6"},  {"175", "P6/m"},  
        {"176", "P63/m"},  {"177", "P622"},  {"178", "P6122"},  {"179", "P6522"},  {"180", "P6222"},  
        {"181", "P6422"},  {"182", "P6322"},  {"183", "P6mm"},  {"184", "P6cc"},  {"185", "P63cm"},  
        {"186", "P63mc"},  {"187", "P-6m2"},  {"188", "P-6c2"},  {"189", "P-62m"},  {"190", "P-62c"},  
        {"191", "P6/mmm"},  {"192", "P6/mcc"},  {"193", "P63/mcm"},  {"194", "P63/mmc"},  {"195", "P23"},  
        {"196", "F23"},  {"197", "I23"},  {"198", "P213"},  {"199", "I213"},  {"200", "Pm-3"},  {"201:1", "Pn-3:1"},  
        {"201:2", "Pn-3:2"},  {"202", "Fm-3"},  {"203:1", "Fd-3:1"},  {"203:2", "Fd-3:2"},  {"204", "Im-3"},  
        {"205", "Pa-3"},  {"206", "Ia-3"},  {"207", "P432"},  {"208", "P4232"},  {"209", "F432"},  {"210", "F4132"},  
        {"211", "I432"},  {"212", "P4332"},  {"213", "P4132"},  {"214", "I4132"},  {"215", "P-43m"},  {"216", "F-43m"},  
        {"217", "I-43m"},  {"218", "P-43n"},  {"219", "F-43c"},  {"220", "I-43d"},  {"221", "Pm-3m"},  {"222:1", "Pn-3n:1"},  
        {"222:2", "Pn-3n:2"},  {"223", "Pm-3n"},  {"224:1", "Pn-3m:1"},  {"224:2", "Pn-3m:2"},  {"225", "Fm-3m"},  
        {"226", "Fm-3c"},  {"227:1", "Fd-3m:1"},  {"227:2", "Fd-3m:2"},  {"228:1", "Fd-3c:1"},  {"228:2", "Fd-3c:2"},  
        {"229", "Im-3m"},  {"230", "Ia-3d"} };
}
// bool yesno();
bool is_similar_rel(const double& first, const double& second, const double& tolerance);
bool is_similar(const double& first, const double& second, const double& tolerance);
bool is_similar_abs(const double& first, const double& second, const double& tolerance);
void cls();
std::string get_home_path(void);
void join_path(std::string& s1, std::string& s2);
inline std::streambuf* coutbuf = std::cout.rdbuf(); // save old buf
inline char asciitolower(char in)
{
    if (in <= 'Z' && in >= 'A')
        return in - ('Z' - 'z');
    return in;
}
inline void error_check(const bool condition, const std::string& file, const int& line, const std::string& function, const std::string& error_mesasge, std::ostream& log_file = std::cout)
{
    if (!condition)
    {
        log_file << "Error in " << function << " at: " << file << " : " << line << " " << error_mesasge << std::endl;
        log_file.flush();
        std::cout.rdbuf(coutbuf); // reset to standard output again
        std::cout << "Error in " << function << " at: " << file << " : " << line << " " << error_mesasge << std::endl;
        exit(-1);
    }
};
inline void not_implemented(const std::string& file, const int& line, const std::string& function, const std::string& error_mesasge, std::ostream& log_file)
{
    log_file << function << " at: " << file << ":" << line << " " << error_mesasge << " not yet implemented!" << std::endl;
    log_file.flush();
    std::cout.rdbuf(coutbuf); // reset to standard output again
    std::cout << "Error in " << function << " at: " << file << " : " << line << " " << error_mesasge << " not yet implemented!" << std::endl;
    exit(-1);
};
#define err_checkf(condition, error_message, file) error_check(condition, __FILE__, __LINE__, __func__, error_message, file)
#define err_chkf(condition, error_message, file) error_check(condition, __FILE__, __LINE__, __func__, error_message, file)
#define err_chekf(condition, error_message, file) error_check(condition, __FILE__, __LINE__, __func__, error_message, file)
#define err_not_impl_f(error_message, file) not_implemented(__FILE__, __LINE__, __func__, error_message, file)

bool generate_sph2cart_mat(std::vector<vec>& p, std::vector<vec>& d, std::vector<vec>& f, std::vector<vec>& g);
bool generate_cart2sph_mat(std::vector<vec>& d, std::vector<vec>& f, std::vector<vec>& g, std::vector<vec>& h);
std::string go_get_string(std::ifstream& file, std::string search, bool rewind = true);

inline const int sht2nbas(const int& type)
{
    const int st2bas[6]{ 1, 3, 6, 10, 15, 21 };
    const int nst2bas[6]{ 11, 9, 7, 5, 4, 1 };
    if (type >= 0)
        return st2bas[type];
    else
        return nst2bas[5 + type];
};

inline const int shell2function(const int& type, const int& prim)
{
    switch (type)
    {
    case (-5):
        return -32 + prim;
    case (-4):
        return -21 + prim;
    case (-3):
        return -12 + prim;
    case (-2):
        return -5 + prim;
    case (-1):
        return 1 + prim;
    case (0):
        return 1;
    case (1):
        return 2 + prim;
    case (2):
        return 5 + prim;
    case (3):
        if (prim == 0)
            return 11;
        if (prim == 1)
            return 12;
        if (prim == 2)
            return 13;
        if (prim == 3)
            return 17;
        if (prim == 4)
            return 14;
        if (prim == 5)
            return 15;
        if (prim == 6)
            return 18;
        if (prim == 7)
            return 19;
        if (prim == 8)
            return 16;
        if (prim == 9)
            return 20;
        break;
    case (4):
        return 21 + prim;
    case (5):
        return 36 + prim;
    default:
        return 0;
    }
    return 0;
}

const double normgauss(const int& type, const double& exp);
const double spherical_harmonic(const int& l, const int& m, const double* d);

template <class T>
std::string toString(const T& t)
{
    std::ostringstream stream;
    stream << t;
    return stream.str();
}

template <class T>
T fromString(const std::string& s)
{
    std::istringstream stream(s);
    T t;
    stream >> t;
    return t;
}

template <typename T>
void shrink_vector(std::vector<T>& g)
{
    g.clear();
    std::vector<T>(g).swap(g);
}

template <class T>
std::vector<T> split_string(const std::string& input, const std::string delimiter)
{
    std::string input_copy = input + delimiter; // Need to add one delimiter in the end to return all elements
    std::vector<T> result;
    size_t pos = 0;
    while ((pos = input_copy.find(delimiter)) != std::string::npos)
    {
        result.push_back(fromString<T>(input_copy.substr(0, pos)));
        input_copy.erase(0, pos + delimiter.length());
    }
    return result;
};

inline void remove_empty_elements(std::vector<std::string>& input, const std::string& empty = " ")
{
    for (int i = (int)input.size() - 1; i >= 0; i--)
        if (input[i] == empty || input[i] == "")
            input.erase(input.begin() + i);
}

inline std::chrono::high_resolution_clock::time_point get_time()
{
    // gets the current time using std chrono library
    std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();
    return time;
}

inline int get_musec(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
    // gets the time difference in microseconds
    std::chrono::microseconds musec = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return musec.count();
}

inline int get_msec(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
    // gets the time difference in milliseconds
    std::chrono::milliseconds msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return msec.count();
}

inline int get_sec(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
    // gets the time difference in seconds
    std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    return sec.count();
}

inline void write_timing_to_file(std::ostream& file,
    time_point start,
    time_point end,
    time_point end_prototypes,
    time_point end_becke,
    time_point end_spherical,
    time_point end_prune,
    time_point end_aspherical,
    time_point before_kpts,
    time_point after_kpts,
    time_point end1)
{
    // writes the timing of different things to a file
    using namespace std;
    int dur = get_sec(start, end);

    if (dur < 1)
        file << "Total Time: " << fixed << setprecision(0) << get_msec(start, end) << " ms\n";
    else if (dur < 60)
        file << "Total Time: " << fixed << setprecision(0) << dur << " s\n";
    else if (dur < 3600)
        file << "Total Time: " << fixed << setprecision(0) << floor(dur / 60) << " m " << dur % 60 << " s\n";
    else
        file << "Total Time: " << fixed << setprecision(0) << floor(dur / 3600) << " h " << (dur % 3600) / 60 << " m\n";
    file << endl;
    file << "Time Breakdown:" << endl;
    if (get_sec(start, end_prototypes) > 1)
        if (get_sec(start, end_prototypes) < 100)
            file << " ... for Prototype Grid setup:" << setw(6) << get_sec(start, end_prototypes) << " s " << get_msec(start, end_prototypes) % 1000 << " ms" << endl;
        else
            file << " ... for Prototype Grid setup:" << setw(6) << get_sec(start, end_prototypes) << " s" << endl;
    else
        file << " ... for Prototype Grid setup:" << setw(6) << get_msec(start, end_prototypes) << " ms" << endl;
    if (get_sec(end_prototypes, end_becke) > 1)
        if (get_sec(end_prototypes, end_becke) < 100)
            file << " ... for Becke Grid setup:    " << setw(6) << get_sec(end_prototypes, end_becke) << " s " << get_msec(end_prototypes, end_becke) % 1000 << " ms" << endl;
        else
            file << " ... for Becke Grid setup:    " << setw(6) << get_sec(end_prototypes, end_becke) << " s" << endl;
    else
        file << " ... for Becke Grid setup:    " << setw(6) << get_msec(end_prototypes, end_becke) << " ms" << endl;
    if (get_sec(end_becke, end_spherical) > 1)
        if (get_sec(end_becke, end_spherical) < 100)
            file << " ... for spherical density:   " << setw(6) << get_sec(end_becke, end_spherical) << " s " << get_msec(end_becke, end_spherical) % 1000 << " ms" << endl;
        else
            file << " ... for spherical density:   " << setw(6) << get_sec(end_becke, end_spherical) << " s" << endl;
    else
        file << " ... for spherical density:   " << setw(6) << get_msec(end_becke, end_spherical) << " ms" << endl;
    if (get_sec(end_spherical, end_prune) > 1)
        if (get_sec(end_spherical, end_prune) < 100)
            file << " ... for Grid Pruning:        " << setw(6) << get_sec(end_spherical, end_prune) << " s " << get_msec(end_spherical, end_prune) % 1000 << " ms" << endl;
        else
            file << " ... for Grid Pruning:        " << setw(6) << get_sec(end_spherical, end_prune) << " s" << endl;
    else
        file << " ... for Grid Pruning:        " << setw(6) << get_msec(end_spherical, end_prune) << " ms" << endl;
    if (get_sec(end_prune, end_aspherical) > 1)
        if (get_sec(end_prune, end_aspherical) < 100)
            file << " ... for aspherical density:  " << setw(6) << get_sec(end_prune, end_aspherical) << " s " << get_msec(end_prune, end_aspherical) % 1000 << " ms" << endl;
        else
            file << " ... for aspherical density:  " << setw(6) << get_sec(end_prune, end_aspherical) << " s" << endl;
    else
        file << " ... for aspherical density:  " << setw(6) << get_msec(end_prune, end_aspherical) << " ms" << endl;
    if (get_sec(end_aspherical, before_kpts) > 1)
        if (get_sec(end_aspherical, before_kpts) < 100)
            file << " ... for density vectors:     " << setw(6) << get_sec(end_aspherical, before_kpts) << " s " << get_msec(end_aspherical, before_kpts) % 1000 << " ms" << endl;
        else
            file << " ... for density vectors:     " << setw(6) << get_sec(end_aspherical, before_kpts) << " s" << endl;
    else
        file << " ... for density vectors:     " << setw(6) << get_msec(end_aspherical, before_kpts) << " ms" << endl;
    if (get_sec(before_kpts, after_kpts) > 1)
        if (get_sec(before_kpts, after_kpts) < 100)
            file << " ... for k-points preparation:" << setw(6) << get_sec(before_kpts, after_kpts) << " s " << get_msec(before_kpts, after_kpts) % 1000 << " ms" << endl;
        else
            file << " ... for k-points preparation:" << setw(6) << get_sec(before_kpts, after_kpts) << " s" << endl;
    else
        file << " ... for k-points preparation:" << setw(6) << get_msec(before_kpts, after_kpts) << " ms" << endl;
    if (get_sec(after_kpts, end1) > 1)
        if (get_sec(after_kpts, end1) < 100)
            file << " ... for final preparation:   " << setw(6) << get_sec(after_kpts, end1) << " s " << get_msec(after_kpts, end1) % 1000 << " ms" << endl;
        else
            file << " ... for final preparation:   " << setw(6) << get_sec(after_kpts, end1) << " s" << endl;
    else
        file << " ... for final preparation:   " << setw(6) << get_msec(after_kpts, end1) << " ms" << endl;
    if (get_sec(end1, end) > 1)
        if (get_sec(end1, end) < 100)
            file << " ... for tsc calculation:     " << setw(6) << get_sec(end1, end) << " s " << get_msec(end1, end) % 1000 << " ms" << endl;
        else
            file << " ... for tsc calculation:     " << setw(6) << get_sec(end1, end) << " s" << endl;
    else
        file << " ... for tsc calculation:     " << setw(6) << get_msec(end1, end) << " ms" << endl;
}

inline int CountWords(const char* str)
{
    if (str == NULL)
        return -1;

    bool inSpaces = true;
    int numWords = 0;

    while (*str != '\0')
    {
        if (std::isspace(*str))
        {
            inSpaces = true;
        }
        else if (inSpaces)
        {
            numWords++;
            inSpaces = false;
        }

        ++str;
    }

    return numWords;
};

inline bool exists(const std::string& name)
{
    if (FILE* file = fopen(name.c_str(), "r"))
    {
        fclose(file);
        return true;
    }
    else
    {
        return false;
    }
};

std::string atnr2letter(const int& nr);
void copy_file(std::string& from, std::string& to);
std::string shrink_string(std::string& input);
std::string shrink_string_to_atom(std::string& input, const int& atom_number);
std::string get_filename_from_path(const std::string& input);
std::string get_foldername_from_path(const std::string& input);
std::string get_basename_without_ending(const std::string& input);
//------------------Functions to read from .fchk files----------------------------------
bool read_fchk_integer_block(std::ifstream& in, std::string heading, ivec& result, bool rewind = true);
bool read_fchk_double_block(std::ifstream& in, std::string heading, vec& result, bool rewind = true);
int read_fchk_integer(std::string in);
int read_fchk_integer(std::ifstream& in, std::string search, bool rewind = true);
double read_fchk_double(std::string in);
double read_fchk_double(std::ifstream& in, std::string search, bool rewind = true);
//------------------Functions to work with configuration files--------------------------
void write_template_confi();
int program_confi(std::string& gaussian_path, std::string& turbomole_path,
    std::string& basis, int& ncpus, double& mem, bool debug = false, bool expert = false, unsigned int counter = 0);
bool check_bohr(WFN& wave, bool debug);
int filetype_identifier(std::string& file, bool debug = false);

/*bool open_file_dialog(std::string &path, bool debug, std::vector <std::string> filter);
bool save_file_dialog(std::string &path, bool debug, const std::vector<std::string> &endings, const std::string &filename_given);
bool save_file_dialog(std::string &path, bool debug, const std::vector<std::string> &endings);*/
void select_cubes(std::vector<std::vector<unsigned int>>& selection, std::vector<WFN>& wavy, unsigned int nr_of_cubes = 1, bool wfnonly = false, bool debug = false);
bool unsaved_files(std::vector<WFN>& wavy);
int get_Z_from_label(const char* tmp);

inline int sum_of_bools(const std::vector<bool> in)
{
    int result = 0;
    for (int i = 0; i < in.size(); i++)
        if (in[i])
            result++;
    return result;
}

inline std::string trim(const std::string& s)
{
    if (s == "")
        return "";
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start))
    {
        start++;
    }

    auto end = s.end();
    do
    {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return std::string(start, end + 1);
}

//-------------------------Progress_bar--------------------------------------------------

class progress_bar
{
    static const auto overhead = sizeof " [100%]";

    std::ostream& os;
    const std::size_t bar_width;
    std::string message;
    const std::string full_bar;
    const double precision;

public:
    progress_bar(std::ostream& os, std::size_t line_width,
        std::string message_, const char symbol = '=', const double p = 0.05)
        : os{ os },
        bar_width{ line_width - overhead },
        message{ std::move(message_) },
        full_bar{ std::string(bar_width, symbol) + std::string(bar_width, ' ') },
        precision{ p }
    {
        if (message.size() + 1 >= bar_width || message.find('\n') != message.npos)
        {
            os << message << '\n';
            message.clear();
        }
        else
        {
            message += ' ';
        }
        write(0.0);
    }

    // not copyable
    progress_bar(const progress_bar&) = delete;
    progress_bar& operator=(const progress_bar&) = delete;

    ~progress_bar()
    {
        write(1.0);
        os << '\n';
    }

    void write(double fraction);
};
/*

class cosinus_annaeherung
{
public:
    cosinus_annaeherung();
    inline double get(double x) const
    {
        double xa = abs(x);
        size_t pos = static_cast<size_t>((xa * mSize) / MPI2); // Stueststelle bestimmen (Wird fuer grosse X ungenau, aber passt fuer x
        double dx = xa - pos * mStepwidth;
        pos = pos % mSize; // Modulo, da sinus periodisch ist.
        double y1 = mBase_values[pos];
        double y2 = mBase_values[pos + 1];
        return y1 + dx * (y2 - y1) / mStepwidth;
    }

    void   resize(size_t size);
    double calculate_error_at(double x) const;
private:
    size_t mSize;
    double* mBase_values;
    double mStepwidth;
};
struct sinus
{
    sinus(cosinus_annaeherung& helper) : helper(helper) {};
    double get(double x) { return helper.get(x - 1.57079632679489661922l); }
    cosinus_annaeherung& helper;
};

struct cosinus
{
    cosinus(cosinus_annaeherung& helper) : helper(helper) {};
    double get(double x) { return helper.get(x); }
    cosinus_annaeherung& helper;
};
*/
void readxyzMinMax_fromWFN(
    WFN& wavy,
    double* CoordMinMax,
    int* NbSteps,
    double Radius,
    double Increments,
    bool no_bohr = false);

void readxyzMinMax_fromCIF(
    std::string cif,
    double* CoordMinMax,
    int* NbSteps,
    std::vector<std::vector<double>>& cm,
    double Resolution,
    std::ofstream& file,
    bool debug = false);

void type2vector(
    const int& index,
    int* vector);

bool read_fracs_ADPs_from_CIF(std::string cif, WFN& wavy, cell& unit_cell, std::ofstream& log3, bool debug);

inline double double_from_string_with_esd(std::string in)
{
    if (in.find('(') == std::string::npos)
        return stod(in);
    else
        return stod(in.substr(0, in.find('(')));
}

void swap_sort(ivec order, cvec& v);

void swap_sort_multi(ivec order, std::vector<ivec>& v);

// Given a 3x3 matrix in a single array of double will find and sort eigenvalues and return biggest eigenvalue
double get_lambda_1(double* a);

double get_decimal_precision_from_CIF_number(std::string& given_string);

template <typename numtype = int>
struct hashFunction
{
    size_t operator()(const std::vector<numtype>& myVector) const
    {
        std::hash<numtype> hasher;
        size_t answer = 0;
        for (numtype i : myVector)
        {
            answer ^= hasher(i) + 0x9e3779b9 + (answer << 6) + (answer >> 2);
        }
        return answer;
    }
};

template <typename numtype = int>
struct hkl_equal
{
    bool operator()(const std::vector<numtype>& vec1, const std::vector<numtype>& vec2) const
    {
        const int size = vec1.size();
        if (size != vec2.size())
            return false;
        int similar = 0;
        for (int i = 0; i < size; i++)
        {
            if (vec1[i] == vec2[i])
                similar++;
            else if (vec1[i] == -vec2[i])
                similar--;
        }
        if (abs(similar) == size)
            return true;
        else
            return false;
    }
};

template <typename numtype = int>
struct hkl_less
{
    bool operator()(const std::vector<numtype>& vec1, const std::vector<numtype>& vec2) const
    {
        if (vec1[0] < vec2[0])
        {
            return true;
        }
        else if (vec1[0] == vec2[0])
        {
            if (vec1[1] < vec2[1])
            {
                return true;
            }
            else if (vec1[1] == vec2[1])
            {
                if (vec1[2] < vec2[2])
                {
                    return true;
                }
                else
                    return false;
            }
            else
                return false;
        }
        else
            return false;
    }
};

inline unsigned int doublefactorial(int n)
{
    if (n <= 1)
        return 1;
    return n * doublefactorial(n - 2);
}

struct primitive
{
    int center, type;
    double exp, coefficient;
    double norm_const = -10;
    void normalize()
    {
        coefficient *= normalization_constant();
    };
    void unnormalize()
    {
        coefficient /= normalization_constant();
    };
    double normalization_constant()
    {
        // assuming type is equal to angular momentum
        return norm_const;
    }
    primitive() : center(0), type(0), exp(0.0), coefficient(0.0) {}
    primitive(int c, int t, double e, double coef) : center(c), type(t), exp(e), coefficient(coef)
    {
        norm_const = pow(
            pow(2, 7 + 4 * type) * pow(exp, 3 + 2 * type) / constants::PI / pow(doublefactorial(2 * type + 1), 2),
            0.25);
    }
};

struct ECP_primitive : primitive
{
    int n;
    ECP_primitive() : primitive(), n(0) {}
    ECP_primitive(int c, int t, double e, double coef, int n) : primitive(c, t, e, coef), n(n) {}
};

struct tonto_primitive
{
    int center, type;
    double exp, coefficient;
    double norm_const = -10;
    void normalize()
    {
        coefficient *= normalization_constant();
    };
    void unnormalize()
    {
        coefficient /= normalization_constant();
    };
    double normalization_constant()
    {
        // assuming type is equal to angular momentum
        return norm_const;
    }
    tonto_primitive() : center(0), type(0), exp(0.0), coefficient(0.0) {}
    tonto_primitive(int c, int t, double e, double coef) : center(c), type(t), exp(e), coefficient(coef)
    {
        norm_const = pow(constants::PI, -0.75) * pow(2.0, type + 0.75) * pow(exp, type * 0.5 + 0.75) / sqrt(doublefactorial(type));
    }
};

typedef std::set<ivec> hkl_list;
typedef std::set<ivec>::const_iterator hkl_list_it;

typedef std::set<vec> hkl_list_d;
typedef std::set<vec>::const_iterator hkl_list_it_d;

//---------------- Object for handling all input options -------------------------------
struct options
    /**
     * @brief The `options` class represents a collection of options and settings for a program.
     *
     * It contains various member variables that store different configuration parameters.
     * These parameters control the behavior and functionality of the program.
     *
     * The `options` class also provides constructors and member functions to initialize and manipulate these parameters.
     *
     * @note This class is used to configure the behavior of a specific program and may have different member variables and functions depending on the program's requirements.
     */
{
    std::ostream& log_file;
    double resolution = 0.1;
    double radius = 2.0;
    double d_sfac_scan = 0.0;
    double sfac_diffuse = 0.0;
    double dmin = 99.0;
    double mem = 0.0;
    double MinMax[6]{ 0, 0, 0, 0, 0, 0 };
    ivec MOs;
    std::vector<ivec> groups;
    std::vector<vec> twin_law;
    std::vector<ivec> combined_tsc_groups;
    std::vector<std::string> combined_tsc_calc_files;
    std::vector<std::string> combined_tsc_calc_cifs;
    std::vector<unsigned int> combined_tsc_calc_mult;
    std::vector<int> combined_tsc_calc_charge;
    std::vector<std::string> arguments;
    std::vector<std::string> combine_mo;
    std::vector<std::string> Cations;
    std::vector<std::string> Anions;
    ivec cmo1;
    ivec cmo2;
    ivec ECP_nrs;
    ivec ECP_elcounts;
    std::string wfn;
    std::string wfn2;
    std::string fchk;
    std::string basis_set;
    std::string hkl;
    std::string cif;
    std::string method;
    std::string xyz_file;
    std::string coef_file;
    std::string fract_name;
    std::string wavename;
    std::string gaussian_path;
    std::string turbomole_path;
    std::string basis_set_path;
    bool debug = false;
    bool calc = false;
    bool eli = false;
    bool esp = false;
    bool elf = false;
    bool lap = false;
    bool rdg = false;
    bool hdef = false;
    bool def = false;
    bool fract = false;
    bool hirsh = false;
    bool s_rho = false;
    bool SALTED = false, SALTED_BECKE = false;
    bool Olex2_1_3_switch = false;
    bool iam_switch = false;
    bool read_k_pts = false;
    bool save_k_pts = false;
    bool combined_tsc_calc = false;
    bool binary_tsc = true;
    bool cif_based_combined_tsc_calc = false;
    bool no_date = false;
    bool gbw2wfn = false;
    bool old_tsc = false;
    bool thakkar_d_plot = false;
    bool write_CIF = false;
    bool all_mos = false;
    bool test = false;
    bool becke = false;
    bool electron_diffraction = false;
    bool ECP = false;
    bool set_ECPs = false;
    int hirsh_number = 0;
    int NbSteps[3]{ 0, 0, 0 };
    int accuracy = 2;
    int threads = -1;
    int pbc = 0;
    int ncpus = 0;
    int charge = 0;
    int ECP_mode = 0;
    unsigned int mult = 0;
    hkl_list m_hkl_list;

    /**
     * @brief Looks for debug mode in the command line arguments.
     *
     * This function searches for a specific debug flag in the command line arguments
     * and modifies the `argc` and `argv` parameters accordingly and stores them internally
     *
     * @param argc The number of command line arguments.
     * @param argv An array of C-style strings representing the command line arguments.
     *
     */
    void look_for_debug(int& argc, char** argv);
    /**
     * @brief Digests the options.
     *
     * This function is responsible for digesting the options and performing the necessary actions based on the selected options.
     *
     * @note Make sure to call this function after looking for debug.
     */
    void digest_options();

    options() : log_file(std::cout)
    {
        groups.resize(1);
    };
    options(int& argc, char** argv, std::ostream& log) : log_file(log)
    {
        groups.resize(1);
        look_for_debug(argc, argv);
    };

    options(int accuracy, int threads, int pbc, double resolution, double radius, bool becke, bool electron_diffraction, bool ECP, bool set_ECPs, int ECP_mode, bool calc, bool eli, bool esp, bool elf, bool lap, bool rdg, bool hdef, bool def, bool fract, bool hirsh, bool s_rho, bool SALTED, bool SALTED_BECKE, bool Olex2_1_3_switch, bool iam_switch, bool read_k_pts, bool save_k_pts, bool combined_tsc_calc, bool binary_tsc, bool cif_based_combined_tsc_calc, bool density_test_cube, bool no_date, bool gbw2wfn, bool old_tsc, bool thakkar_d_plot, bool spherical_harmonic, bool ML_test, double sfac_scan, double sfac_diffuse, double dmin, int hirsh_number, const ivec& MOs, const std::vector<ivec>& groups, const std::vector<vec>& twin_law, const std::vector<ivec>& combined_tsc_groups, bool all_mos, bool test, const std::string& wfn, const std::string& fchk, const std::string& basis_set, const std::string& hkl, const std::string& cif, const std::string& method, const std::string& xyz_file, const std::string& coef_file, const std::string& fract_name, const std::vector<std::string>& combined_tsc_calc_files, const std::vector<std::string>& combined_tsc_calc_cifs, const std::string& wavename, const std::string& gaussian_path, const std::string& turbomole_path, const std::string& basis_set_path, const std::vector<std::string>& arguments, const std::vector<std::string>& combine_mo, const std::vector<std::string>& Cations, const std::vector<std::string>& Anions, const ivec& cmo1, const ivec& cmo2, const ivec& ECP_nrs, const ivec& ECP_elcounts, int ncpus, double mem, unsigned int mult, bool debug, const hkl_list& m_hkl_list, std::ostream& log_file)
        : accuracy(accuracy), threads(threads), pbc(pbc), resolution(resolution), radius(radius), becke(becke), electron_diffraction(electron_diffraction), ECP(ECP), set_ECPs(set_ECPs), ECP_mode(ECP_mode), calc(calc), eli(eli), esp(esp), elf(elf), lap(lap), rdg(rdg), hdef(hdef), def(def), fract(fract), hirsh(hirsh), s_rho(s_rho), SALTED(SALTED), SALTED_BECKE(SALTED_BECKE), Olex2_1_3_switch(Olex2_1_3_switch), iam_switch(iam_switch), read_k_pts(read_k_pts), save_k_pts(save_k_pts), combined_tsc_calc(combined_tsc_calc), binary_tsc(binary_tsc), cif_based_combined_tsc_calc(cif_based_combined_tsc_calc), no_date(no_date), gbw2wfn(gbw2wfn), old_tsc(old_tsc), thakkar_d_plot(thakkar_d_plot), d_sfac_scan(sfac_scan), sfac_diffuse(sfac_diffuse), dmin(dmin), hirsh_number(hirsh_number), MOs(MOs), groups(groups), twin_law(twin_law), combined_tsc_groups(combined_tsc_groups), all_mos(all_mos), test(test), wfn(wfn), fchk(fchk), basis_set(basis_set), hkl(hkl), cif(cif), method(method), xyz_file(xyz_file), coef_file(coef_file), fract_name(fract_name), combined_tsc_calc_files(combined_tsc_calc_files), combined_tsc_calc_cifs(combined_tsc_calc_cifs), wavename(wavename), gaussian_path(gaussian_path), turbomole_path(turbomole_path), basis_set_path(basis_set_path), arguments(arguments), combine_mo(combine_mo), Cations(Cations), Anions(Anions), cmo1(cmo1), cmo2(cmo2), ECP_nrs(ECP_nrs), ECP_elcounts(ECP_elcounts), ncpus(ncpus), mem(mem), mult(mult), debug(debug), m_hkl_list(m_hkl_list), log_file(log_file)
    {
    }
};

const double gaussian_radial(primitive& p, double& r);

const double calc_density_ML(double& x,
    double& y,
    double& z,
    vec& coefficients,
    std::vector<atom>& atoms,
    const int& exp_coefs);
const double calc_density_ML(double& x,
    double& y,
    double& z,
    vec& coefficients,
    std::vector<atom>& atoms,
    const int& exp_coefs,
    const int& atom_nr);

int load_basis_into_WFN(WFN& wavy, const std::vector<std::vector<primitive>>& b);

inline const std::vector<std::vector<primitive>> TZVP_JKfit(
    {
        {
            // center  type  exp         coef
            {0, 0, 9.5302493327, 1.0},
            {0, 0, 1.9174506246, 1.0},
            {0, 0, 0.68424049142, 1.0},
            {0, 0, 0.28413255710, 1.0},
            {0, 1, 2.9133232035, 1.0},
            {0, 1, 1.2621205398, 1.0},
            {0, 1, 0.50199775874, 1.0},
            {0, 2, 2.3135329149, 1.0},
            {0, 2, 0.71290724024, 1.0},
            {0, 3, 1.6565726132, 1.0},
        },  // H
        {}, // He
        {}, // Li
        {}, // Be
        {}, // B
        {
            {0, 0, 1113.9867719, 1.0},
            {0, 0, 369.16234180, 1.0},
            {0, 0, 121.79275232, 1.0},
            {0, 0, 48.127114540, 1.0},
            {0, 0, 20.365074004, 1.0},
            {0, 0, 8.0883596856, 1.0},
            {0, 0, 2.5068656570, 1.0},
            {0, 0, 1.2438537380, 1.0},
            {0, 0, 0.48449899601, 1.0},
            {0, 0, 0.19185160296, 1.0},
            {0, 1, 102.99176249, 1.0},
            {0, 1, 28.132594009, 1.0},
            {0, 1, 9.8364318173, 1.0},
            {0, 1, 3.3490544980, 1.0},
            {0, 1, 1.4947618613, 1.0},
            {0, 1, 0.57690108899, 1.0},
            {0, 1, 0.20320063291, 1.0},
            {0, 2, 10.594068356, 1.0},
            {0, 2, 3.5997195366, 1.0},
            {0, 2, 1.3355691094, 1.0},
            {0, 2, 0.51949764954, 1.0},
            {0, 2, 0.19954125200, 1.0},
            {0, 3, 1.194866338369, 1.0},
            {0, 3, .415866338369, 1.0},
            {0, 4, .858866338369, 1.0}}, // C
        {
            {0, 0, 1102.8622453, 1.0},
            {0, 0, 370.98041153, 1.0},
            {0, 0, 136.73555938, 1.0},
            {0, 0, 50.755871924, 1.0},
            {0, 0, 20.535656095, 1.0},
            {0, 0, 7.8318737184, 1.0},
            {0, 0, 3.4784063855, 1.0},
            {0, 0, 1.4552856603, 1.0},
            {0, 0, 0.63068989071, 1.0},
            {0, 0, 0.27276596483, 1.0},
            {0, 1, 93.540954073, 1.0},
            {0, 1, 29.524019527, 1.0},
            {0, 1, 10.917502987, 1.0},
            {0, 1, 4.3449288991, 1.0},
            {0, 1, 1.8216912640, 1.0},
            {0, 1, 0.75792424494, 1.0},
            {0, 1, 0.28241469033, 1.0},
            {0, 2, 16.419378926, 1.0},
            {0, 2, 5.0104049385, 1.0},
            {0, 2, 1.9793971884, 1.0},
            {0, 2, 0.78495771518, 1.0},
            {0, 2, 0.28954065963, 1.0},
            {0, 3, 1.79354239843, 1.0},
            {0, 3, .60854239843, 1.0},
            {0, 4, 1.23254239843, 1.0}}, // N
        {
            {0, 0, 1517.8667506, 1.0},
            {0, 0, 489.67952008, 1.0},
            {0, 0, 176.72118665, 1.0},
            {0, 0, 63.792233137, 1.0},
            {0, 0, 25.366499130, 1.0},
            {0, 0, 9.9135491200, 1.0},
            {0, 0, 4.4645306584, 1.0},
            {0, 0, 1.8017743661, 1.0},
            {0, 0, 0.80789710965, 1.0},
            {0, 0, 0.33864326862, 1.0},
            {0, 1, 120.16030921, 1.0},
            {0, 1, 34.409622474, 1.0},
            {0, 1, 12.581148610, 1.0},
            {0, 1, 5.0663824249, 1.0},
            {0, 1, 2.0346927092, 1.0},
            {0, 1, 0.86092967212, 1.0},
            {0, 1, 0.36681356726, 1.0},
            {0, 2, 19.043062805, 1.0},
            {0, 2, 5.8060381104, 1.0},
            {0, 2, 2.1891841580, 1.0},
            {0, 2, 0.87794613558, 1.0},
            {0, 2, 0.35623646700, 1.0},
            {0, 3, 2.493914788135, 1.0},
            {0, 3, .824914788135, 1.0},
            {0, 4, 1.607914788135, 1.0}}, // O
        {},                               // F
        {}                                // Ne
    });

inline const std::vector<std::vector<primitive>> QZVP_JKfit(
    {
        {// center  type  exp         coef
         {0, 0, 9.5302493327, 1.0},
         {0, 0, 1.9174506246, 1.0},
         {0, 0, 0.68424049142, 1.0},
         {0, 0, 0.28413255710, 1.0},
         {0, 1, 2.9133232035, 1.0},
         {0, 1, 1.2621205398, 1.0},
         {0, 1, 0.50199775874, 1.0},
         {0, 2, 2.8832083931, 1.0},
         {0, 2, 1.2801701725, 1.0},
         {0, 2, 0.52511317770, 1.0},
         {0, 3, 2.7489448439, 1.0},
         {0, 3, 1.1900885456, 1.0},
         {0, 4, 1.4752662714, 1.0}}, // H
        {},                          // He
        {},                          // Li
        {},                          // Be
        {},                          // B
        {
            {0, 0, 1113.9867719, 1.0},
            {0, 0, 369.16234180, 1.0},
            {0, 0, 121.79275232, 1.0},
            {0, 0, 48.127114540, 1.0},
            {0, 0, 20.365074004, 1.0},
            {0, 0, 8.0883596856, 1.0},
            {0, 0, 2.5068656570, 1.0},
            {0, 0, 1.2438537380, 1.0},
            {0, 0, 0.48449899601, 1.0},
            {0, 0, 0.19185160296, 1.0},
            {0, 1, 102.99176249, 1.0},
            {0, 1, 28.132594009, 1.0},
            {0, 1, 9.8364318173, 1.0},
            {0, 1, 3.3490544980, 1.0},
            {0, 1, 1.4947618613, 1.0},
            {0, 1, 0.57690108899, 1.0},
            {0, 1, 0.20320063291, 1.0},
            {0, 2, 10.594068356, 1.0},
            {0, 2, 3.5997195366, 1.0},
            {0, 2, 1.3355691094, 1.0},
            {0, 2, 0.51949764954, 1.0},
            {0, 2, 0.19954125200, 1.0},
            {0, 3, 1.95390000000, 1.0},
            {0, 3, .75490000000, 1.0},
            {0, 3, .33390000000, 1.0},
            {0, 4, 1.52490000000, 1.0},
            {0, 4, .59090000000, 1.0},
            {0, 5, 1.11690000000, 1.0}}, // C
        {
            {0, 0, 1102.8622453, 1.0},
            {0, 0, 370.98041153, 1.0},
            {0, 0, 136.73555938, 1.0},
            {0, 0, 50.755871924, 1.0},
            {0, 0, 20.535656095, 1.0},
            {0, 0, 7.8318737184, 1.0},
            {0, 0, 3.4784063855, 1.0},
            {0, 0, 1.4552856603, 1.0},
            {0, 0, 0.63068989071, 1.0},
            {0, 0, 0.27276596483, 1.0},
            {0, 1, 93.540954073, 1.0},
            {0, 1, 29.524019527, 1.0},
            {0, 1, 10.917502987, 1.0},
            {0, 1, 4.3449288991, 1.0},
            {0, 1, 1.8216912640, 1.0},
            {0, 1, 0.75792424494, 1.0},
            {0, 1, 0.28241469033, 1.0},
            {0, 2, 16.419378926, 1.0},
            {0, 2, 5.0104049385, 1.0},
            {0, 2, 1.9793971884, 1.0},
            {0, 2, 0.78495771518, 1.0},
            {0, 2, 0.28954065963, 1.0},
            {0, 3, 2.98600000000, 1.0},
            {0, 3, 1.11700000000, 1.0},
            {0, 3, .48400000000, 1.0},
            {0, 4, 2.17600000000, 1.0},
            {0, 4, .83400000000, 1.0},
            {0, 5, 1.57600000000, 1.0}}, // N
        {
            {0, 0, 1517.8667506, 1.0},
            {0, 0, 489.67952008, 1.0},
            {0, 0, 176.72118665, 1.0},
            {0, 0, 63.792233137, 1.0},
            {0, 0, 25.366499130, 1.0},
            {0, 0, 9.9135491200, 1.0},
            {0, 0, 4.4645306584, 1.0},
            {0, 0, 1.8017743661, 1.0},
            {0, 0, 0.80789710965, 1.0},
            {0, 0, 0.33864326862, 1.0},
            {0, 1, 120.16030921, 1.0},
            {0, 1, 34.409622474, 1.0},
            {0, 1, 12.581148610, 1.0},
            {0, 1, 5.0663824249, 1.0},
            {0, 1, 2.0346927092, 1.0},
            {0, 1, 0.86092967212, 1.0},
            {0, 1, 0.36681356726, 1.0},
            {0, 2, 19.043062805, 1.0},
            {0, 2, 5.8060381104, 1.0},
            {0, 2, 2.1891841580, 1.0},
            {0, 2, 0.87794613558, 1.0},
            {0, 2, 0.35623646700, 1.0},
            {0, 3, 3.96585000000, 1.0},
            {0, 3, 1.49085000000, 1.0},
            {0, 3, .63485000000, 1.0},
            {0, 4, 2.85685000000, 1.0},
            {0, 4, 1.04985000000, 1.0},
            {0, 5, 2.03685000000, 1.0}}, // O
        {},                              // F
        {}                               // Ne
    });

inline double hypergeometric(double a, double b, double c, double x)
{
    const double TOLERANCE = 1.0e-10;
    double term = a * b * x / c;
    double value = 1.0 + term;
    int n = 1;

    while (std::abs(term) > TOLERANCE)
    {
        a++, b++, c++, n++;
        term *= a * b * x / c / n;
        value += term;
    }

    return value;
}

inline cdouble hypergeometric(double a, double b, double c, cdouble x)
{
    const double TOLERANCE = 1.0e-10;
    cdouble term = a * b * x / c;
    cdouble value = 1.0 + term;
    int n = 1;

    while (std::abs(term) > TOLERANCE)
    {
        a++, b++, c++, n++;
        term *= a * b * x / c / static_cast<double>(n);
        value += term;
    }

    return value;
}

inline bool ends_with(const std::string& str, const std::string& suffix)
{
    if (str.length() >= suffix.length())
    {
        return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
    }
    return false;
}

template <typename T>
inline bool is_nan(T in)
{
    if (typeid(in) == typeid(double))
    {
        return in != in;
    }
    else if (typeid(in) == typeid(float))
    {
        return in != in;
    }
    else if (typeid(in) == typeid(long double))
    {
        return in != in;
    }
    else
    {
        return false;
    }
}

#include "wfn_class.h"
#include "atoms.h"