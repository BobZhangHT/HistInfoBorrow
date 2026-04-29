/* ===================================================================
 * radish_core.c — Native (C) implementations of KBCD / CAHB / RADISH
 * hot loops, exposed via a flat C ABI for ctypes.
 *
 * Numerical contract: every routine here is a 1-to-1 port of the
 * corresponding Python in methods.py, using identical formulas, the
 * same epsilon (_EPS = 1e-12) and the same numerical safeguards.
 *
 * Conventions
 * -----------
 *   - All floating-point arrays are double, C-contiguous (row-major).
 *   - Matrices A of shape (m, n) are stored as A[i*n + j].
 *   - Kernel weight matrix returned to Python has shape (n_eval, n_ref):
 *       K[i, j] = kernel( Xref[j] - Xeval[i] )
 *     matching gaussian_weight_matrix() in methods.py.
 *   - Symbol exports use EXPORT macro for Windows DLL portability.
 * ================================================================= */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

static const double EPS = 1e-12;

/* --------- small numeric helpers --------- */
static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dclip(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* Phi (standard normal CDF) — Abramowitz–Stegun 7.1.26 / erf-based.
 * matches scipy.stats.norm.cdf to ~1e-15.                          */
static inline double phi_cdf(double z) {
    return 0.5 * erfc(-z * 0.7071067811865475);  /* z / sqrt(2) */
}

/* ---------- Squared scaled distance for product Gaussian kernel:
 *   d2(Xa[i], Xb[j]) = sum_k ((Xa[i,k] - Xb[j,k]) / h[k])^2          */
static inline double squared_scaled_dist(
    const double *xa, const double *xb,
    const double *h, int p)
{
    double s = 0.0;
    for (int k = 0; k < p; ++k) {
        double d = (xa[k] - xb[k]) / h[k];
        s += d * d;
    }
    return s;
}

/* =================================================================
 * KERNEL WEIGHT MATRICES
 * ================================================================= */

/* Gaussian (un-normalized) kernel: K[i,j] = exp(-0.5 * ||(Xref[j]-Xeval[i])/h||^2)
 * Matches gaussian_weight_matrix() in methods.py.                  */
EXPORT void kernel_gauss_matrix(
    const double *Xref, int n_ref,
    const double *Xeval, int n_eval,
    int p, const double *h,
    double *K)
{
    for (int i = 0; i < n_eval; ++i) {
        const double *xe = Xeval + (size_t)i * p;
        double *Krow = K + (size_t)i * n_ref;
        for (int j = 0; j < n_ref; ++j) {
            double s = squared_scaled_dist(xe, Xref + (size_t)j*p, h, p);
            Krow[j] = exp(-0.5 * s);
        }
    }
}

/* Gaussian density kernel with full Gaussian normalization, used by
 * CAHB._km in methods.py:  ln_norm = -0.5*(p*log(2π) + sum log h^2). */
EXPORT void kernel_gauss_density_matrix(
    const double *Xa, int na,
    const double *Xb, int nb,
    int p, const double *h_in,
    double *K)
{
    /* clamp h to >= 1e-3 (mirrors Python) */
    double h_safe[16];
    if (p > 16) {
        /* Use heap fallback (rare; bivariate covariate in our setup) */
        double *h_heap = (double *)malloc(sizeof(double) * (size_t)p);
        for (int k = 0; k < p; ++k)
            h_heap[k] = h_in[k] < 1e-3 ? 1e-3 : h_in[k];
        double ld = 0.0;
        for (int k = 0; k < p; ++k) ld += log(h_heap[k] * h_heap[k]);
        double ln_norm = -0.5 * ((double)p * log(2.0 * M_PI) + ld);
        for (int i = 0; i < nb; ++i) {
            const double *xb = Xb + (size_t)i * p;
            double *Krow = K + (size_t)i * na;
            for (int j = 0; j < na; ++j) {
                double s = squared_scaled_dist(xb, Xa + (size_t)j*p, h_heap, p);
                Krow[j] = exp(-0.5 * s + ln_norm);
            }
        }
        free(h_heap);
        return;
    }
    for (int k = 0; k < p; ++k)
        h_safe[k] = h_in[k] < 1e-3 ? 1e-3 : h_in[k];
    double ld = 0.0;
    for (int k = 0; k < p; ++k) ld += log(h_safe[k] * h_safe[k]);
    double ln_norm = -0.5 * ((double)p * log(2.0 * M_PI) + ld);
    for (int i = 0; i < nb; ++i) {
        const double *xb = Xb + (size_t)i * p;
        double *Krow = K + (size_t)i * na;
        for (int j = 0; j < na; ++j) {
            double s = squared_scaled_dist(xb, Xa + (size_t)j*p, h_safe, p);
            Krow[j] = exp(-0.5 * s + ln_norm);
        }
    }
}

/* In-place row normalization (each row sums to 1; rows that sum to <EPS
 * are scaled by 1/EPS as in methods.normalise_rows).                 */
EXPORT void normalize_rows(double *K, int nrow, int ncol) {
    for (int i = 0; i < nrow; ++i) {
        double *r = K + (size_t)i * ncol;
        double s = 0.0;
        for (int j = 0; j < ncol; ++j) s += r[j];
        double denom = s > EPS ? s : EPS;
        double inv = 1.0 / denom;
        for (int j = 0; j < ncol; ++j) r[j] *= inv;
    }
}

/* Gaussian (un-normalized) kernel weight VECTOR for one query x_star.
 * Used by methods.gaussian_weights().                                */
EXPORT void kernel_gauss_vec(
    const double *Xref, int n_ref, int p,
    const double *x_star, const double *h,
    double *w)
{
    for (int j = 0; j < n_ref; ++j) {
        double s = squared_scaled_dist(x_star, Xref + (size_t)j*p, h, p);
        w[j] = exp(-0.5 * s);
    }
}

/* Product Epanechnikov kernel weight VECTOR.
 *   k(u) = max(0, 1 - u^2) * 0.75, product over dimensions.          */
EXPORT void kernel_epan_vec(
    const double *Xref, int n_ref, int p,
    const double *x_star, const double *h_in,
    double *w)
{
    /* clamp h to >= EPS as in Python */
    double h_safe[16]; double *h = h_safe; int alloc = 0;
    if (p > 16) { h = (double *)malloc(sizeof(double)*(size_t)p); alloc = 1; }
    for (int k = 0; k < p; ++k) h[k] = h_in[k] < EPS ? EPS : h_in[k];
    for (int j = 0; j < n_ref; ++j) {
        const double *xj = Xref + (size_t)j*p;
        double prod = 1.0;
        for (int k = 0; k < p; ++k) {
            double u = (xj[k] - x_star[k]) / h[k];
            double v = 1.0 - u * u;
            prod *= (v > 0.0 ? v : 0.0) * 0.75;
            if (prod == 0.0) break;
        }
        w[j] = prod;
    }
    if (alloc) free(h);
}

/* =================================================================
 * STAGE-III: linearized ATE estimator (shared by all methods)
 *
 *   a1 = mean_i normalize(K(X1; X[i]))
 *   a0 = mean_i (1 - W[i]) * normalize(K(X0; X[i]))
 *   aH = mean_i  W[i]      * normalize(K(Xh; X[i]))
 *   ATE = a1·Y1 - a0·Y0 - aH·Yh
 *
 * Variance σ̂² computed via residual kernel smoother per arm:
 *   S_self = normalize(K(Xa; Xa)),  σ̂² = ||Y - S·Y||² / (n - tr(S))
 * ================================================================= */

/* Build *normalized* kernel matrix S[n_eval, n_ref] in one pass with
 * minimal temp memory.                                                */
static void build_normalized_S(
    const double *Xref, int n_ref,
    const double *Xeval, int n_eval,
    int p, const double *h,
    double *S)   /* (n_eval, n_ref) */
{
    kernel_gauss_matrix(Xref, n_ref, Xeval, n_eval, p, h, S);
    normalize_rows(S, n_eval, n_ref);
}

/* Residual variance for one arm (matches methods._resid_var).
 * Returns max(_EPS, sum((Y - SY)^2) / (n - tr(S))).                  */
static double residual_variance(
    const double *Xa, const double *Ya, int na,
    int p, const double *h)
{
    if (na < 3) {
        if (na <= 1) return EPS;
        double mean = 0.0;
        for (int i = 0; i < na; ++i) mean += Ya[i];
        mean /= na;
        double s = 0.0;
        for (int i = 0; i < na; ++i) { double d = Ya[i] - mean; s += d*d; }
        s /= (double)(na - 1);
        return s > EPS ? s : EPS;
    }
    double *S = (double *)malloc(sizeof(double) * (size_t)na * (size_t)na);
    if (!S) return EPS;
    build_normalized_S(Xa, na, Xa, na, p, h, S);
    double tr = 0.0;
    double sse = 0.0;
    for (int i = 0; i < na; ++i) {
        double *row = S + (size_t)i * na;
        tr += row[i];
        double pred = 0.0;
        for (int j = 0; j < na; ++j) pred += row[j] * Ya[j];
        double r = Ya[i] - pred;
        sse += r * r;
    }
    free(S);
    double dof = (double)na - tr;
    if (dof < 1.0) dof = 1.0;
    double v = sse / dof;
    return v > EPS ? v : EPS;
}

/* ATEResult mirrors estimate_ate() return tuple.                    */
typedef struct {
    double ate;
    double ci_low;
    double ci_high;
    double prob_pos;
    double V;
} ATEResult;

EXPORT void estimate_ate_c(
    const double *Xeval, int n_eval, int p,
    const double *X0, const double *Y0, int n0,
    const double *X1, const double *Y1, int n1,
    const double *Xh, const double *Yh, int nh,
    const double *h, const double *W_vec,
    double alpha,
    /* outputs */
    double *out_ate, double *out_lo, double *out_hi,
    double *out_pp,  double *out_V)
{
    if (n0 < 2 || n1 < 2) {
        double m1 = 0.0, m0 = 0.0;
        if (n1 > 0) { for (int i = 0; i < n1; ++i) m1 += Y1[i]; m1 /= n1; }
        if (n0 > 0) { for (int i = 0; i < n0; ++i) m0 += Y0[i]; m0 /= n0; }
        *out_ate = m1 - m0;
        *out_lo = NAN; *out_hi = NAN; *out_pp = 0.5; *out_V = NAN;
        return;
    }

    /* Allocate three n_eval × n_ref kernel matrices */
    double *S0 = (double *)malloc(sizeof(double) * (size_t)n_eval * (size_t)n0);
    double *S1 = (double *)malloc(sizeof(double) * (size_t)n_eval * (size_t)n1);
    double *SH = (double *)malloc(sizeof(double) * (size_t)n_eval * (size_t)nh);
    if (!S0 || !S1 || !SH) { free(S0); free(S1); free(SH);
        *out_ate = NAN; *out_lo=NAN; *out_hi=NAN; *out_pp=0.5; *out_V=NAN; return; }
    build_normalized_S(X0, n0, Xeval, n_eval, p, h, S0);
    build_normalized_S(X1, n1, Xeval, n_eval, p, h, S1);
    build_normalized_S(Xh, nh, Xeval, n_eval, p, h, SH);

    /* Influence vectors a1, a0, aH (averaged across evaluation points) */
    double *a1 = (double *)calloc((size_t)n1, sizeof(double));
    double *a0 = (double *)calloc((size_t)n0, sizeof(double));
    double *aH = (double *)calloc((size_t)nh, sizeof(double));
    if (!a1 || !a0 || !aH) { free(S0); free(S1); free(SH);
        free(a1); free(a0); free(aH);
        *out_ate=NAN; *out_lo=NAN; *out_hi=NAN; *out_pp=0.5; *out_V=NAN; return; }

    double inv_n_eval = 1.0 / (double)n_eval;
    for (int i = 0; i < n_eval; ++i) {
        double Wi = W_vec[i];
        double oneW = 1.0 - Wi;
        double *r1 = S1 + (size_t)i * n1;
        double *r0 = S0 + (size_t)i * n0;
        double *rH = SH + (size_t)i * nh;
        for (int k = 0; k < n1; ++k) a1[k] += r1[k];
        for (int k = 0; k < n0; ++k) a0[k] += oneW * r0[k];
        for (int k = 0; k < nh; ++k) aH[k] += Wi  * rH[k];
    }
    for (int k = 0; k < n1; ++k) a1[k] *= inv_n_eval;
    for (int k = 0; k < n0; ++k) a0[k] *= inv_n_eval;
    for (int k = 0; k < nh; ++k) aH[k] *= inv_n_eval;

    /* Point estimate */
    double t1 = 0.0, t0 = 0.0, tH = 0.0;
    double n1n = 0.0, n0n = 0.0, nHn = 0.0;
    for (int k = 0; k < n1; ++k) { t1 += a1[k]*Y1[k]; n1n += a1[k]*a1[k]; }
    for (int k = 0; k < n0; ++k) { t0 += a0[k]*Y0[k]; n0n += a0[k]*a0[k]; }
    for (int k = 0; k < nh; ++k) { tH += aH[k]*Yh[k]; nHn += aH[k]*aH[k]; }
    double ate = t1 - t0 - tH;

    /* Sandwich variance with kernel-residual estimators */
    double v1 = residual_variance(X1, Y1, n1, p, h);
    double v0 = residual_variance(X0, Y0, n0, p, h);
    double vH = residual_variance(Xh, Yh, nh, p, h);
    double V = v1*n1n + v0*n0n + vH*nHn;
    if (V < EPS) V = EPS;
    double se = sqrt(V);
    /* z = invPhi(1 - alpha/2) — caller passes 1.96 via alpha=0.05 */
    /* compute standard normal inverse for 1 - alpha/2 */
    /* Use Wichura’s algorithm AS 241 — but we only need a few alpha values.
     * We accept a small library: use the Beasley-Springer-Moro approximation. */
    double q = 1.0 - alpha / 2.0;
    /* Acklam's inverse cdf approximation */
    double a_[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                    -2.759285104469687e+02, 1.383577518672690e+02,
                    -3.066479806614716e+01, 2.506628277459239e+00};
    double b_[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                    -1.556989798598866e+02, 6.680131188771972e+01,
                    -1.328068155288572e+01};
    double c_[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                    -2.400758277161838e+00, -2.549732539343734e+00,
                    4.374664141464968e+00,  2.938163982698783e+00};
    double d_[4] = {7.784695709041462e-03,  3.224671290700398e-01,
                    2.445134137142996e+00,  3.754408661907416e+00};
    double pl = 0.02425, ph = 1.0 - pl;
    double z;
    if (q < pl) {
        double r = sqrt(-2.0*log(q));
        z = (((((c_[0]*r+c_[1])*r+c_[2])*r+c_[3])*r+c_[4])*r+c_[5]) /
            ((((d_[0]*r+d_[1])*r+d_[2])*r+d_[3])*r+1.0);
    } else if (q <= ph) {
        double r = q - 0.5; double r2 = r*r;
        z = (((((a_[0]*r2+a_[1])*r2+a_[2])*r2+a_[3])*r2+a_[4])*r2+a_[5])*r /
            (((((b_[0]*r2+b_[1])*r2+b_[2])*r2+b_[3])*r2+b_[4])*r2+1.0);
    } else {
        double r = sqrt(-2.0*log(1.0-q));
        z = -(((((c_[0]*r+c_[1])*r+c_[2])*r+c_[3])*r+c_[4])*r+c_[5]) /
             ((((d_[0]*r+d_[1])*r+d_[2])*r+d_[3])*r+1.0);
    }
    *out_ate = ate;
    *out_lo  = ate - z*se;
    *out_hi  = ate + z*se;
    *out_pp  = (se > EPS) ? phi_cdf(ate / se) : 0.5;
    *out_V   = V;

    free(S0); free(S1); free(SH);
    free(a1); free(a0); free(aH);
}

/* =================================================================
 * RADISH: per-evaluation-point diagnostics & W vector
 * ================================================================= */

/* Compute (R_n, W, D_PDC, tau²_H) at a single evaluation point.
 * Mirrors RADISH per-iteration computation in methods.py.
 *   R_n = 1 + sigma2_0c / (Nc * tau2_H * exp(D_pdc))
 *   W   = (R_n - 1) / R_n
 * with safeguards: Nc_s = max(Nc, nc_floor), R_n clipped to [1, 1e4].
 */
static void radish_single_eval(
    const double *X0, const double *Y0, int n0,
    const double *Xh, const double *Yh, int nh,
    int p, const double *h,
    const double *x_eval, double nc_floor,
    double *out_Rn, double *out_W,
    double *out_Dpdc, double *out_tau2H)
{
    /* Stage-I local historical prior */
    double *wH = (double *)malloc(sizeof(double) * (size_t)nh);
    kernel_gauss_vec(Xh, nh, p, x_eval, h, wH);
    double swH = 0.0, swH2 = 0.0, sumH = 0.0;
    for (int j = 0; j < nh; ++j) { swH += wH[j]; }

    double theta, tau2H;
    if (swH <= EPS) {
        /* fallback: global mean / large variance */
        double m = 0.0;
        for (int j = 0; j < nh; ++j) m += Yh[j];
        m /= nh;
        theta = m;
        tau2H = 1e6;
    } else {
        for (int j = 0; j < nh; ++j) sumH += wH[j] * Yh[j];
        theta = sumH / swH;
        double s2 = 0.0;
        for (int j = 0; j < nh; ++j) {
            double d = Yh[j] - theta;
            s2 += wH[j] * d * d;
        }
        s2 /= swH;
        if (s2 < EPS) s2 = EPS;
        for (int j = 0; j < nh; ++j) swH2 += wH[j] * wH[j];
        double neff = (swH * swH) / (swH2 > EPS ? swH2 : EPS);
        if (neff < 1e-6) neff = 1e-6;
        tau2H = s2 / neff;
        if (tau2H < EPS) tau2H = EPS;
    }
    free(wH);

    /* Stage-II concurrent-control summary */
    double *w0 = (double *)malloc(sizeof(double) * (size_t)(n0 > 0 ? n0 : 1));
    double Nc = 0.0, ybar = 0.0, s2c = 1.0;
    if (n0 > 0) {
        kernel_gauss_vec(X0, n0, p, x_eval, h, w0);
        for (int i = 0; i < n0; ++i) Nc += w0[i];
        if (Nc > EPS) {
            double sy = 0.0;
            for (int i = 0; i < n0; ++i) sy += w0[i] * Y0[i];
            ybar = sy / Nc;
            double s2 = 0.0;
            for (int i = 0; i < n0; ++i) {
                double d = Y0[i] - ybar;
                s2 += w0[i] * d * d;
            }
            s2 /= Nc;
            s2c = s2 > EPS ? s2 : EPS;
        } else {
            ybar = 0.0; s2c = 1.0;
        }
    }
    free(w0);

    /* PDC discrepancy */
    double tau = sqrt(tau2H > EPS ? tau2H : EPS);
    double Z = (ybar - theta) / tau;
    if (Z >  1e10) Z =  1e10;
    if (Z < -1e10) Z = -1e10;
    double pu = phi_cdf(Z);
    double pmin = pu < (1.0 - pu) ? pu : (1.0 - pu);
    double pn = 2.0 * pmin;
    if (pn < EPS) pn = EPS;
    double Dn = -log(pn);

    /* Information ratio + W */
    double Nc_s = Nc > nc_floor ? Nc : nc_floor;
    double gD = exp(Dn);
    double PiH = 1.0 / dmax(tau2H * gD, EPS);
    double PiC = dmax(Nc_s / s2c, EPS);
    double Rn = 1.0 + PiH / PiC;
    if (Rn < 1.0) Rn = 1.0;
    if (Rn > 1e4) Rn = 1e4;
    double W = (Rn - 1.0) / Rn;

    *out_Rn = Rn; *out_W = W; *out_Dpdc = Dn; *out_tau2H = tau2H;
}

/* Batch version: produces R_n[i], W[i], D_pdc[i], tau²_H[i] for each X[i].
 * Pass NULL for any output you don't need.                             */
EXPORT void radish_diagnostics_batch(
    const double *X, int n,
    const double *X0, const double *Y0, int n0,
    const double *Xh, const double *Yh, int nh,
    int p, const double *h, double nc_floor,
    double *out_Rn, double *out_W,
    double *out_Dpdc, double *out_tau2H)
{
    for (int i = 0; i < n; ++i) {
        double Rn, W, D, t2;
        radish_single_eval(X0, Y0, n0, Xh, Yh, nh, p, h,
                           X + (size_t)i * p, nc_floor,
                           &Rn, &W, &D, &t2);
        if (out_Rn)    out_Rn[i]    = Rn;
        if (out_W)     out_W[i]     = W;
        if (out_Dpdc)  out_Dpdc[i]  = D;
        if (out_tau2H) out_tau2H[i] = t2;
    }
}

/* Stage-II allocation probability for a NEW patient at x_new.
 * Returns scalar pi (allocation prob to treatment arm).               */
EXPORT double radish_allocation_prob(
    const double *X_curr, const double *Y_curr, const int *Z_curr, int n_curr,
    const double *Xh, const double *Yh, int nh,
    int p, const double *h_fit, const double *h_alloc,
    const double *x_new, double nc_floor)
{
    if (n_curr < 2) return 0.5;

    /* Build X0/Y0 (control subset of current trial) */
    int n0 = 0;
    for (int i = 0; i < n_curr; ++i) if (Z_curr[i] == 0) ++n0;
    double *X0 = NULL, *Y0 = NULL;
    if (n0 > 0) {
        X0 = (double *)malloc(sizeof(double)*(size_t)n0*p);
        Y0 = (double *)malloc(sizeof(double)*(size_t)n0);
        int k = 0;
        for (int i = 0; i < n_curr; ++i) {
            if (Z_curr[i] == 0) {
                memcpy(X0 + (size_t)k*p, X_curr + (size_t)i*p, sizeof(double)*p);
                Y0[k] = Y_curr[i];
                ++k;
            }
        }
    }

    double Rn, W, D, t2;
    radish_single_eval(X0, Y0, n0, Xh, Yh, nh, p, h_fit, x_new,
                       nc_floor, &Rn, &W, &D, &t2);
    free(X0); free(Y0);

    /* Allocation: epanechnikov-weighted local arm sizes */
    double *wa = (double *)malloc(sizeof(double)*(size_t)n_curr);
    kernel_epan_vec(X_curr, n_curr, p, x_new, h_alloc, wa);
    double N0 = 0.0, N1 = 0.0;
    for (int i = 0; i < n_curr; ++i) {
        if (Z_curr[i] == 0) N0 += wa[i];
        else                N1 += wa[i];
    }
    free(wa);

    double n0e = Rn * N0;
    if (n0e < EPS) n0e = EPS;
    if (N1  < EPS) N1  = EPS;
    double pi = (n0e * n0e) / (n0e * n0e + N1 * N1);
    if (pi < 0.0) pi = 0.0;
    if (pi > 1.0) pi = 1.0;
    return pi;
}

/* =================================================================
 * KBCD allocation probability
 *   No borrowing; just balances local arm sizes using Epanechnikov.
 * ================================================================= */
EXPORT double kbcd_allocation_prob(
    const double *X_curr, const int *Z_curr, int n_curr, int p,
    const double *h_alloc, const double *x_new)
{
    if (n_curr < 2) return 0.5;
    double *wa = (double *)malloc(sizeof(double)*(size_t)n_curr);
    kernel_epan_vec(X_curr, n_curr, p, x_new, h_alloc, wa);
    double N0 = 0.0, N1 = 0.0;
    for (int i = 0; i < n_curr; ++i) {
        if (Z_curr[i] == 0) N0 += wa[i];
        else                N1 += wa[i];
    }
    free(wa);
    if (N0 < EPS) N0 = EPS;
    if (N1 < EPS) N1 = EPS;
    double pi = (N0*N0) / (N0*N0 + N1*N1);
    if (pi < 0.0) pi = 0.0;
    if (pi > 1.0) pi = 1.0;
    return pi;
}

/* =================================================================
 * CAHB: coordinate-ascent fit + R_n predictor
 * ================================================================= */

/* Helper: kernel-weighted historical mean θ₀(x) at each X_eval[i]. */
static void cahb_hist_mean(
    const double *Xh, const double *Yh, int nh,
    const double *Xe, int ne, int p, const double *h_hist,
    double *out)
{
    double *w = (double *)malloc(sizeof(double)*(size_t)nh);
    for (int i = 0; i < ne; ++i) {
        kernel_gauss_vec(Xh, nh, p, Xe + (size_t)i*p, h_hist, w);
        double s = 0.0;
        for (int j = 0; j < nh; ++j) s += w[j];
        if (s > EPS) {
            double sy = 0.0;
            for (int j = 0; j < nh; ++j) sy += w[j] * Yh[j];
            out[i] = sy / s;
        } else {
            out[i] = 0.0;
        }
    }
    free(w);
}

/* μ₀ update: μ₀(x_i) = num/den (matches CAHB._mu0).
 * Pre-compute M[i], W[i] for control patients, then loop over j.    */
static void cahb_update_mu0(
    const double *KM, const double *Y, const double *Z, int n,
    const double *tau, double phi, const double *theta0,
    double *mu0_out)
{
    double inv2phi2 = 0.5 / (phi * phi);
    /* Cache control indices and per-control M/W values once.        */
    int *idx0 = (int *)malloc(sizeof(int)*(size_t)n);
    double *Mv = (double *)malloc(sizeof(double)*(size_t)n);
    double *Wv = (double *)malloc(sizeof(double)*(size_t)n);
    int n0 = 0;
    for (int i = 0; i < n; ++i) {
        if (Z[i] == 0.0) {
            Mv[n0] = Y[i] * inv2phi2 + 0.5 * tau[i] * theta0[i];
            Wv[n0] = inv2phi2 + 0.5 * tau[i];
            idx0[n0++] = i;
        }
    }
    for (int j = 0; j < n; ++j) {
        double num = 0.0, den = 0.0;
        for (int t = 0; t < n0; ++t) {
            int i = idx0[t];
            double wij = KM[(size_t)i*n + j];
            num += wij * Mv[t];
            den += wij * Wv[t];
        }
        mu0_out[j] = (den > EPS) ? (num / den) : 0.0;
    }
    free(idx0); free(Mv); free(Wv);
}

static double cahb_update_phi(
    const double *Y, const double *Z, const double *mu0, int n)
{
    double sum_sZ = 0.0, sse = 0.0;
    for (int i = 0; i < n; ++i) {
        if (Z[i] != 0.0) continue;
        double r = Y[i] - mu0[i];
        sse += r * r; sum_sZ += 1.0;
    }
    double v = (0.5 * sse + 0.01) / (1.01 + 0.5 * sum_sZ);
    if (v < EPS) v = EPS;
    return sqrt(v);
}

/* Raw τ vector (before truncation/L1 projection) — matches _tau_raw.
 * Pre-computes (mu0-theta0)^2 over control patients only.            */
static void cahb_tau_raw(
    const double *KM, const double *Z, const double *mu0,
    const double *theta0, int n, double invg2, double *raw)
{
    int *idx0 = (int *)malloc(sizeof(int)*(size_t)n);
    double *dsqv = (double *)malloc(sizeof(double)*(size_t)n);
    int n0 = 0;
    for (int i = 0; i < n; ++i) {
        if (Z[i] == 0.0) {
            double d = mu0[i] - theta0[i];
            dsqv[n0] = d * d;
            idx0[n0++] = i;
        }
    }
    for (int j = 0; j < n; ++j) {
        double num = 0.0, den = 0.0;
        for (int t = 0; t < n0; ++t) {
            int i = idx0[t];
            double wij = KM[(size_t)i*n + j];
            num += wij;
            den += wij * dsqv[t];
        }
        den += invg2;
        double v = num / (den > EPS ? den : EPS);
        if (v < 0.0) v = 0.0;
        raw[j] = v;
    }
    free(idx0); free(dsqv);
}

/* qsort double descending */
static int dcmp_desc(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) - (x > y);
}

/* L1 simplex projection then truncation to bound ||τ||_1 ≤ r — matches _tau. */
static void cahb_tau_full(
    const double *KM, const double *Z, const double *mu0,
    const double *theta0, int n, double invg2,
    double lt, double lam, double *tau_out)
{
    cahb_tau_raw(KM, Z, mu0, theta0, n, invg2, tau_out);
    /* threshold by lambda_trunc */
    for (int j = 0; j < n; ++j) if (tau_out[j] <= lt) tau_out[j] = 0.0;
    /* L1 projection */
    double sum = 0.0;
    for (int j = 0; j < n; ++j) sum += tau_out[j];
    double r = lam * dmax(log((double)dmax(n, 1)), 1.0);
    if (!isfinite(r) || sum <= r) return;

    /* Sort descending copy */
    double *u = (double *)malloc(sizeof(double)*(size_t)n);
    memcpy(u, tau_out, sizeof(double)*(size_t)n);
    qsort(u, n, sizeof(double), dcmp_desc);
    /* Find largest index k such that u[k]*(k+1) > (cs[k] - r) */
    double cs = 0.0;
    int last = -1; double last_cs = 0.0;
    for (int k = 0; k < n; ++k) {
        cs += u[k];
        if (u[k] * (double)(k + 1) > (cs - r)) {
            last = k; last_cs = cs;
        }
    }
    if (last >= 0) {
        double theta_proj = (last_cs - r) / (double)(last + 1);
        for (int j = 0; j < n; ++j) {
            double v = tau_out[j] - theta_proj;
            tau_out[j] = v > 0.0 ? v : 0.0;
        }
    }
    free(u);
}

/* Posterior variance Var(μ₀(x)) at evaluation points Xe (matches _var_mu0).
 *   rv[k] = Σ_i KM(Xe[k], X[i]) * sZ[i] * (1/φ² + (use_tau ? tau[i] : 0))
 *   var = clamp_to_max(1/rv, 10*φ²)
 * Hoisted ln_norm and h-clamp out of inner loop for ~2× speedup.    */
static void cahb_var_mu0(
    const double *Xe, int ne,
    const double *X,  int n,  int p, const double *h_fit,
    const double *Z,  const double *tau,
    double phi_sq, int use_tau,
    double *out_var)
{
    double inv_phi2 = 1.0 / phi_sq;
    double cap = 10.0 * phi_sq;
    /* Precompute ln_norm (constant across the loop) once per call */
    double ld = 0.0;
    for (int d = 0; d < p; ++d) {
        double hk = h_fit[d] < 1e-3 ? 1e-3 : h_fit[d];
        ld += log(hk * hk);
    }
    double ln_norm = -0.5 * ((double)p * log(2.0 * M_PI) + ld);

    /* Pre-collect indices of control patients (sZ != 0) once.        */
    int *idx0 = (int *)malloc(sizeof(int) * (size_t)n);
    int n0 = 0;
    for (int i = 0; i < n; ++i) if (Z[i] == 0.0) idx0[n0++] = i;

    for (int k = 0; k < ne; ++k) {
        const double *xk = Xe + (size_t)k * p;
        double rv = 0.0;
        for (int t = 0; t < n0; ++t) {
            int i = idx0[t];
            double s = squared_scaled_dist(xk, X + (size_t)i*p, h_fit, p);
            double kij = exp(-0.5 * s + ln_norm);
            double tau_i = use_tau ? tau[i] : 0.0;
            rv += kij * (inv_phi2 + tau_i);
        }
        double v = (rv > EPS) ? (1.0 / rv) : cap;
        if (v > cap) v = cap;
        out_var[k] = v;
    }
    free(idx0);
}

/* Coordinate-ascent fit. Returns 0 on success, -1 on failure (n_ctrl<2). */
EXPORT int cahb_fit(
    const double *X, const double *Y, const double *Z, int n, int p,
    const double *Xh, const double *Yh, int nh,
    const double *h_fit, const double *h_hist,
    double gamma, double lam,
    int max_iter,
    double *out_mu0,    /* (n,) */
    double *out_tau,    /* (n,) */
    double *out_phi,    /* scalar */
    double *out_theta0  /* (n,) */)
{
    int n0 = 0;
    for (int i = 0; i < n; ++i) if (Z[i] == 0.0) ++n0;
    if (n0 < 2) return -1;

    double invg2 = 1.0 / dmax(gamma * gamma, EPS);

    /* KM[i,j] = density kernel between X[i] and X[j] */
    double *KM = (double *)malloc(sizeof(double)*(size_t)n*(size_t)n);
    if (!KM) return -1;
    kernel_gauss_density_matrix(X, n, X, n, p, h_fit, KM);

    /* θ₀ from historical data at each current X */
    cahb_hist_mean(Xh, Yh, nh, X, n, p, h_hist, out_theta0);

    /* Step 1 of lam.sel.fn: fit mu0 with τ=0, θ₀=0 (no borrowing)         */
    double *zero_vec = (double *)calloc((size_t)n, sizeof(double));
    double *mu0_self = (double *)malloc(sizeof(double)*(size_t)n);
    cahb_update_mu0(KM, Y, Z, n, zero_vec, 1.0, zero_vec, mu0_self);

    /* Step 2: τ_null_raw using mu0_self as both μ₀ and θ₀                  */
    double *tau_null = (double *)malloc(sizeof(double)*(size_t)n);
    cahb_tau_raw(KM, Z, mu0_self, mu0_self, n, invg2, tau_null);

    /* λ_trunc = numpy 10th percentile (linear interp) of finite tau_null */
    double *finite_buf = (double *)malloc(sizeof(double)*(size_t)n);
    int nf = 0;
    for (int i = 0; i < n; ++i) if (isfinite(tau_null[i])) finite_buf[nf++] = tau_null[i];
    double lt = 0.0;
    if (nf > 0) {
        /* ascending insertion sort (n ≤ a few hundred in our setup) */
        for (int i = 1; i < nf; ++i) {
            double v = finite_buf[i]; int j = i;
            while (j > 0 && finite_buf[j-1] > v) {
                finite_buf[j] = finite_buf[j-1]; --j;
            }
            finite_buf[j] = v;
        }
        double q = 0.10;
        double idx = q * (double)(nf - 1);
        int lo = (int)floor(idx), hi = (int)ceil(idx);
        double frac = idx - (double)lo;
        lt = finite_buf[lo] * (1.0 - frac) + finite_buf[hi] * frac;
    }
    free(finite_buf);
    free(zero_vec); free(mu0_self); free(tau_null);

    /* Main coordinate ascent */
    double *tau = (double *)calloc((size_t)n, sizeof(double));
    double *mu0 = (double *)malloc(sizeof(double)*(size_t)n);
    /* Initial phi = max(std(Y[Z==0], ddof=1), 1.0)                         */
    double mean0 = 0.0; int cnt0 = 0;
    for (int i = 0; i < n; ++i) if (Z[i] == 0.0) { mean0 += Y[i]; ++cnt0; }
    if (cnt0 > 0) mean0 /= cnt0;
    double s2 = 0.0;
    for (int i = 0; i < n; ++i) if (Z[i] == 0.0) {
        double d = Y[i] - mean0; s2 += d*d;
    }
    double phi = 1.0;
    if (cnt0 > 1) {
        s2 /= (double)(cnt0 - 1);
        double s = sqrt(s2);
        if (s > 1.0) phi = s;
    }

    double *mu0_prev = (double *)malloc(sizeof(double)*(size_t)n);
    double *mu0_chk  = (double *)malloc(sizeof(double)*(size_t)n);

    for (int it = 0; it < max_iter; ++it) {
        /* Snapshot tau for the convergence check (Python uses tau before update) */
        cahb_update_mu0(KM, Y, Z, n, tau, phi, out_theta0, mu0);
        double phi_n = cahb_update_phi(Y, Z, mu0, n);
        double *tau_n = (double *)malloc(sizeof(double)*(size_t)n);
        cahb_tau_full(KM, Z, mu0, out_theta0, n, invg2, lt, lam, tau_n);
        /* Python's convergence check: ((mu0 - _mu0(KM,Y,Z,tau,phi,th0))**2).mean() < 1e-5
         * That re-runs the update with OLD tau/phi which yields the same mu0.
         * The condition is therefore trivially true at every iteration → loop
         * effectively does ONE iteration after the first.  We replicate the
         * exact behaviour: compute mu0 again with the OLD tau/phi, compare. */
        cahb_update_mu0(KM, Y, Z, n, tau, phi, out_theta0, mu0_chk);
        double mse = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = mu0[i] - mu0_chk[i];
            mse += d * d;
        }
        mse /= (double)n;
        memcpy(tau, tau_n, sizeof(double)*(size_t)n);
        phi = phi_n;
        free(tau_n);
        if (mse < 1e-5) break;
    }
    memcpy(out_mu0, mu0, sizeof(double)*(size_t)n);
    memcpy(out_tau, tau, sizeof(double)*(size_t)n);
    *out_phi = phi;

    free(KM); free(tau); free(mu0); free(mu0_prev); free(mu0_chk);
    return 0;
}

/* Compute R_n^CAHB(x) = Var_ref / Var_borrow.
 * use_tau=0 → no borrowing variance (reference)
 * use_tau=1 → with τ borrowing                                       */
EXPORT double cahb_compute_Rn(
    const double *X, const double *Z, int n, int p,
    const double *tau, double phi,
    const double *h_fit, const double *x_eval)
{
    double phi_sq = phi * phi;
    double vref, vborrow;
    cahb_var_mu0(x_eval, 1, X, n, p, h_fit, Z, tau, phi_sq, 0, &vref);
    cahb_var_mu0(x_eval, 1, X, n, p, h_fit, Z, tau, phi_sq, 1, &vborrow);
    if (!isfinite(vref) || !isfinite(vborrow) || vborrow <= EPS) return 1.0;
    double r = vref / dmax(vborrow, EPS);
    if (r < 1.0) r = 1.0;
    if (r > 1e4) r = 1e4;
    return r;
}

/* Batch: compute R_n at a vector of evaluation points (n_eval).        */
EXPORT void cahb_Rn_batch(
    const double *Xeval, int n_eval,
    const double *X, const double *Z, int n, int p,
    const double *tau, double phi,
    const double *h_fit,
    double *out_Rn)
{
    double phi_sq = phi * phi;
    double *vref = (double *)malloc(sizeof(double)*(size_t)n_eval);
    double *vbor = (double *)malloc(sizeof(double)*(size_t)n_eval);
    cahb_var_mu0(Xeval, n_eval, X, n, p, h_fit, Z, tau, phi_sq, 0, vref);
    cahb_var_mu0(Xeval, n_eval, X, n, p, h_fit, Z, tau, phi_sq, 1, vbor);
    for (int k = 0; k < n_eval; ++k) {
        double r;
        if (!isfinite(vref[k]) || !isfinite(vbor[k]) || vbor[k] <= EPS) r = 1.0;
        else {
            r = vref[k] / dmax(vbor[k], EPS);
            if (r < 1.0) r = 1.0;
            if (r > 1e4) r = 1e4;
        }
        out_Rn[k] = r;
    }
    free(vref); free(vbor);
}

/* CAHB allocation probability for a NEW patient (full pipeline).
 * Refits the model internally using current trial data — matches
 * CAHB.get_allocation_prob().                                          */
EXPORT double cahb_allocation_prob(
    const double *X_curr, const double *Y_curr, const double *Z_curr, int n_curr,
    const double *Xh, const double *Yh, int nh,
    int p, const double *h_fit, const double *h_hist, const double *h_alloc,
    double gamma, double lam, int max_iter,
    const double *x_new)
{
    if (n_curr < 2) return 0.5;

    double *mu0 = (double *)malloc(sizeof(double)*(size_t)n_curr);
    double *tau = (double *)malloc(sizeof(double)*(size_t)n_curr);
    double *theta0 = (double *)malloc(sizeof(double)*(size_t)n_curr);
    double phi = 1.0;
    int rc = cahb_fit(X_curr, Y_curr, Z_curr, n_curr, p,
                      Xh, Yh, nh, h_fit, h_hist, gamma, lam, max_iter,
                      mu0, tau, &phi, theta0);
    if (rc != 0) { free(mu0); free(tau); free(theta0); return 0.5; }

    /* allocation kernel weights */
    double *wa = (double *)malloc(sizeof(double)*(size_t)n_curr);
    kernel_epan_vec(X_curr, n_curr, p, x_new, h_alloc, wa);
    double N0 = 0.0, N1 = 0.0;
    for (int i = 0; i < n_curr; ++i) {
        if (Z_curr[i] == 0.0) N0 += wa[i];
        else                  N1 += wa[i];
    }
    free(wa);

    double Rn = cahb_compute_Rn(X_curr, Z_curr, n_curr, p, tau, phi, h_fit, x_new);
    free(mu0); free(tau); free(theta0);

    double n0e = Rn * N0;
    if (n0e < EPS) n0e = EPS;
    if (N1  < EPS) N1  = EPS;
    double pi = (n0e * n0e) / (n0e * n0e + N1 * N1);
    if (pi < 0.0) pi = 0.0;
    if (pi > 1.0) pi = 1.0;
    return pi;
}
