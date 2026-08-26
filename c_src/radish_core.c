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

/* Inverse standard-normal CDF via Acklam's rational approximation
 * (P. Acklam 2003). Accurate to ~1.15e-9; sufficient for CI z-quantiles. */
static double inv_normal_cdf(double q) {
    static const double a_[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                 -2.759285104469687e+02, 1.383577518672690e+02,
                                 -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b_[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                 -1.556989798598866e+02, 6.680131188771972e+01,
                                 -1.328068155288572e+01};
    static const double c_[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                 -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d_[4] = {7.784695709041462e-03,  3.224671290700398e-01,
                                 2.445134137142996e+00,  3.754408661907416e+00};
    const double pl = 0.02425, ph = 1.0 - pl;
    if (q < pl) {
        double r = sqrt(-2.0*log(q));
        return (((((c_[0]*r+c_[1])*r+c_[2])*r+c_[3])*r+c_[4])*r+c_[5]) /
               ((((d_[0]*r+d_[1])*r+d_[2])*r+d_[3])*r+1.0);
    } else if (q <= ph) {
        double r = q - 0.5; double r2 = r*r;
        return (((((a_[0]*r2+a_[1])*r2+a_[2])*r2+a_[3])*r2+a_[4])*r2+a_[5])*r /
               (((((b_[0]*r2+b_[1])*r2+b_[2])*r2+b_[3])*r2+b_[4])*r2+1.0);
    } else {
        double r = sqrt(-2.0*log(1.0-q));
        return -(((((c_[0]*r+c_[1])*r+c_[2])*r+c_[3])*r+c_[4])*r+c_[5]) /
                ((((d_[0]*r+d_[1])*r+d_[2])*r+d_[3])*r+1.0);
    }
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

/* Residual variance for one arm with second-order degrees-of-freedom
 * correction (Hardle 1990, §4.2; Wand & Jones 1995, §3.4.4):
 *
 *     dof = n − 2·tr(S) + tr(S'S)
 *     σ̂² = ‖Y − S Y‖² / dof
 *
 * The plain n − tr(S) formula systematically under-estimates Var(Y|X,Z)
 * because it does not account for the curvature of the smoother — the
 * residual Y − SY contains both stochastic noise AND a smoothing-bias
 * component, and the additional tr(S'S) term pays for the latter. The
 * second-order DoF makes the estimator approximately unbiased and is
 * the standard correction in kernel-regression inference.            */
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
    double tr_S = 0.0;
    double tr_StS = 0.0;
    double sse = 0.0;
    for (int i = 0; i < na; ++i) {
        double *row = S + (size_t)i * na;
        tr_S += row[i];
        /* tr(S'S) = sum_{i,j} S_{ij}^2  (Frobenius norm of S squared) */
        for (int j = 0; j < na; ++j) tr_StS += row[j] * row[j];
        double pred = 0.0;
        for (int j = 0; j < na; ++j) pred += row[j] * Ya[j];
        double r = Ya[i] - pred;
        sse += r * r;
    }
    free(S);
    /* Second-order effective degrees of freedom */
    double dof = (double)na - 2.0 * tr_S + tr_StS;
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

/* Kernel g-formula Stage III estimator.
 *
 * Robins (1986) g-computation in nonparametric kernel-regression form
 * (Snowden et al. 2011, AJE; Vansteelandt & Keiding 2011, AJE):
 *
 *     Δ̂ = (1/n) Σ_i [m̂₁(X_i) − m̂₀(X_i)]
 *
 * with borrowing-augmented control regression
 *     m̂₀(x) = (1−W(x)) m̂₀^C(x) + W(x) m̂₀^H(x).
 *
 * Algebraically equivalent to weighted-Y form ate = a₁'Y₁ − a₀'Y₀ − a_H'Y_H
 * with influence vectors a_z = (1/n_eval) Σ_i [W-modulated] S_z[i, ·].
 *
 * Variance: M-estimator sandwich for a kernel-regression projection
 * (Härdle 1990 §4.2; Wand & Jones 1995 §3.4.4; Ruppert et al. 2003 §3.13):
 *
 *     V̂ = v̂₁ ‖a₁‖² + v̂₀ ‖a₀‖² + v̂_H ‖a_H‖²
 *
 * with σ̂²_z(Y|X,Z=z) estimated via a kernel smoother and the second-order
 * effective DoF correction dof = n − 2·tr(S) + tr(S'S), the standard
 * adjustment for kernel-regression residuals (Wand & Jones 1995 §3.4.4). */
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

    double z = inv_normal_cdf(1.0 - alpha / 2.0);
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

/* ─────────────────────────────────────────────────────────────────
 * Tunable commensurability-map parameters.
 * Production configuration is G_MAP=MAP_EXCESS and G_USE_TAU=0; there is no
 * separate calibration flag. Other maps and the tau_H^2 denominator are
 * retained only for explicit audit variants.
 *
 * The native default uses the centered excess-surprisal rule.  Setting
 * use_tau=1 replaces the reference-predictive denominator SE² by the
 * Stage-I reference variance tau²_H in Pi_H, removing the structural
 * ceiling W < 1/2.  The production MAP_EXCESS rule replaces the earlier
 * raw surprisal -log p by (-log p - 1)_+; MAP_CHISQ is retained only as a
 * tunable audit alternative.  The one-nat offset is the ideal
 * standard-normal PIT reference calibration, not an assertion about the
 * finite-sample mean of an estimated kernel statistic.
 * ───────────────────────────────────────────────────────────────── */
static double G_LAM_LOC = 1.0;   /* local  sharpness (MAP_CHISQ only) */
static double G_LAM_GLB = 1.0;   /* global sharpness (MAP_CHISQ only) */
static int    G_USE_TAU = 0;     /* production Stages II/III use predictive SE^2 */

/* Commensurability map.  Only MAP_LEGACY and MAP_CHISQ carry free constants;
 * the remaining three are constant-free.
 *
 *  0 MAP_LEGACY   D = -log kappa                      (earlier draft)
 *  1 MAP_CHISQ    D = lam (Z^2 - 1)_+ / 2             (tunable, for comparison)
 *  2 MAP_EXCESS   D = (-log kappa - 1)_+              <-- default
 *                 1 = one-nat ideal standard-normal PIT reference offset
 *                 constant, not a tuning parameter.  Equivalently
 *                 M = min(1, e * kappa).
 *  3 MAP_MEDIAN   D = (-log kappa - log 2)_+          M = min(1, 2 kappa)
 *                 same idea centred on the null MEDIAN of the p-value.
 *  4 MAP_EB       Pi_H = 1/(tau2_H + psi2), psi2 = moment estimate of the
 *                 between-source variance.  Constant-free and MSE-optimal,
 *                 reported to show why an exponential map is needed.
 */
#define MAP_LEGACY 0
#define MAP_CHISQ  1
#define MAP_EXCESS 2
#define MAP_MEDIAN 3
#define MAP_EB     4
/* The repaired centered map is the native default.  MAP_LEGACY remains
 * available through radish_set_borrow_params(..., MAP_LEGACY) for historical
 * reproducibility. */
static int    G_MAP     = MAP_EXCESS;
static int    G_ADJ_GLOBAL = 0;  /* 1 = covariate-standardised trial-level check */
static int    G_SEQ_GLOBAL = 0;  /* 1 = recompute the root-node check at each interim */
static double G_PHI_LO  = 0.0;   /* allocation-probability guard rails */
static double G_PHI_HI  = 1.0;
static int    G_ALLOC_LEGACY = 0;/* 1 = opt into the earlier-draft uncentered Stage-II map */
static int    G_ALLOC_MIN_N0 = 0;/* no borrowing at allocation until this many
                                  *     concurrent controls have accrued         */

/* Kish effective sample size for a local kernel-weighted mean.  The weighted
 * mean/variance retain the raw kernel mass; only sampling variance and
 * precision use n_eff=(sum w)^2/sum(w^2). */
static double kernel_effective_n(const double *w, int n)
{
    double sw = 0.0, sw2 = 0.0;
    if (!w || n <= 0) return 1e-6;
    for (int i = 0; i < n; ++i) {
        sw += w[i];
        sw2 += w[i] * w[i];
    }
    if (!isfinite(sw) || !isfinite(sw2) || sw <= EPS || sw2 <= EPS)
        return 1e-6;
    double neff = (sw * sw) / sw2;
    return (isfinite(neff) && neff > 1e-6) ? neff : 1e-6;
}
EXPORT void radish_set_borrow_params(double lam_loc, double lam_glb,
                                     int use_tau, int map)
{
    G_LAM_LOC = lam_loc; G_LAM_GLB = lam_glb;
    G_USE_TAU = use_tau; G_MAP     = map;
}

EXPORT void radish_set_adj_global(int on) { G_ADJ_GLOBAL = on; }

EXPORT void radish_set_alloc_params(int seq_global, double phi_lo, double phi_hi,
                                    int alloc_legacy, int alloc_min_n0)
{
    G_SEQ_GLOBAL = seq_global; G_PHI_LO = phi_lo; G_PHI_HI = phi_hi;
    G_ALLOC_LEGACY = alloc_legacy; G_ALLOC_MIN_N0 = alloc_min_n0;
}

EXPORT void radish_get_borrow_params(double *lam_loc, double *lam_glb,
                                     int *use_tau, int *map)
{
    if (lam_loc) *lam_loc = G_LAM_LOC;
    if (lam_glb) *lam_glb = G_LAM_GLB;
    if (use_tau) *use_tau = G_USE_TAU;
    if (map)     *map     = G_MAP;
}

/* Compute (R_n, W, D_PDC, tau²_H) at a single evaluation point.
 *
 * The PDC surprisal at x is the Fisher (1925) combination of the local
 * (per-x) Marshall–Spiegelhalter (2007) leaf-node conflict and the trial-
 * level (root-node) conflict, both calibrated under H0 to standard normal:
 * The production MAP_EXCESS rule centers every node contribution as
 * D_node = max(-log(p_node) - 1, 0); the raw formulas below describe only
 * the legacy map.
 *
 *   D_PDC(x) = D_local(x) + D_global,  D_local = −log p_local,  D_global = −log p_global
 *
 * The discount enters R_n multiplicatively as exp(D_PDC):
 *   R_n = 1 + (σ²_{0,c}/n_eff) / (SE² · exp(D_PDC))
 *
 * Caller passes D_global; pass 0 to use the leaf-only (allocation-time)
 * surprisal, or the precomputed trial-level surprisal for Stage III.
 */
static void radish_single_eval(
    const double *X0, const double *Y0, int n0,
    const double *Xh, const double *Yh, int nh,
    int p, const double *h,
    const double *x_eval, double nc_floor, double Dg,
    double *out_Rn, double *out_W,
    double *out_Dpdc, double *out_tau2H)
{
    /* Stage-I local historical prior */
    double *wH = (double *)malloc(sizeof(double) * (size_t)nh);
    kernel_gauss_vec(Xh, nh, p, x_eval, h, wH);
    double swH = 0.0, sumH = 0.0;
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
        double neff = kernel_effective_n(wH, nh);
        tau2H = s2 / neff;
        if (tau2H < EPS) tau2H = EPS;
    }
    free(wH);

    /* Stage-II concurrent-control summary */
    double *w0 = (double *)malloc(sizeof(double) * (size_t)(n0 > 0 ? n0 : 1));
    double Nc_raw = 0.0, Nc_eff = 1e-6, ybar = 0.0, s2c = 1.0;
    if (n0 > 0) {
        kernel_gauss_vec(X0, n0, p, x_eval, h, w0);
        for (int i = 0; i < n0; ++i) Nc_raw += w0[i];
        Nc_eff = kernel_effective_n(w0, n0);
        if (Nc_raw > EPS) {
            double sy = 0.0;
            for (int i = 0; i < n0; ++i) sy += w0[i] * Y0[i];
            ybar = sy / Nc_raw;
            double s2 = 0.0;
            for (int i = 0; i < n0; ++i) {
                double d = Y0[i] - ybar;
                s2 += w0[i] * d * d;
            }
            s2 /= Nc_raw;
            s2c = s2 > EPS ? s2 : EPS;
        } else {
            ybar = 0.0; s2c = 1.0;
        }
    }
    free(w0);

    /* PDC discrepancy with canonical Evans–Moshonov (2006) standardization:
     *   SE^2 = tau2H + s2c / Nc_s
     *   Z    = (ȳ_c − θ) / SE
     * The data sampling variance s2c/Nc_s places Z on its ideal
     * standard-normal reference scale under compatibility and avoids the
     * inflated discrepancy caused by omitting this term.              */
    /* Nc_eff, not raw kernel mass, controls local concurrent SE and
     * precision.  The configured floor remains the sparse-neighbourhood
     * fallback used by the allocation and final-stage paths. */
    double Nc_s = Nc_eff > nc_floor ? Nc_eff : nc_floor;
    double se2 = tau2H + s2c / Nc_s;
    if (se2 < EPS) se2 = EPS;
    double tau = sqrt(se2);
    double Z = (ybar - theta) / tau;
    if (Z >  1e10) Z =  1e10;
    if (Z < -1e10) Z = -1e10;
    double pu = phi_cdf(Z);
    double pmin = pu < (1.0 - pu) ? pu : (1.0 - pu);
    double kap = 2.0 * pmin;
    if (kap < EPS) kap = EPS;

    double Dn = 0.0, psi2_loc = 0.0;
    switch (G_MAP) {
    case MAP_CHISQ: {
        double ex = Z * Z - 1.0;
        Dn = ex > 0.0 ? 0.5 * G_LAM_LOC * ex : 0.0;
        break; }
    case MAP_EXCESS: {                       /* one-nat reference calibration */
        double ex = -log(kap) - 1.0;         /* ideal standard-normal PIT scale */
        Dn = ex > 0.0 ? ex : 0.0;
        break; }
    case MAP_MEDIAN: {
        double ex = -log(kap) - M_LN2;       /* log 2 = median of -log kappa     */
        Dn = ex > 0.0 ? ex : 0.0;
        break; }
    case MAP_EB: {                           /* moment estimate of psi^2         */
        double d2 = (ybar - theta) * (ybar - theta);
        psi2_loc = d2 > se2 ? d2 - se2 : 0.0;
        break; }
    default:
        Dn = -log(kap);
    }

    /* Hierarchical PDC: combine leaf-node (local) and root-node (trial-level)
     * surprisal via Fisher (1925) addition on the −log p scale.  Marshall &
     * Spiegelhalter (2007) show each node-level conflict p-value is calibrated
     * Uniform under H0; Presanis et al. (2013) combine them additively in
     * surprisal form to obtain a single coherent conflict statistic.        */
    double Dpdc = Dn + Dg;

    /* Coherent R_n with combined SE² as prior-precision denominator
     * (Bayesian-coherent: posterior precision of μ_0 given (θ, ȳ_c)). */
    double den = G_USE_TAU ? tau2H : se2;
    double PiH;
    if (G_MAP == MAP_EB) {
        /* Variance-additive form: the caller supplies the trial-level excess
         * variance in Dg, which adds to the local excess on the variance
         * scale rather than multiplying on the surprisal scale.            */
        PiH = 1.0 / dmax(den + psi2_loc + Dg, EPS);
        Dpdc = psi2_loc + Dg;
    } else {
        PiH = 1.0 / dmax(den * exp(Dpdc), EPS);
    }
    double PiC = dmax(Nc_s / s2c, EPS);
    double Rn = 1.0 + PiH / PiC;
    if (Rn < 1.0) Rn = 1.0;
    if (Rn > 1e4) Rn = 1e4;
    double W = (Rn - 1.0) / Rn;

    *out_Rn = Rn; *out_W = W; *out_Dpdc = Dpdc; *out_tau2H = tau2H;
}

/* Trial-level (root-node) PDC surprisal D_global = −log p_global, where
 * p_global = 2 min{Φ(Z_g), 1 − Φ(Z_g)} and
 *
 *     Z_g = (ȳ_H − ȳ_C) / sqrt(s²_C/n_C + s²_H/n_H).
 *
 * For the normal-normal hierarchical model this is the Marshall–Spiegelhalter
 * (2007, Test) node-split conflict p-value at the root mean parameter,
 * calibrated to Uniform(0,1) under no-conflict (H0).  Returned in surprisal
 * form so it adds to the leaf-node local surprisal via Fisher (1925)
 * combination — see Presanis et al. (2013, Stat Sci) for the multi-node
 * combination framework in DAGs.                                          */
static double global_pdc_surprisal(
    const double *Y0, int n0,
    const double *YH, int nH)
{
    if (n0 < 2 || nH < 2) return 0.0;
    double m0 = 0.0, mH = 0.0;
    for (int i = 0; i < n0; ++i) m0 += Y0[i]; m0 /= n0;
    for (int j = 0; j < nH; ++j) mH += YH[j]; mH /= nH;
    double s0 = 0.0, sH = 0.0;
    for (int i = 0; i < n0; ++i) { double d = Y0[i] - m0; s0 += d*d; }
    for (int j = 0; j < nH; ++j) { double d = YH[j] - mH; sH += d*d; }
    s0 /= (double)(n0 - 1);
    sH /= (double)(nH - 1);
    double se = sqrt(s0 / (double)n0 + sH / (double)nH);
    if (se < EPS) return 0.0;
    double Zg = (mH - m0) / se;
    if (Zg >  1e10) Zg =  1e10;
    if (Zg < -1e10) Zg = -1e10;
    double pu = phi_cdf(Zg);
    double pmin = pu < (1.0 - pu) ? pu : (1.0 - pu);
    double kg = 2.0 * pmin;
    if (kg < EPS) kg = EPS;

    switch (G_MAP) {
    case MAP_CHISQ: {
        double ex = Zg * Zg - 1.0;
        return ex > 0.0 ? 0.5 * G_LAM_GLB * ex : 0.0; }
    case MAP_EXCESS: {
        double ex = -log(kg) - 1.0;
        return ex > 0.0 ? ex : 0.0; }
    case MAP_MEDIAN: {
        double ex = -log(kg) - M_LN2;
        return ex > 0.0 ? ex : 0.0; }
    case MAP_EB: {
        /* trial-level excess variance: (Ybar_H - Ybar_C)^2 - SE_g^2 */
        double d2 = (mH - m0) * (mH - m0), s2 = se * se;
        return d2 > s2 ? d2 - s2 : 0.0; }
    default:
        return -log(kg);
    }
}

/* ─────────────────────────────────────────────────────────────────
 * Covariate-adjusted trial-level conflict check.
 *
 * The published Z_g compares the RAW means Ybar^C and Ybar^H, standardised by
 * the MARGINAL sample variances.  Each marginal variance carries
 * Var{mu_0(X)} on top of the residual noise.  That covariate variation is
 * common to both cohorts and carries no information about conflict, so it
 * inflates the standard error and strictly reduces the power of the check --
 * in this design Var{mu_0(X)} is about half of s^2_C.
 *
 * The covariate-standardised discrepancy is the same g-formula contrast used
 * for Delta_N, evaluated on the control side only:
 *
 *   delta_hat = (1/N) sum_i { mu0C_hat(X_i) - theta(X_i) } = a0'Y0 - aH'YH,
 *   var_hat   = sigma0^2 ||a0||^2 + sigmaH^2 ||aH||^2,
 *
 * with a0, aH the averaged normalised kernel rows and sigma^2 the kernel
 * residual variances already used by Stage III.  No new constants.
 * ───────────────────────────────────────────────────────────────── */
static double resid_var_kernel(const double *Xa, const double *Ya, int na,
                               int p, const double *h)
{
    if (na < 3) {
        if (na < 2) return EPS;
        double m = 0.0; for (int i = 0; i < na; ++i) m += Ya[i]; m /= na;
        double s = 0.0; for (int i = 0; i < na; ++i) { double d = Ya[i]-m; s += d*d; }
        s /= (double)(na - 1);
        return s > EPS ? s : EPS;
    }
    double *row = (double *)malloc(sizeof(double) * (size_t)na);
    double rss = 0.0, trS = 0.0, trStS = 0.0;
    for (int i = 0; i < na; ++i) {
        kernel_gauss_vec(Xa, na, p, Xa + (size_t)i * p, h, row);
        double sw = 0.0;
        for (int j = 0; j < na; ++j) sw += row[j];
        if (sw < EPS) sw = EPS;
        double fit = 0.0;
        for (int j = 0; j < na; ++j) {
            row[j] /= sw;
            fit += row[j] * Ya[j];
            trStS += row[j] * row[j];
        }
        trS += row[i];
        double r = Ya[i] - fit;
        rss += r * r;
    }
    free(row);
    double dof = (double)na - 2.0 * trS + trStS;
    if (dof < 1.0) dof = 1.0;
    double v = rss / dof;
    return v > EPS ? v : EPS;
}

static double global_conflict_adjusted(
    const double *X, int n,
    const double *X0, const double *Y0, int n0,
    const double *Xh, const double *Yh, int nh,
    int p, const double *h)
{
    if (n0 < 2 || nh < 2) return 0.0;
    double *a0 = (double *)calloc((size_t)n0, sizeof(double));
    double *aH = (double *)calloc((size_t)nh, sizeof(double));
    double *r0 = (double *)malloc(sizeof(double) * (size_t)n0);
    double *rH = (double *)malloc(sizeof(double) * (size_t)nh);
    for (int i = 0; i < n; ++i) {
        const double *xi = X + (size_t)i * p;
        kernel_gauss_vec(X0, n0, p, xi, h, r0);
        kernel_gauss_vec(Xh, nh, p, xi, h, rH);
        double s0 = 0.0, sH = 0.0;
        for (int j = 0; j < n0; ++j) s0 += r0[j];
        for (int j = 0; j < nh; ++j) sH += rH[j];
        if (s0 < EPS) s0 = EPS;
        if (sH < EPS) sH = EPS;
        for (int j = 0; j < n0; ++j) a0[j] += r0[j] / s0;
        for (int j = 0; j < nh; ++j) aH[j] += rH[j] / sH;
    }
    free(r0); free(rH);
    double delta = 0.0, q0 = 0.0, qH = 0.0;
    for (int j = 0; j < n0; ++j) { a0[j] /= n; delta += a0[j]*Y0[j]; q0 += a0[j]*a0[j]; }
    for (int j = 0; j < nh; ++j) { aH[j] /= n; delta -= aH[j]*Yh[j]; qH += aH[j]*aH[j]; }
    free(a0); free(aH);

    double v0 = resid_var_kernel(X0, Y0, n0, p, h);
    double vH = resid_var_kernel(Xh, Yh, nh, p, h);
    double var = v0 * q0 + vH * qH;
    if (var < EPS) return 0.0;
    double Zg = -delta / sqrt(var);       /* sign convention: theta - mu0C */
    if (Zg >  1e10) Zg =  1e10;
    if (Zg < -1e10) Zg = -1e10;
    double pu = phi_cdf(Zg);
    double pmin = pu < (1.0 - pu) ? pu : (1.0 - pu);
    double kg = 2.0 * pmin;
    if (kg < EPS) kg = EPS;

    switch (G_MAP) {
    case MAP_CHISQ: {
        double ex = Zg*Zg - 1.0;
        return ex > 0.0 ? 0.5 * G_LAM_GLB * ex : 0.0; }
    case MAP_EXCESS: {
        double ex = -log(kg) - 1.0;
        return ex > 0.0 ? ex : 0.0; }
    case MAP_MEDIAN: {
        double ex = -log(kg) - M_LN2;
        return ex > 0.0 ? ex : 0.0; }
    case MAP_EB: {
        double d2 = delta*delta;
        return d2 > var ? d2 - var : 0.0; }
    default:
        return -log(kg);
    }
}

/* Batch version: produces R_n[i], W[i], D_pdc[i], tau²_H[i] for each X[i],
 * with the trial-level surprisal D_global folded uniformly into every D_pdc[i]
 * via Fisher addition — replacing the prior version's binary Welch gate with
 * a continuous, literature-grounded conflict statistic.                    */
EXPORT void radish_diagnostics_batch(
    const double *X, int n,
    const double *X0, const double *Y0, int n0,
    const double *Xh, const double *Yh, int nh,
    int p, const double *h, double nc_floor,
    double *out_Rn, double *out_W,
    double *out_Dpdc, double *out_tau2H)
{
    double Dg = G_ADJ_GLOBAL
        ? global_conflict_adjusted(X, n, X0, Y0, n0, Xh, Yh, nh, p, h)
        : global_pdc_surprisal(Y0, n0, Yh, nh);
    for (int i = 0; i < n; ++i) {
        double Rn, W, D, t2;
        radish_single_eval(X0, Y0, n0, Xh, Yh, nh, p, h,
                           X + (size_t)i * p, nc_floor, Dg,
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

    /* Allocation-time trial-level surprisal.  The default is the centered
     * leaf-only map, D_n^A = (-log(kappa_n)-1)_+.  In legacy mode the old raw
     * surprisal is used for reproducibility.  With G_SEQ_GLOBAL the root-node
     * check is recomputed from controls accrued so far using only interim
     * information, and the same commensurability map is applied.             */
    double Rn, W, D, t2;
    /* Stage II retains the predictive SE^2 precision denominator. The
     * discrepancy-map choice is independent: centered by default, with the
     * uncentered legacy map available only for explicit reproduction. */
    double sv_ll = G_LAM_LOC, sv_lg = G_LAM_GLB;
    int    sv_ut = G_USE_TAU, sv_mp = G_MAP;
    G_USE_TAU = 0;
    if (G_ALLOC_LEGACY) { G_LAM_LOC = 1.0; G_LAM_GLB = 1.0; G_MAP = MAP_LEGACY; }

    double Dg_now = 0.0;
    if (G_SEQ_GLOBAL && !G_ALLOC_LEGACY && n0 >= 2)
        Dg_now = global_pdc_surprisal(Y0, n0, Yh, nh);

    if (n0 < G_ALLOC_MIN_N0) {
        Rn = 1.0;                       /* not enough concurrent controls yet */
    } else {
        radish_single_eval(X0, Y0, n0, Xh, Yh, nh, p, h_fit, x_new,
                           nc_floor, Dg_now, &Rn, &W, &D, &t2);
    }

    G_LAM_LOC = sv_ll; G_LAM_GLB = sv_lg; G_USE_TAU = sv_ut; G_MAP = sv_mp;
    free(X0); free(Y0);

    /* Allocation: Gaussian-weighted local arm sizes (single kernel; the
     * bandwidth h_alloc passed in equals the shared Scott-ROT bandwidth). */
    double *wa = (double *)malloc(sizeof(double)*(size_t)n_curr);
    kernel_gauss_vec(X_curr, n_curr, p, x_new, h_alloc, wa);
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
    if (pi < G_PHI_LO) pi = G_PHI_LO;
    if (pi > G_PHI_HI) pi = G_PHI_HI;
    return pi;
}

/* =================================================================
 * KBCD allocation probability
 *   No borrowing; just balances local arm sizes using the single
 *   Gaussian kernel.
 * ================================================================= */
EXPORT double kbcd_allocation_prob(
    const double *X_curr, const int *Z_curr, int n_curr, int p,
    const double *h_alloc, const double *x_new)
{
    if (n_curr < 2) return 0.5;
    double *wa = (double *)malloc(sizeof(double)*(size_t)n_curr);
    kernel_gauss_vec(X_curr, n_curr, p, x_new, h_alloc, wa);
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

    /* allocation kernel weights (single Gaussian kernel) */
    double *wa = (double *)malloc(sizeof(double)*(size_t)n_curr);
    kernel_gauss_vec(X_curr, n_curr, p, x_new, h_alloc, wa);
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
