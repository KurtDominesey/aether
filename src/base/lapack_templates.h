#ifndef AETHER_BASE_LAPACK_TEMPLATES_H_
#define AETHER_BASE_LAPACK_TEMPLATES_H_

#include <deal.II/base/config.h>
#include <deal.II/lac/lapack_support.h>

// this should be included from config.h, but isn't for some reason
// TODO: fix this while integrating these changes into (a fork of) deal.II
#ifndef DEAL_II_FORTRAN_MANGLE
#define DEAL_II_FORTRAN_MANGLE(name, NAME) name ## _
#endif

extern "C" {

void DEAL_II_FORTRAN_MANGLE(dtrsm, 
                            DTRSM)(const char* side,
                                   const char* uplo,
                                   const char* transa,
                                   const char* diag,
                                   const dealii::types::blas_int* m,
                                   const dealii::types::blas_int* n,
                                   const double* alpha,
                                   const double* a,
                                   const dealii::types::blas_int* lda,
                                   double* b,
                                   const dealii::types::blas_int* ldb);

void DEAL_II_FORTRAN_MANGLE(dlaswp,
                            DLASWP)(const dealii::types::blas_int* n,
                                    double* a,
                                    const dealii::types::blas_int* lda,
                                    const dealii::types::blas_int* k1,
                                    const dealii::types::blas_int* k2,
                                    const dealii::types::blas_int* ipiv,
                                    const dealii::types::blas_int* incx);

}

inline void trsm(const char* side,
                 const char* uplo,
                 const char* transa,
                 const char* diag,
                 const dealii::types::blas_int* m,
                 const dealii::types::blas_int* n,
                 const double* alpha,
                 const double* a,
                 const dealii::types::blas_int* lda,
                 double* b,
                 const dealii::types::blas_int* ldb) {
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(dtrsm, DTRSM)
  (side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
#else
  (void)side;
  (void)uplo;
  (void)transa;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  Assert(false, dealii::LAPACKSupport::ExcMissing("dtrsm"));
#endif  // DEAL_II_WITH_LAPACK
}

inline void laswp(const dealii::types::blas_int* n,
                  double* a,
                  const dealii::types::blas_int* lda,
                  const dealii::types::blas_int* k1,
                  const dealii::types::blas_int* k2,
                  const dealii::types::blas_int* ipiv,
                  const dealii::types::blas_int* incx) {
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(dlaswp, DLASWP)
  (n, a, lda, k1, k2, ipiv, incx);
#else
  (void)n;
  (void)a;
  (void)lda;
  (void)k1;
  (void)k2;
  (void)ipiv;
  (void)incx;
  Assert(false, dealii::LAPACKSupport::ExcMissing("dlaswp"));
#endif  // DEAL_II_WITH_LAPACK
}

#endif  // AETHER_BASE_LAPACK_TEMPLATES_H_