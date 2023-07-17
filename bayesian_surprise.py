import numpy as np


def kl_divergence_zero(cov_t, cov_tp1):
  k = cov_t.shape[-1]
  inv_cov_tp1 = np.linalg.inv(cov_tp1)

  out = np.trace((inv_cov_tp1 @ cov_t)) - k + np.log((np.linalg.det(cov_tp1) / np.linalg.det(cov_t)))

  return (out / 2)


def kl_divergence(cov_t, cov_tp1, mu_t, mu_tp1):
  k = cov_t.shape[-1]
  inv_cov_tp1 = np.linalg.inv(cov_tp1) 

  out = np.trace((inv_cov_tp1 @ cov_t)) - k + (mu_t - mu_tp1).T @ inv_cov_tp1 @ (mu_t - mu_tp1) + \
        np.log((np.linalg.det(cov_tp1) / np.linalg.det(cov_t)))

  return (out / 2)