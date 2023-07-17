import numpy as np
import scipy.linalg


# Bayesian Linear Regression Model
class BayesianModel():
  def __init__(self, d, default_variance=50, default_noise_precision=0.33):
    # d = amount of topics, or dimensions
    self.d = d

    # B = noise precision parameter, a known constant
    self.B = default_noise_precision

    # Setting all preferences to zero, and setting default covariance
    self.p = np.zeros(d)
    self.covariance = np.diag([default_variance] * d).astype(np.float64)
  
  def update(self, x, r):
    assert x.shape[0] == self.d
    x = x.reshape(1, -1)

    # Equations from Bishop, 2006
    orig_inv_covariance = np.linalg.inv(self.covariance)
    new_inv_covariance = orig_inv_covariance + self.B * (x.T @ x)
    new_covariance = np.linalg.inv(new_inv_covariance)

    self.p = (new_covariance @ ((orig_inv_covariance @ self.p) + (self.B * (x.T * r)).T).T).reshape(self.d)
    self.covariance = new_covariance

    return self.p, self.covariance
  
  def predict(self, x):
    return self.p @ x

  def get_params(self):
    return self.p, self.covariance

  def set_params(self, p, c):
    self.p = p
    self.covariance = c


# Variance Bounded Bayesian Linear Regression Model
class VarianceBoundedBayesianModel():
  def __init__(self, d, default_variance=15, default_noise_precision=0.33, tau=1.0):
    # d = amount of topics, or dimensions
    self.d = d

    # tau = the minimum value of the variance for each variable
    self.tau = tau

    # B = noise precision parameter, a known constant
    self.B = default_noise_precision

    # Setting all preferences to zero, and setting default covariance
    self.default_variance = default_variance
    self.p = np.zeros(d)
    self.covariance = np.diag([default_variance] * d).astype(np.float64)
  
  def update(self, x, r):
    assert x.shape[0] == self.d
    x = x.reshape(1, -1)
    
    e, v = scipy.linalg.eigh(self.covariance)                
    e_prime = np.clip(e, a_min=self.tau, a_max=None)
    s = v @ np.diag(e_prime) @ v.T
    s_inv = np.linalg.inv(s)

    new_inv_covariance = s_inv + self.B * (x.T * x)
    new_covariance = np.linalg.inv(new_inv_covariance)

    self.p = (new_covariance @ ((s_inv @ self.p) + (self.B * (x.T * r)).T).T).reshape(self.d)
    self.covariance = new_covariance

    return self.p, s, self.covariance
  
  def predict(self, x):
    return self.p @ x

  def get_params(self):
    return self.p, self.covariance

  def set_params(self, p, c):
    self.p = p
    self.covariance = c


# Adaptive Regularization of Weights for Regression Model
class AROW_Regression():
  def __init__(self, k_topics, lam1, lam2):
    self.k = k_topics
    self.mu = np.zeros([self.k, 1])

    self.cov = np.diag(np.ones(self.k))

    self.lam1 = lam1
    self.lam2 = lam2

  def update(self, x, y):
    x = np.reshape(x, (-1, 1))

    r1 = 1.0 / (2 * self.lam1)
    r2 = 1.0 / (2 * self.lam2)

    beta_mu =  1.0 / (x.T @ self.cov @ x + r1)
    alpha = (y - x.T @ self.mu) * beta_mu
    self.mu = self.mu + (alpha * (self.cov @ x))

    beta_sigma =  1.0 / (x.T @ self.cov @ x + r2)
    self.cov = self.cov - ((beta_sigma * self.cov) @ x @ x.T @ self.cov)

    return self.mu, self.cov

  def get_params(self):
    return self.mu, self.cov

  def predict(self, x):
    return self.mu.T @ x.reshape(-1, 1)
    

# Normalized Least Mean Square Model
class NLMS():
    def __init__(self, k_topics, step_size = 0.1, eps = 0.001):
        # k_topics = amount of topics
        self.k = k_topics
        self.mean = np.zeros(k_topics)
        self.step_size = step_size
        self.eps = eps

    def update(self, x, y):
        e = y - self.predict(x)
        self.mean = self.mean +  (self.step_size * e * x) / (self.eps + x.T @ x)  
        return self.mean

    def predict(self, x):
        return self.mean @ x

    def get_params(self):
        return self.mean




