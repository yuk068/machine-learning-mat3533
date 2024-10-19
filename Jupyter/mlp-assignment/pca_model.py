import numpy as np

def eigen_decomposition(X):
    C_x = covariance_matrix(X)
    Lambda, V = np.linalg.eig(C_x)
    return V, Lambda


def covariance_matrix(X, population=True):
    if population:
        return np.cov(X, rowvar=False)
    else:
        return np.cov(X, rowvar=False, bias=False)


def singular_value_decomposition(X, num_components=2):
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    return U, Sigma, Vt.T


def truncated_svd(X, num_components=2):
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    U_k = U[:, :num_components]
    Sigma_k = Sigma[:num_components]
    V_kt = Vt[:num_components, :].T
    return U_k, Sigma_k, V_kt


def power_method(X, num_components=2, max_iter=100, tol=1e-8):
    n, d = X.shape
    np.random.seed(42)
    components = []
    for _ in range(num_components):
        eigenvector = np.random.rand(d)
        for _ in range(max_iter):
            previous = eigenvector
            eigenvector = X.T @ (X @ eigenvector)
            eigenvector = eigenvector / np.linalg.norm(eigenvector)
            if np.abs(eigenvector @ previous - 1) < tol:
                break
        components.append(eigenvector)
        X = X - X @ eigenvector.reshape(-1, 1) @ eigenvector.reshape(1, -1)
    return np.array(components).T


def pca_transform(X, method='eigen_decomposition', num_components=2, max_iter=100, tol=1e-8):
    if method == 'eigen':
        V, Lambda = eigen_decomposition(X)
        X_transformed = X @ V[:, :num_components]
    elif method == 'svd':
        U, Sigma, Vt = singular_value_decomposition(X, num_components)
        X_transformed = U[:, :num_components] @ np.diag(Sigma[:num_components])
    elif method == 'pow_m':
        components = power_method(X, num_components, max_iter, tol)
        X_transformed = X @ components
    else:
        raise ValueError("Invalid method specified for PCA transformation.")

    return X_transformed