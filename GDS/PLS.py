import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import scale


def svd(crosscov, n_components=None):
    '''Calculates the SVD of `cross covariance` and returns singular vectors/values'''
    crosscov = np.asanyarray(crosscov)

    if n_components is None:
        n_components = min(crosscov.shape)
    elif not isinstance(n_components, int):
        raise TypeError('Provided `n_components` {} must be of type int'
                        .format(n_components))

    # run most computationally efficient SVD
    if crosscov.shape[0] <= crosscov.shape[1]:
        U, d, V = randomized_svd(crosscov.T, n_components=n_components,
                                 transpose=False)
        V = V.T
    else:
        V, d, U = randomized_svd(crosscov, n_components=n_components,
                                transpose=False)
        U = U.T

    return U, np.diag(d), V

def simpls(X, Y, n_components=None):

    X, Y = np.asanyarray(X), np.asanyarray(Y)
    if n_components is None:
        n_components = min(len(X) - 1, X.shape[1])

    # center variables and calculate covariance matrix
    X0 = scale(X)
    Y0 = scale(Y)
    Cov = X0.T @ Y0

    # to store outputs
    y_loadings = np.zeros((Y.shape[1], n_components))
    x_weights = np.zeros((X.shape[1], n_components))

    for comp in range(n_components):
        # get first component of SVD of covariance matrix
        ci, si, ri = svd(Cov, n_components=1)

        ti = X0 @ ri
        normti = np.linalg.norm(ti)

        # rescale such that:
        #     np.diag(x_weights.T @ X0.T @ X0 @ x_weights)
        #     == np.diag(x_scores.T @ x_scores)
        #     == 1
        x_weights[:, [comp]] = ri / normti

        # rescale such that np.diag(x_scores.T @ x_scores) == 1
        ti /= normti

        qi = Y0.T @ ti
        y_loadings[:, [comp]] = qi
    # calculate betas matrix(p,m,len(n_components) that
    # includes regression coefficients
    beta = x_weights @ y_loadings.T
    return beta
