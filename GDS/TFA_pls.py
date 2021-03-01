import pandas as pd
from sklearn.preprocessing import StandardScaler

from PLS import *
from GDS import Matrix_of_data, TF_targets_in_data


def TFA_estimate(n_comp=None) :
    TF_matrix = Matrix_of_data()
    GDS_data = TF_targets_in_data()
    n = GDS_data.shape[0]  # number of genes
    m = GDS_data.shape[1]  # number of samples
    p = TF_matrix.shape[1]  # number of TFs

    if n_comp is None :  # number of comppnents to be used in PLS
        n_comp = min(n, p)

    if n != TF_matrix.shape[0] :
        return 'Number of genes must be the same for GDS and Tf matrices'

    # scale and center data
    X = StandardScaler().fit(TF_matrix)
    Y = StandardScaler().fit(GDS_data)

    # get diagonals of scaled data
    Dx = np.diag(1 / X.scale_)
    Dy = np.diag(1 / Y.scale_)

    beta = simpls(TF_matrix, GDS_data)  # obtain the regression coefficients

    # inverse of Dy
    dy_inv = np.linalg.inv(Dy)

    TFA = Dx @ beta @ dy_inv

    TFA = pd.DataFrame(TFA, index=TF_matrix.columns, columns=GDS_data.columns)


    row_order = TFA.std(axis=1).to_frame().sort_values(by=[0],ascending=False).index
    TFA_ranked = TFA.loc[TFA.index.isin(row_order)]

    TFA_ranked.to_csv('TFA_ranked.csv', sep='\t')
    return TFA_ranked


print(TFA_estimate(10).shape)