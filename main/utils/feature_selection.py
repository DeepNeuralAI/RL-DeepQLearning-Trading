from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def pca(data, variance = 0.9):
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    df_scaled = pd.DataFrame(np_scaled, columns = data.columns, index = data.index)
    df_pca = PCA(variance).fit_transform(df_scaled)
    columns = [f'f_{i}' for i in range(df_pca.shape[1])]
    return pd.DataFrame(df_pca, columns = columns, index = data.index)
