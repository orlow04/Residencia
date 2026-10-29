import sys, runpy, io, contextlib, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
SCR, DATA = sys.argv[1:3]
sys.argv = [SCR, "--data", DATA, "--matpre", f"{DATA}/matPre.pickle", "--out", "/dev/null"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(SCR)
# cópia de KuaiEnv.load_mat
df_small = pd.read_csv(f"{DATA}/small_matrix.csv", header=0, usecols=['user_id', 'video_id', 'watch_ratio']).rename(columns={"video_id": "item_id"})
df_small.loc[df_small['watch_ratio'] > 5, 'watch_ratio'] = 5
lbe_item = LabelEncoder(); lbe_item.fit(df_small['item_id'].unique())
lbe_user = LabelEncoder(); lbe_user.fit(df_small['user_id'].unique())
mat = csr_matrix((df_small['watch_ratio'], (lbe_user.transform(df_small['user_id']), lbe_item.transform(df_small['item_id']))),
                 shape=(df_small['user_id'].nunique(), df_small['item_id'].nunique())).toarray()
mat[np.isnan(mat)] = df_small['watch_ratio'].mean()
mat[np.isinf(mat)] = df_small['watch_ratio'].mean()
assert np.allclose(mat, g["ev"]), np.abs(mat - g["ev"]).max()
print("OK: matriz de avaliação idêntica à de KuaiEnv.load_mat (com duplicata, célula ausente, NaN e valor > 5)")
