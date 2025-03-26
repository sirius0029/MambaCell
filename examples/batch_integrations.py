import scvi
import scanpy as sc
import torch
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from model import MambaCellForEmbedding
from sklearn.metrics import adjusted_rand_score, silhouette_score

#load dataset
adata = scvi.data.pbmc_dataset(save_path="data/")
adata.obs['batch'] = adata.obs['batch'].astype('category')

#extract expression metric
expression_data = adata.X

expression_data = expression_data.toarray()
expression_data = (expression_data - np.mean(expression_data, axis=0)) / np.std(expression_data, axis=0)

expression_tensor = torch.tensor(expression_data, dtype=torch.float32)
batch_labels = adata.obs['batch'].cat.codes.values
batch_labels = torch.tensor(batch_labels, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#params
vocab_size = expression_tensor.size(1)
d_model = 128
n_layer = 6

#model definition
mamba_model = MambaCellForEmbedding(vocab_size=vocab_size, d_model=d_model,
                                    n_layer=n_layer, task="joint")
mamba_model = mamba_model.to(device)
expression_tensor=expression_tensor.to(device)
with torch.no_grad():

    outputs = mamba_model(input_ids=expression_tensor.T)
    embeddings = outputs["contrastive"]

embeddings_np = embeddings.cpu().numpy()

n_samples, n_features = embeddings_np.shape
n_components = min(n_samples, n_features, 50)
pca = PCA(n_components=50)
pca_result = pca.fit_transform(embeddings_np)


umap = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
umap_result = umap.fit_transform(pca_result)
adata.obsm['X_umap'] = umap_result
sc.pl.umap(adata, color='batch', title="UMAP (Batch Integration)")

#result
ari_score = adjusted_rand_score(adata.obs['batch'], adata.obs['cluster'])
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")

asw_score = silhouette_score(embeddings_np, adata.obs['batch'].cat.codes)
print(f"Average Silhouette Width (ASW): {asw_score:.4f}")