from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import scale

from sklearn.cluster import AgglomerativeClustering
def cluster_videos_kmeans(X, names, n_clusters = 970):
	X 				= scale(X)
	kmeans			= KMeans(n_clusters=n_clusters, n_jobs = -1)

	belonging_index	= kmeans.fit_predict(X)
	clusters		= [set() for _ in range(n_clusters)]
	for cluster_idx, name in zip(belonging_index, names):
		clusters[cluster_idx].add(name)

	return clusters

def cluster_videos_gmm(X, names, n_clusters = 970, cov_type = "tied"):
	X 			= scale(X)
	n_init		= 3

	gmm 		= GMM( n_components = n_clusters, covariance_type = cov_type, n_init = n_init )
	gmm.fit(X)
	belonging_index	= gmm.predict(X)
	clusters		= [set() for _ in range(n_clusters)]
	for cluster_idx, name in zip(belonging_index, names):
		clusters[cluster_idx].add(name)
	return clusters

def cluster_videos_ac(X, names, n_clusters = 970):
    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='hamming',linkage = 'average')
    belonging_index = model.fit_predict(X)
    print belonging_index
    clusters		= [set() for _ in range(n_clusters)]
    for cluster_idx, name in zip(belonging_index, names):
        clusters[cluster_idx].add(name)

    return clusters
