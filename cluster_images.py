from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

def cluster_videos_kmeans(X, names, n_clusters = 970):
	kmeans			= KMeans(n_clusters=n_clusters, n_jobs = -1)
	
	belonging_index	= kmeans.fit_predict(X)
	clusters		= [set() for _ in range(n_clusters)]
	for cluster_idx, name in zip(belonging_index, names):
		clusters[cluster_idx].add(name)
	
	return clusters
	
def cluster_videos_gmm(X, names, n_clusters = 970):
	cov_type	= "tied"
	n_init		= 3
	
	gmm 		= GMM( n_components = n_clusters, covariance_type = cov_type, n_init = n_init )
	gmm.fit(X)
	belonging_index	= gmm.predict(X)
	clusters		= [set() for _ in range(n_clusters)]
	for cluster_idx, name in zip(belonging_index, names):
		clusters[cluster_idx].add(name)
	return clusters
