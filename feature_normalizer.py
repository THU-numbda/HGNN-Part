"""
Advanced feature normalization methods for hypergraph partitioning
"""
import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import PowerTransformer, RobustScaler
from hyperopt_config import NormalizationMethod

class FeatureNormalizer:
    """Enhanced feature normalization with multiple methods"""
    
    def __init__(self, method=NormalizationMethod.STANDARD, feature_scaling=1.0, eps=1e-8):
        self.method = method
        self.feature_scaling = feature_scaling
        self.eps = eps
        self.scaler_params = {}
    
    def normalize_features(self, features, reference_idx=4):
        """
        Apply various normalization methods to features
        Args:
            features: numpy array of shape (n_nodes, n_features)
            reference_idx: index of reference feature for scaling (default: degree feature)
        """
        if self.method == NormalizationMethod.STANDARD:
            return self._standard_normalization(features, reference_idx)
        elif self.method == NormalizationMethod.MIN_MAX:
            return self._min_max_normalization(features)
        elif self.method == NormalizationMethod.ROBUST:
            return self._robust_normalization(features)
        elif self.method == NormalizationMethod.UNIT_NORM:
            return self._unit_norm_normalization(features)
        elif self.method == NormalizationMethod.Z_SCORE:
            return self._z_score_normalization(features)
        elif self.method == NormalizationMethod.POWER:
            return self._power_normalization(features)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _standard_normalization(self, features, reference_idx):
        """Original method: normalize to reference feature norm"""
        features = features.copy()
        ref_norm = np.linalg.norm(features[:, reference_idx])
        
        for i in range(features.shape[1]):
            if i != reference_idx:
                feature_norm = np.linalg.norm(features[:, i])
                if feature_norm > self.eps:
                    features[:, i] = features[:, i] / feature_norm * ref_norm * self.feature_scaling
                else:
                    features[:, i] = 0.0
        return features
    
    def _min_max_normalization(self, features):
        """Min-max scaling to [0, 1] range"""
        features = features.copy()
        for i in range(features.shape[1]):
            col = features[:, i]
            min_val, max_val = col.min(), col.max()
            if max_val - min_val > self.eps:
                features[:, i] = (col - min_val) / (max_val - min_val) * self.feature_scaling
            else:
                features[:, i] = 0.0
        return features
    
    def _robust_normalization(self, features):
        """Robust scaling using median and IQR"""
        features = features.copy()
        for i in range(features.shape[1]):
            col = features[:, i]
            median = np.median(col)
            q75, q25 = np.percentile(col, [75, 25])
            iqr = q75 - q25
            if iqr > self.eps:
                features[:, i] = (col - median) / iqr * self.feature_scaling
            else:
                features[:, i] = col - median
        return features
    
    def _unit_norm_normalization(self, features):
        """L2 unit normalization for each feature"""
        features = features.copy()
        for i in range(features.shape[1]):
            norm = np.linalg.norm(features[:, i])
            if norm > self.eps:
                features[:, i] = features[:, i] / norm * self.feature_scaling
            else:
                features[:, i] = 0.0
        return features
    
    def _z_score_normalization(self, features):
        """Standard z-score normalization"""
        features = features.copy()
        for i in range(features.shape[1]):
            col = features[:, i]
            mean, std = col.mean(), col.std()
            if std > self.eps:
                features[:, i] = (col - mean) / std * self.feature_scaling
            else:
                features[:, i] = col - mean
        return features
    
    def _power_normalization(self, features):
        """Power transformation + z-score normalization"""
        features = features.copy()
        for i in range(features.shape[1]):
            col = features[:, i].reshape(-1, 1)
            if col.std() > self.eps and col.min() >= 0:
                # Use Yeo-Johnson for non-negative data
                pt = PowerTransformer(method='yeo-johnson', standardize=True)
                try:
                    features[:, i] = pt.fit_transform(col).flatten() * self.feature_scaling
                except:
                    # Fallback to z-score if power transform fails
                    mean, std = col.mean(), col.std()
                    if std > self.eps:
                        features[:, i] = (col.flatten() - mean) / std * self.feature_scaling
            else:
                # Fallback to z-score for problematic features
                mean, std = col.mean(), col.std()
                if std > self.eps:
                    features[:, i] = (col.flatten() - mean) / std * self.feature_scaling
        return features

def create_enhanced_features(hypergraph_vertices, hypergraph_edges, filename, num_nodes, num_nets, 
                           norm_method=NormalizationMethod.STANDARD, spectral_norm=True, 
                           feature_scaling=1.0):
    """
    Enhanced feature creation with configurable normalization
    """
    from utils import create_clique_expansion_graph, compute_topological_features, create_partition_id_feature
    from utils import normalize_hypergraph_incidence_matrix
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds
    
    # Create base features
    adj_matrix, node_degree, pin_count = create_clique_expansion_graph(hypergraph_vertices, hypergraph_edges)
    clique_topo_features = compute_topological_features(adj_matrix, 2, True, False)
    
    # Enhanced spectral features
    row, col, value = [], [], []
    for i, e in enumerate(hypergraph_edges):
        for v in e:
            row.append(v)
            col.append(i)
            value.append(1)
    H = coo_matrix((value, (row, col)), shape=(num_nodes, num_nets), dtype=float)
    
    if spectral_norm:
        H = normalize_hypergraph_incidence_matrix(H)
    
    # Use more stable SVD parameters
    k = min(4, min(num_nodes, num_nets) - 1)  # Adaptive number of components
    U, S, Vt = svds(H, k=k, which='LM', random_state=42, solver='propack', maxiter=20000)
    U = U[:, np.argsort(S)[::-1]][:, :2]  # Take top 2 components
    
    # Sign correction for consistency
    for i in range(U.shape[1]):    
        if U[np.argmax(np.absolute(U[:,i])),i] < 0:
            U[:,i] = -U[:,i]
    
    star_topo_features = U.copy()
    partition_feature = create_partition_id_feature(len(hypergraph_vertices), filename)
    
    # Combine features
    features = np.column_stack([clique_topo_features, star_topo_features, node_degree, pin_count, partition_feature])
    
    # Apply enhanced normalization
    normalizer = FeatureNormalizer(method=norm_method, feature_scaling=feature_scaling)
    features = normalizer.normalize_features(features, reference_idx=4)  # degree feature as reference
    
    # Cleanup
    del adj_matrix, node_degree, pin_count, clique_topo_features, star_topo_features, H, U, S, Vt
    
    return features