import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict

"""
This utility module provides tools for feature scaling in ML pipelines. The ScalerUtils class offers rebuild_scaler to reconstruct StandardScaler objects from saved metadata, enabling proper feature normalization during inference. The implementation handles missing or incomplete scaler information gracefully and maintains compatibility with scikit-learn's preprocessing pipeline. It assumes scaler metadata includes mean and scale arrays with matching dimensions.

"""

class ScalerUtils:

	@staticmethod
	def rebuild_scaler(metadata: Dict):
		if metadata.get('scaler_mean') is None or metadata.get('scaler_scale') is None:
			return None
		scaler = StandardScaler()
		scaler.mean_ = np.array(metadata['scaler_mean'])
		scaler.scale_ = np.array(metadata['scaler_scale'])
		scaler.var_ = scaler.scale_ ** 2
		scaler.n_features_in_ = len(scaler.mean_)
		scaler.n_samples_seen_ = 1
		return scaler








