"""
This module handles dataset I/O for machine learning pipelines.

The StabilityDataset class provides methods to load feature matrices and labels from CSV
files, extract feature names and scaler metadata from headers, handle missing values and
validate data consistency, and reconstruct preprocessing pipelines from saved
information. The implementation supports the complete ML workflow from data generation
through model deployment. It assumes CSV files follow the expected format with proper
headers and metadata encoding.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict



class StabilityDataset:
    
	@staticmethod
	def load(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
		feature_names = None
		scaler_info = {}
		
		with open(path, 'r') as f:
			first_line = f.readline()
			if first_line.startswith('# feature_names:'):
				feature_names_str = first_line.strip().split(':', 1)[1].strip()
				feature_names = feature_names_str.split(',')
		
		df = pd.read_csv(path, comment='#')
		
		if "is_stable" not in df.columns:
			print("[error] CSV must contain 'is_stable' column")
			return np.array([]), np.array([]), []
		
		exclude_cols = ["simulation_id", "is_stable", "mode", "dataset_version"]
		scaler_cols = []
		for col in df.columns:
			if col.startswith('scaler_'):
				scaler_cols.append(col)
		exclude_cols.extend(scaler_cols)
		
		if scaler_cols:

			mean_cols = []
			scale_cols = []

			for col in scaler_cols:
				if col.startswith('scaler_mean_'):
					mean_cols.append(col)
				elif col.startswith('scaler_scale_'):
					scale_cols.append(col)

			mean_cols  = sorted(mean_cols)
			scale_cols = sorted(scale_cols)



			
			if mean_cols:
				scaler_info['mean'] = df[mean_cols].iloc[0].values
			if scale_cols:
				scaler_info['scale'] = df[scale_cols].iloc[0].values
		
		feature_cols = []
		for col in df.columns:
			if col not in exclude_cols:
				feature_cols.append(col)
		
		if feature_names is None:
			feature_names = feature_cols
		
		X = df[feature_cols].values
		y = df["is_stable"].values
		
		valid_mask = ~np.isnan(y)
		X = X[valid_mask]
		y = y[valid_mask]
		
		print(f"Loaded {len(X)} samples with {X.shape[1]} features")
		
		if np.any(np.isnan(X)):
			print("[warning] NaN values found in features. Replacing with 0.")
			X = np.nan_to_num(X, nan=0.0)
		
		return X, y, feature_names



	@staticmethod
	def get_metadata(path: str) -> Dict:
		metadata = {
			'feature_names': None,
			'scaler_mean': None,
			'scaler_scale': None
		}
		
		with open(path, 'r') as f:
			first_line = f.readline()
			if first_line.startswith('# feature_names:'):
				feature_names_str = first_line.strip().split(':', 1)[1].strip()
				metadata['feature_names'] = feature_names_str.split(',')
		
		df = pd.read_csv(path, comment='#', nrows=1)
		
		mean_cols = []
		for col in df.columns:
			if col.startswith('scaler_mean_'):
				mean_cols.append(col)
		
		scale_cols = []
		for col in df.columns:
			if col.startswith('scaler_scale_'):
				scale_cols.append(col)
		
		if mean_cols:
			metadata['scaler_mean'] = df[mean_cols].iloc[0].values
		if scale_cols:
			metadata['scaler_scale'] = df[scale_cols].iloc[0].values
			
		return metadata






