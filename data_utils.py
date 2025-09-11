import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


"""
This module provides utilities for preparing datasets for machine learning training. The DataUtils class offers the split_and_scale static method which performs train/validation/test splitting with stratification, handles edge cases like single-class datasets or insufficient samples, applies StandardScaler normalization to features, and returns the scaler for later inference. The implementation carefully validates split ratios, checks class distributions for stratification feasibility, and provides informative warnings when stratification cannot be applied. It assumes input data is properly formatted as numpy arrays with matching lengths for features and labels.



"""




class DataUtils:
	@staticmethod
	def split_and_scale(X: np.ndarray, y: np.ndarray,
					   test_size: float = 0.2,
					   val_size: float = 0.2,
					   seed: int = 42) -> Tuple[np.ndarray, np.ndarray, 
											   np.ndarray, np.ndarray, 
											   np.ndarray, np.ndarray, 
											   StandardScaler]:
		if len(X) != len(y):
			print(f"[error] X and y have different lengths: {len(X)} vs {len(y)}")
			return None, None, None, None, None, None, None
		
		if test_size + val_size >= 1.0:
			print("[error] test_size + val_size must be < 1.0")
			return None, None, None, None, None, None, None
		
		unique_labels = np.unique(y)
		if unique_labels.size < 2:
			print("[warning] Only one class found in dataset. Using non-stratified split.")
			stratify_flag = None
		else:
			class_counts = []
			for label in unique_labels:
				class_counts.append(np.sum(y == label))
			if min(class_counts) < 2:
				print("[warning] Some classes have fewer than two samples. Using non-stratified split.")
				stratify_flag = None
			else:
				stratify_flag = y
		
		X_temp, X_test, y_temp, y_test = train_test_split(
			X, y,
			test_size=test_size,
			random_state=seed,
			stratify=stratify_flag
		)
		
		adjusted_val_size = val_size / (1 - test_size)
		unique_temp_labels = np.unique(y_temp)
		if unique_temp_labels.size < 2:
			print("[warning] Only one class in temp set. Using non-stratified split.")
			stratify_val = None
		else:
			temp_class_counts = []
			for label in unique_temp_labels:
				temp_class_counts.append(np.sum(y_temp == label))
			if min(temp_class_counts) < 2:
				print("[warning] Some classes in temp set have fewer than two samples. Using non-stratified split.")
				stratify_val = None
			else:
				stratify_val = y_temp
		
		X_train, X_val, y_train, y_val = train_test_split(
			X_temp, y_temp,
			test_size=adjusted_val_size,
			random_state=seed,
			stratify=stratify_val
		)
		
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_val = scaler.transform(X_val)
		X_test = scaler.transform(X_test)
		
		return X_train, X_val, X_test, y_train, y_val, y_test, scaler
