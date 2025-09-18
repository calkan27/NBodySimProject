import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .stability_dataset import StabilityDataset
from .data_utils import DataUtils
from .utils import set_global_seed

"""
This module implements gradient boosting model training for stability prediction. The main function loads datasets using StabilityDataset utilities, performs hyperparameter tuning via grid search with cross-validation, trains LightGBM models with optimal parameters, evaluates performance metrics (accuracy, precision, recall, F1, AUROC), and saves trained models and preprocessors. The implementation leverages LightGBM's efficiency for tabular data while maintaining scikit-learn compatibility. It assumes properly formatted input data and sufficient samples for meaningful cross-validation.

"""


def main():
	set_global_seed(42)

	csv_path = "stability_data.csv"
	X, y, feature_names = StabilityDataset.load(csv_path)

	if len(X) == 0:
		print("[error] No data loaded")
		return

	X_train, X_val, X_test, y_train, y_val, y_test, scaler = DataUtils.split_and_scale(
		X, y, test_size=0.15, val_size=0.15, seed=42
	)

	if X_train is None:
		print("[error] Data splitting failed")
		return

	print(f"Data shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

	param_grid = {
		'num_leaves': [31, 50, 70, 100],
		'learning_rate': [0.01, 0.05, 0.1, 0.2]
	}

	base_params = {
		'objective': 'binary',
		'metric': 'binary_logloss',
		'boosting_type': 'gbdt',
		'verbose': -1,
		'random_state': 42,
		'seed': 42
	}

	lgb_model = lgb.LGBMClassifier(**base_params)

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	grid_search = GridSearchCV(
		lgb_model,
		param_grid,
		cv=cv,
		scoring='roc_auc',
		n_jobs=-1,
		verbose=1
	)

	print("Starting hyperparameter tuning...")
	grid_search.fit(X_train, y_train)

	print(f"Best parameters: {grid_search.best_params_}")
	print(f"Best CV score: {grid_search.best_score_:.4f}")

	best_model = grid_search.best_estimator_

	y_pred = best_model.predict(X_test)
	y_proba = best_model.predict_proba(X_test)[:, 1]

	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	auroc = roc_auc_score(y_test, y_proba)

	print("\nTest Set Performance:")
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1 Score: {f1:.4f}")
	print(f"AUROC: {auroc:.4f}")

	best_model.booster_.save_model('model.txt')
	print("Model saved to model.txt")

	with open('scaler.pkl', 'wb') as f:
		pickle.dump(scaler, f)
	print("Scaler saved to scaler.pkl")


if __name__ == "__main__":
	main()




