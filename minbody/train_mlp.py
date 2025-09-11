import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .stability_dataset import StabilityDataset
from .data_utils import DataUtils
from .model_zoo import make_mlp
from .utils import set_global_seed

"""
This module implements neural network training for stability prediction. The MLPTrainer class manages the complete training pipeline including data loading and preprocessing with StandardScaler, model training with early stopping and validation monitoring, optimal threshold selection using Youden's index, comprehensive evaluation metrics, and model serialization with metadata. The implementation uses PyTorch with GPU support when available, includes dropout regularization to prevent overfitting, and maintains training history for analysis. It assumes CUDA-compatible hardware for optimal performance and sufficient data for train/validation/test splits.


"""

class MLPTrainer:
	def __init__(self, csv_path="stability_data.csv", device=None):
		self.csv_path = csv_path
		if device is None:
			if torch.cuda.is_available():
				self.device = torch.device("cuda")
			else:
				self.device = torch.device("cpu")
		else:
			self.device = device
		self.model = None
		self.scaler = None
		self.optimal_threshold = 0.5
		self.feature_names = None
		
	def load_and_prepare_data(self):
		X, y, feature_names = StabilityDataset.load(self.csv_path)
		self.feature_names = feature_names
		
		if len(X) == 0:
			print("[error] No data loaded")
			return None
			
		X_train, X_val, X_test, y_train, y_val, y_test, scaler = DataUtils.split_and_scale(
			X, y, test_size=0.15, val_size=0.15, seed=42
		)
		
		if X_train is None:
			print("[error] Data splitting failed")
			return None
			
		self.scaler = scaler
		
		print(f"Data shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
		
		train_dataset = TensorDataset(
			torch.FloatTensor(X_train),
			torch.FloatTensor(y_train)
		)
		val_dataset = TensorDataset(
			torch.FloatTensor(X_val),
			torch.FloatTensor(y_val)
		)
		test_dataset = TensorDataset(
			torch.FloatTensor(X_test),
			torch.FloatTensor(y_test)
		)
		
		train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
		test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
		
		return train_loader, val_loader, test_loader, X_train.shape[1]
		
	def train(self, train_loader, val_loader, input_dim, epochs=200, patience=20):
		self.model = make_mlp(input_dim).to(self.device)
		
		criterion = nn.BCEWithLogitsLoss()
		optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		
		best_val_loss = float('inf')
		patience_counter = 0
		best_model_state = None
		
		for epoch in range(epochs):
			self.model.train()
			train_loss = 0.0
			
			for X_batch, y_batch in train_loader:
				X_batch = X_batch.to(self.device)
				y_batch = y_batch.to(self.device).unsqueeze(1)
				
				optimizer.zero_grad()
				outputs = self.model(X_batch)
				loss = criterion(outputs, y_batch)
				loss.backward()
				optimizer.step()
				
				train_loss += loss.item()
				
			train_loss /= len(train_loader)
			
			self.model.eval()
			val_loss = 0.0
			
			with torch.no_grad():
				for X_batch, y_batch in val_loader:
					X_batch = X_batch.to(self.device)
					y_batch = y_batch.to(self.device).unsqueeze(1)
					
					outputs = self.model(X_batch)
					loss = criterion(outputs, y_batch)
					val_loss += loss.item()
					
			val_loss /= len(val_loader)
			
			if epoch % 10 == 0:
				print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
				
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
				best_model_state = self.model.state_dict()
			else:
				patience_counter += 1
				
			if patience_counter >= patience:
				print(f"Early stopping at epoch {epoch}")
				break
				
		self.model.load_state_dict(best_model_state)
		
	def compute_optimal_threshold(self, val_loader):
		self.model.eval()
		
		all_probs = []
		all_labels = []
		
		with torch.no_grad():
			for X_batch, y_batch in val_loader:
				X_batch = X_batch.to(self.device)
				
				outputs = self.model(X_batch)
				probs = torch.sigmoid(outputs).cpu().numpy()
				
				all_probs.extend(probs.squeeze())
				all_labels.extend(y_batch.numpy())
				
		all_probs = np.array(all_probs)
		all_labels = np.array(all_labels)
		
		thresholds = np.linspace(0.1, 0.9, 100)
		best_j_stat = -1
		best_threshold = 0.5
		
		for threshold in thresholds:
			preds = (all_probs > threshold).astype(int)
			tp = np.sum((preds == 1) & (all_labels == 1))
			tn = np.sum((preds == 0) & (all_labels == 0))
			fp = np.sum((preds == 1) & (all_labels == 0))
			fn = np.sum((preds == 0) & (all_labels == 1))
			
			if (tp + fn) > 0:
				tpr = tp / (tp + fn)
			else:
				tpr = 0
			if (tn + fp) > 0:
				tnr = tn / (tn + fp)
			else:
				tnr = 0
			
			j_stat = tpr + tnr - 1
			
			if j_stat > best_j_stat:
				best_j_stat = j_stat
				best_threshold = threshold
				
		self.optimal_threshold = best_threshold
		print(f"Optimal threshold (Youden index): {self.optimal_threshold:.3f}")
		
	def evaluate(self, test_loader):
		self.model.eval()
		
		all_probs = []
		all_labels = []
		
		with torch.no_grad():
			for X_batch, y_batch in test_loader:
				X_batch = X_batch.to(self.device)
				
				outputs = self.model(X_batch)
				probs = torch.sigmoid(outputs).cpu().numpy()
				
				all_probs.extend(probs.squeeze())
				all_labels.extend(y_batch.numpy())
				
		all_probs = np.array(all_probs)
		all_labels = np.array(all_labels)
		
		all_preds = (all_probs > self.optimal_threshold).astype(int)
		
		accuracy = accuracy_score(all_labels, all_preds)
		precision = precision_score(all_labels, all_preds)
		recall = recall_score(all_labels, all_preds)
		f1 = f1_score(all_labels, all_preds)
		auroc = roc_auc_score(all_labels, all_probs)
		
		print("\nTest Set Performance:")
		print(f"Threshold used: {self.optimal_threshold:.3f}")
		print(f"Accuracy: {accuracy:.4f}")
		print(f"Precision: {precision:.4f}")
		print(f"Recall: {recall:.4f}")
		print(f"F1 Score: {f1:.4f}")
		print(f"AUROC: {auroc:.4f}")
		
	def save_model(self):
		torch.save(self.model.state_dict(), 'mlp_model.pth')
		print("Model saved to mlp_model.pth")
		
		with open('scaler.pkl', 'wb') as f:
			pickle.dump(self.scaler, f)
		print("Scaler saved to scaler.pkl")
		
		model_metadata = {
			'feature_names': self.feature_names,
			'optimal_threshold': self.optimal_threshold,
			'input_dim': self.model.fc1.in_features
		}
		with open('model_metadata.json', 'w') as f:
			json.dump(model_metadata, f, indent=2)
		print("Model metadata saved to model_metadata.json")
		
	def run(self):
		data = self.load_and_prepare_data()
		if data is None:
			return
			
		train_loader, val_loader, test_loader, input_dim = data
		
		print("Starting training...")
		self.train(train_loader, val_loader, input_dim)
		
		print("\nComputing optimal threshold on validation set...")
		self.compute_optimal_threshold(val_loader)
		
		print("\nEvaluating on test set...")
		self.evaluate(test_loader)
		
		self.save_model()


def main():
	set_global_seed(42)
	trainer = MLPTrainer()
	trainer.run()


if __name__ == "__main__":
	main()


