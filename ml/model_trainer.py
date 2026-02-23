"""
Model Training for Viral Prediction
"""
import sys
from pathlib import Path
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml import config


class ModelTrainer:
    """Train XGBoost model for viral prediction"""
    
    def __init__(self, xgb_params: dict = None):
        """
        Args:
            xgb_params: XGBoost hyperparameters (default from config)
        """
        self.xgb_params = xgb_params or config.XGBOOST_PARAMS
        self.model = None
    
    def train(self, X_train, y_train):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print("\n" + "=" * 70)
        print("🎯 TRAINING XGBOOST MODEL")
        print("=" * 70)
        
        print("\n⏳ Training in progress...")
        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.model.fit(X_train, y_train)
        print("✅ Training completed!")
        
        return self.model
    
    def save_model(self, feature_cols: list, viral_threshold: float, 
                   viral_percentile: float, filepath: str = None):
        """
        Save trained model to disk
        
        Args:
            feature_cols: List of feature column names
            viral_threshold: Viral threshold value
            viral_percentile: Viral percentile used
            filepath: Path to save model (default from config)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        filepath = filepath or config.MODEL_PATH
        model_path = Path(__file__).parent / filepath
        
        # Save model with metadata
        joblib.dump({
            'model': self.model,
            'feature_cols': feature_cols,
            'viral_threshold': viral_threshold,
            'viral_percentile': viral_percentile,
            'xgb_params': self.xgb_params
        }, model_path)
        
        print(f"\n💾 Model saved: {model_path}")
        return str(model_path)
