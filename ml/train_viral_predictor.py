"""
Train XGBoost model to predict viral videos
Main training script - orchestrates the entire training pipeline
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_processor import DataProcessor
from ml.model_trainer import ModelTrainer
from ml.evaluator import ModelEvaluator


def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("🚀 YOUTUBE VIRAL PREDICTION MODEL - XGBoost")
    print("=" * 70)
    print("📌 Predicting if a video will go viral based on engagement metrics")
    print("=" * 70)
    
    # Step 1: Process data
    processor = DataProcessor()
    df = processor.fetch_data()
    X, y = processor.prepare_features(df)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # Step 2: Train model
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train)
    
    # Step 3: Evaluate model
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(X_test, y_test, processor.feature_cols)
    
    # Step 4: Save model
    trainer.save_model(
        feature_cols=processor.feature_cols,
        viral_threshold=processor.viral_threshold,
        viral_percentile=processor.viral_percentile
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("💾 Model saved and ready for predictions")
    print("=" * 70)


if __name__ == "__main__":
    main()
