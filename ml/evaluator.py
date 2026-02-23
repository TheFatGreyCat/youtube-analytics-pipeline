"""
Model Evaluation for Viral Prediction
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from tabulate import tabulate


class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model):
        """
        Args:
            model: Trained model to evaluate
        """
        self.model = model
        self.results = {}
    
    def evaluate(self, X_test, y_test, feature_cols: list):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            feature_cols: List of feature column names
            
        Returns:
            dict: Evaluation results
        """
        print("\n" + "=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
        
        # Print results
        self._print_metrics()
        self._print_confusion_matrix()
        self._print_feature_importance(feature_cols)
        
        return self.results
    
    def _print_metrics(self):
        """Print performance metrics table"""
        metrics_table = [
            ["Accuracy", f"{self.results['accuracy']:.4f}", f"{self.results['accuracy']*100:.2f}%"],
            ["Precision", f"{self.results['precision']:.4f}", f"{self.results['precision']*100:.2f}%"],
            ["Recall", f"{self.results['recall']:.4f}", f"{self.results['recall']*100:.2f}%"],
            ["F1-Score", f"{self.results['f1_score']:.4f}", f"{self.results['f1_score']*100:.2f}%"],
            ["ROC-AUC", f"{self.results['roc_auc']:.4f}", f"{self.results['roc_auc']*100:.2f}%"]
        ]
        
        print("\n📈 Performance Metrics:")
        print(tabulate(metrics_table, headers=["Metric", "Score", "Percentage"], tablefmt="pretty"))
    
    def _print_confusion_matrix(self):
        """Print confusion matrix table"""
        cm = self.results['confusion_matrix']
        cm_df = pd.DataFrame(
            cm, 
            index=['Actual Not Viral', 'Actual Viral'],
            columns=['Predicted Not Viral', 'Predicted Viral']
        )
        
        print("\n📉 Confusion Matrix:")
        print(tabulate(cm_df, headers='keys', tablefmt='pretty', showindex=True))
    
    def _print_feature_importance(self, feature_cols: list):
        """Print feature importance table"""
        feature_imp = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.model.feature_importances_,
            'Importance %': self.model.feature_importances_ * 100
        }).sort_values('Importance', ascending=False)
        
        print("\n🔝 Feature Importance:")
        print(tabulate(
            feature_imp, 
            headers='keys', 
            tablefmt='pretty', 
            showindex=False, 
            floatfmt=('.0f', '.4f', '.2f')
        ))
    
    def get_metrics_summary(self) -> dict:
        """Get summary of key metrics"""
        return {
            'accuracy': self.results['accuracy'],
            'precision': self.results['precision'],
            'recall': self.results['recall'],
            'f1_score': self.results['f1_score'],
            'roc_auc': self.results['roc_auc']
        }
