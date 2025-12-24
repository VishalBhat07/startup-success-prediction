"""
Model utility functions for startup success prediction
Extracted from predict_model.ipynb for Flask integration
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Any


class ModelPredictor:
    """Handles model loading and predictions for startup success"""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize the predictor with model and configuration
        
        Args:
            model_path: Path to the trained model pickle file
            config_path: Path to the model configuration JSON file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"✅ Model loaded: {len(self.config['feature_columns'])} features")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing steps used during training
        
        Args:
            df: Input dataframe with startup data
            
        Returns:
            Preprocessed feature dataframe ready for prediction
        """
        df = df.copy()
        
        # Funding features
        df['funding_amount_usd'] = df['funding_amount_usd'].fillna(0)
        df['log_funding'] = np.log1p(df['funding_amount_usd'])
        
        # Revenue features
        df['estimated_revenue_usd'] = df['estimated_revenue_usd'].fillna(0)
        df['log_revenue'] = np.log1p(df['estimated_revenue_usd'])
        
        # Efficiency metric
        df['revenue_per_employee'] = df['estimated_revenue_usd'] / df['employee_count'].replace(0, 1)
        
        # Investor features
        df['investor_count'] = df['co_investors'].fillna('').apply(
            lambda x: x.count(',') + 1 if x != '' else 0
        )
        
        # Round encoding
        round_map = self.config['round_map']
        df['round_encoded'] = df['funding_round'].map(round_map).fillna(0)
        
        # Categorical encoding
        features_to_encode = self.config['features_to_encode']
        X_encoded = pd.get_dummies(df[features_to_encode], drop_first=True)
        
        # Combine features
        X = pd.concat([
            df[['log_funding', 'log_revenue', 'revenue_per_employee', 
                'employee_count', 'investor_count', 'round_encoded']],
            X_encoded
        ], axis=1)
        
        # Ensure all training features are present
        for col in self.config['feature_columns']:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training
        X = X[self.config['feature_columns']]
        
        return X
    
    def validate_input(self, data: Dict[str, Any]) -> None:
        """
        Validate input data has required fields
        
        Args:
            data: Input dictionary with startup data
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = [
            'company_name', 'founded_year', 'industry', 'region',
            'funding_amount_usd', 'estimated_revenue_usd', 'employee_count',
            'funding_round', 'co_investors'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    def predict_single(self, startup_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict success for a single startup
        
        Args:
            startup_data: Dictionary containing startup information
            
        Returns:
            Dictionary with prediction results
        """
        # Validate input
        self.validate_input(startup_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([startup_data])
        
        # Preprocess
        X = self.preprocess_data(df)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        return {
            'company_name': startup_data['company_name'],
            'predicted_success': bool(prediction),
            'success_probability': float(probability),
            'prediction_label': 'High Value (Top 25%)' if prediction == 1 else 'Standard',
            'confidence_level': self._get_confidence_level(probability),
            'input_data': {
                'industry': startup_data['industry'],
                'region': startup_data['region'],
                'founded_year': startup_data['founded_year'],
                'funding_amount_usd': startup_data['funding_amount_usd'],
                'estimated_revenue_usd': startup_data['estimated_revenue_usd'],
                'employee_count': startup_data['employee_count'],
                'funding_round': startup_data['funding_round']
            }
        }
    
    def predict_batch(self, startups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict success for multiple startups
        
        Args:
            startups: List of dictionaries containing startup information
            
        Returns:
            List of prediction results
        """
        if not startups:
            raise ValueError("Empty startup list provided")
        
        # Validate all inputs
        for i, startup in enumerate(startups):
            try:
                self.validate_input(startup)
            except ValueError as e:
                raise ValueError(f"Startup at index {i}: {str(e)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(startups)
        
        # Preprocess
        X = self.preprocess_data(df)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Format results
        results = []
        for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
            results.append({
                'company_name': startups[i]['company_name'],
                'predicted_success': bool(prediction),
                'success_probability': float(probability),
                'prediction_label': 'High Value (Top 25%)' if prediction == 1 else 'Standard',
                'confidence_level': self._get_confidence_level(probability),
                'input_data': {
                    'industry': startups[i]['industry'],
                    'region': startups[i]['region'],
                    'founded_year': startups[i]['founded_year'],
                    'funding_amount_usd': startups[i]['funding_amount_usd'],
                    'estimated_revenue_usd': startups[i]['estimated_revenue_usd'],
                    'employee_count': startups[i]['employee_count'],
                    'funding_round': startups[i]['funding_round']
                }
            })
        
        # Sort by probability (descending)
        results.sort(key=lambda x: x['success_probability'], reverse=True)
        
        return results
    
    def _get_confidence_level(self, probability: float) -> str:
        """
        Get confidence level based on probability
        
        Args:
            probability: Success probability (0-1)
            
        Returns:
            Confidence level string
        """
        if probability >= 0.75:
            return 'Very High'
        elif probability >= 0.60:
            return 'High'
        elif probability >= 0.40:
            return 'Medium'
        elif probability >= 0.25:
            return 'Low'
        else:
            return 'Very Low'
