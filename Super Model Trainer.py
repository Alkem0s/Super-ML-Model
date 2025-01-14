import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import optuna
import warnings
import joblib
from typing import Dict, List, Tuple, Any
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class RegressionPipeline:
    def __init__(self, random_state: int = 42, n_splits: int = 5, n_trials: int = 50):
        self.random_state = random_state
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.models: Dict = {}
        self.preprocessors: Dict = {}
        self.feature_importances: Dict = {}
        self.cv_results: Dict = {}
        
    def _create_preprocessor(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('power', PowerTransformer(method='yeo-johnson', standardize=True))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop',
            verbose_feature_names_out=True
        )
        
        transformed_features = (
            [f'num__{feat}' for feat in numeric_features] +
            [f'cat__{feat}' for feat in categorical_features]
        )
        
        return preprocessor, transformed_features
    
    def _get_cv_scores(self, model: Any, X: np.ndarray, y: np.ndarray):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        scorers = {
            'r2': make_scorer(r2_score),
            'neg_mse': make_scorer(mean_squared_error, greater_is_better=False)
        }
        
        cv_results = {}
        for scorer_name, scorer in scorers.items():
            scores = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1)
            
            if scorer_name == 'neg_mse':
                scores = -scores
                cv_results['mse'] = {'mean': scores.mean(), 'std': scores.std()}
                cv_results['rmse'] = {'mean': np.sqrt(scores).mean(), 'std': np.sqrt(scores).std()}
            else:
                cv_results[scorer_name] = {'mean': scores.mean(), 'std': scores.std()}
                
        return cv_results
    
    def _optimize_model(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray):
        model_type = trial.suggest_categorical('model_type', ['rf', 'gb'])
        
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            model = RandomForestRegressor(**params, random_state=self.random_state)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            model = GradientBoostingRegressor(**params, random_state=self.random_state)
        
        cv_scores = self._get_cv_scores(model, X, y)
        return cv_scores['r2']['mean']
    
    def _build_stacking_model(self, X: np.ndarray, y: np.ndarray, transformed_features: List[str]):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._optimize_model(trial, X, y), n_trials=self.n_trials)
        
        best_params = study.best_params
        model_type = best_params.pop('model_type')
        
        if model_type == 'rf':
            optimized_model = RandomForestRegressor(**best_params, random_state=self.random_state)
        else:
            optimized_model = GradientBoostingRegressor(**best_params, random_state=self.random_state)
        
        base_estimators = [
            ('optimized', optimized_model),
            ('rf', RandomForestRegressor(n_estimators=200, random_state=self.random_state)),
            ('gb', GradientBoostingRegressor(n_estimators=200, random_state=self.random_state)),
            ('svr', SVR(kernel='rbf', C=1.0))
        ]
        
        feature_importance_models = {}
        for name, model in base_estimators:
            if hasattr(model, 'feature_importances_'):
                feature_importance_models[name] = {
                    'model': model,
                    'features': transformed_features
                }
        
        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            cv=self.n_splits,
            passthrough=False,
            n_jobs=-1
        )
        
        return stacking_model, base_estimators, feature_importance_models
    
    def fit(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]) -> None:
        for target_name, y in y_dict.items():
            logging.info(f"Training model for {target_name}...")
            
            preprocessor, transformed_features = self._create_preprocessor(X)
            X_processed = preprocessor.fit_transform(X)
            self.preprocessors[target_name] = preprocessor
            self.feature_names[target_name] = transformed_features
            
            model, base_estimators, feature_importance_models = self._build_stacking_model(
                X_processed, y.values, transformed_features
            )
            model.fit(X_processed, y.values)
            self.models[target_name] = model
            
            self.cv_results[target_name] = {}
            for name, base_model in base_estimators:
                self.cv_results[target_name][name] = self._get_cv_scores(base_model, X_processed, y.values)
            
            self.feature_importances[target_name] = {}
            for name, model_info in feature_importance_models.items():
                if hasattr(model.named_estimators_[name], 'feature_importances_'):
                    importances = pd.Series(
                        model.named_estimators_[name].feature_importances_,
                        index=model_info['features']
                    ).sort_values(ascending=False)
                    self.feature_importances[target_name][name] = importances
    
    def evaluate(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]):
        predictions = self.predict(X)
        results = {}
        
        for target_name, y_true in y_dict.items():
            y_pred = predictions[target_name]
            
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            
            results[target_name] = {
                'Performance': {
                    'MSE': float(mean_squared_error(y_true, y_pred)),
                    'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'R2': float(r2_score(y_true, y_pred)),
                    'CV_Results': self.cv_results[target_name]
                }
            }
            
            if target_name in self.feature_importances:
                results[target_name]['FeatureImportances'] = {}
                for model_name, importances in self.feature_importances[target_name].items():
                    top_features = importances.head(10)
                    results[target_name]['FeatureImportances'][model_name] = [
                        {'Feature': feature, 'Importance': float(importance)}
                        for feature, importance in top_features.items()
                    ]
            
            sample_size = min(100, len(y_true))
            sample_indices = np.random.choice(len(y_true), sample_size, replace=False)
            sample_predictions = pd.DataFrame({
                'Actual': y_true[sample_indices],
                'Predicted': y_pred[sample_indices]
            })
            
            results[target_name]['SamplePredictions'] = sample_predictions.to_dict(orient='records')
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        for target_name, model in self.models.items():
            try:
                X_processed = self.preprocessors[target_name].transform(X)
                predictions[target_name] = model.predict(X_processed)
            except Exception as e:
                logging.error(f"Error predicting {target_name}: {str(e)}")
                raise
        return predictions

if __name__ == "__main__":
    data = pd.read_csv('super_model_synthetic_dataset.csv')
    
    target_columns = ['linear_performance', 'nonlinear_performance', 'tree_performance']
    feature_columns = [col for col in data.columns if col not in target_columns]
    
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42, shuffle=True)
    
    X_train = data.loc[train_idx, feature_columns]
    X_test = data.loc[test_idx, feature_columns]
    y_train_dict = {target: data.loc[train_idx, target] for target in target_columns}
    y_test_dict = {target: data.loc[test_idx, target] for target in target_columns}
    
    pipeline = RegressionPipeline(n_trials=50)
    pipeline.fit(X_train, y_train_dict)
    results = pipeline.evaluate(X_test, y_test_dict)
    
    with open("model_results_synthetic.json", 'w') as f:
        json.dump(results, f, indent=4)