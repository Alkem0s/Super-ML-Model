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
import os
warnings.filterwarnings('ignore')

class EnhancedRegressionPipeline:
    def __init__(self, random_state=42, n_splits=5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.models = {}
        self.preprocessors = {}
        self.feature_importances = {}
        self.base_rf_models = {}
        self.base_model_results = {}
        
    def create_preprocessor(self, X):
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
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
            remainder='passthrough'
        )
        
        return preprocessor
    
    def custom_cv_score(self, model, X, y):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        cv_scores = {
            'r2': [],
            'mse': [],
            'rmse': []
        }
        
        y_np = y.to_numpy() if isinstance(y, pd.Series) else y
        
        for train_idx, val_idx in cv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y_np[train_idx], y_np[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            
            cv_scores['r2'].append(r2_score(y_val_cv, y_pred))
            mse = mean_squared_error(y_val_cv, y_pred)
            cv_scores['mse'].append(mse)
            cv_scores['rmse'].append(np.sqrt(mse))
        
        return {
            'r2_mean': np.mean(cv_scores['r2']),
            'r2_std': np.std(cv_scores['r2']),
            'mse_mean': np.mean(cv_scores['mse']),
            'mse_std': np.std(cv_scores['mse']),
            'rmse_mean': np.mean(cv_scores['rmse']),
            'rmse_std': np.std(cv_scores['rmse'])
        }
    
    def optimize_model(self, trial, X, y, target_name):
        model_type = trial.suggest_categorical('model_type', ['rf', 'gb'])
        
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = RandomForestRegressor(**params, random_state=self.random_state)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = GradientBoostingRegressor(**params, random_state=self.random_state)
        
        cv_results = self.custom_cv_score(model, X, y)
        return cv_results['r2_mean']
    
    def build_stacking_model(self, X, y, target_name):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.optimize_model(trial, X, y, target_name), n_trials=20)
        
        self.base_rf_models[target_name] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=self.random_state
        )
        self.base_rf_models[target_name].fit(X, y)
        
        best_params = study.best_params
        if best_params['model_type'] == 'rf':
            optimized_model = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                max_features=best_params['max_features'],
                random_state=self.random_state
            )
        else:
            optimized_model = GradientBoostingRegressor(
                n_estimators=best_params['n_estimators'],
                learning_rate=best_params['learning_rate'],
                max_depth=best_params['max_depth'],
                subsample=best_params['subsample'],
                max_features=best_params['max_features'],
                random_state=self.random_state
            )
        
        estimators = [
            ('optimized', optimized_model),
            ('rf', RandomForestRegressor(random_state=self.random_state)),
            ('gb', GradientBoostingRegressor(random_state=self.random_state)),
            ('svr', SVR(kernel='rbf')),
        ]
        
        return StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(random_state=self.random_state),
            cv=self.n_splits,
            passthrough=True,
            n_jobs=-1
        ), estimators
    
    def fit(self, X, y_dict):
        for target_name, y in y_dict.items():
            print(f"\nTraining model for {target_name}...")
            
            self.preprocessors[target_name] = self.create_preprocessor(X)
            X_processed = self.preprocessors[target_name].fit_transform(X)
            
            model, base_estimators = self.build_stacking_model(X_processed, y, target_name)
            model.fit(X_processed, y)
            self.models[target_name] = model
            
            self.feature_importances[target_name] = pd.Series(
                self.base_rf_models[target_name].feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            self.base_model_results[target_name] = {}
            for name, base_model in base_estimators:
                cv_results = self.custom_cv_score(base_model, X_processed, y)
                self.base_model_results[target_name][name] = {
                    'MSE': cv_results['mse_mean'],
                    'MSE_std': cv_results['mse_std'],
                    'R2': cv_results['r2_mean'],
                    'R2_std': cv_results['r2_std']
                }
    
    def predict(self, X):
        predictions = {}
        
        for target_name, model in self.models.items():
            X_processed = self.preprocessors[target_name].transform(X)
            predictions[target_name] = model.predict(X_processed)
            
        return predictions
    
    def evaluate(self, X, y_dict):
        predictions = self.predict(X)
        results = {}
        
        for target_name in y_dict.keys():
            mse = mean_squared_error(y_dict[target_name], predictions[target_name])
            r2 = r2_score(y_dict[target_name], predictions[target_name])
            
            results[target_name] = {
                'Performance': {
                    'MSE': mse,
                    'RMSE': np.sqrt(mse),
                    'R2': r2,
                    'CV_Results': self.custom_cv_score(
                        self.models[target_name],
                        self.preprocessors[target_name].transform(X),
                        y_dict[target_name]
                    )
                }
            }
            
            if target_name in self.feature_importances:
                top_features = self.feature_importances[target_name].head(10)
                results[target_name]['TopFeatures'] = [
                    {
                        'Feature': feature,
                        'Importance': importance
                    }
                    for feature, importance in top_features.items()
                ]
            
            if target_name in self.base_model_results:
                results[target_name]['BaseModels'] = self.base_model_results[target_name]
            
            sample_predictions = pd.DataFrame({
                'Actual': y_dict[target_name],
                'Predicted': predictions[target_name]
            }).head(100)
            results[target_name]['SamplePredictions'] = sample_predictions.to_dict(orient='records')
        
        return results

if __name__ == "__main__":
    data = pd.read_csv('super_model_synthetic_dataset.csv')
    
    target_columns = ['linear_performance', 'nonlinear_performance', 'tree_performance']
    feature_columns = [col for col in data.columns if col not in target_columns]
    
    X = data[feature_columns]
    y_dict = {target: data[target] for target in target_columns}
    
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)
    y_train_dict = {target: data[target].loc[X_train.index] for target in target_columns}
    y_test_dict = {target: data[target].loc[X_test.index] for target in target_columns}
    
    pipeline = EnhancedRegressionPipeline()
    pipeline.fit(X_train, y_train_dict)
    results = pipeline.evaluate(X_test, y_test_dict)
    
    with open("enhanced_model_results_synthetic.json", 'w') as f:
        json.dump(results, f, indent=4)