import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import warnings
warnings.filterwarnings('ignore')

class AdvancedRegressionPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.feature_importances = {}
        self.base_rf_models = {}
        self.base_model_results = {}
        
    def preprocess_data(self, X):
        X_scaled = pd.DataFrame(index=X.index)
        
        for column in X.columns:
            if column not in self.scalers:
                self.scalers[column] = RobustScaler()
                X_scaled[column] = self.scalers[column].fit_transform(X[[column]])
            else:
                X_scaled[column] = self.scalers[column].transform(X[[column]])
        
        return X_scaled
    
    def optimize_model(self, trial, X, y, target_name):
        model_type = trial.suggest_categorical('model_type', ['rf', 'gb'])
        
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            model = RandomForestRegressor(**params, random_state=self.random_state)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
            }
            model = GradientBoostingRegressor(**params, random_state=self.random_state)
        
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        return scores.mean()
    
    def build_stacking_model(self, X, y, target_name):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.optimize_model(trial, X, y, target_name), n_trials=20)
        
        self.base_rf_models[target_name] = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10,
            random_state=self.random_state
        )
        self.base_rf_models[target_name].fit(X, y)
        
        best_model_type = study.best_params['model_type']
        if best_model_type == 'rf':
            optimized_model = RandomForestRegressor(
                n_estimators=study.best_params['n_estimators'],
                max_depth=study.best_params['max_depth'],
                min_samples_split=study.best_params['min_samples_split'],
                min_samples_leaf=study.best_params['min_samples_leaf'],
                random_state=self.random_state
            )
        else:
            optimized_model = GradientBoostingRegressor(
                n_estimators=study.best_params['n_estimators'],
                learning_rate=study.best_params['learning_rate'],
                max_depth=study.best_params['max_depth'],
                subsample=study.best_params['subsample'],
                random_state=self.random_state
            )
        
        rf = RandomForestRegressor(random_state=self.random_state)
        gb = GradientBoostingRegressor(random_state=self.random_state)
        svr = SVR(kernel='rbf')
        lasso = LassoCV(random_state=self.random_state)
        ridge = RidgeCV()
        
        estimators = [
            ('optimized', optimized_model),
            ('rf', rf),
            ('gb', gb),
            ('svr', svr),
            ('lasso', lasso),
            ('ridge', ridge)
        ]
        
        return StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(random_state=self.random_state),
            cv=5
        ), estimators
    
    def fit(self, X, y_dict):
        X_scaled = self.preprocess_data(X)
        
        for target_name, y in y_dict.items():
            print(f"\nTraining model for {target_name}...")
            model, base_estimators = self.build_stacking_model(X_scaled, y, target_name)
            model.fit(X_scaled, y)
            self.models[target_name] = model
            
            self.feature_importances[target_name] = pd.Series(
                self.base_rf_models[target_name].feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

            self.base_model_results[target_name] = {}
            for name, base_model in base_estimators:
                base_model.fit(X_scaled, y)
                y_pred = base_model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                self.base_model_results[target_name][name] = {'MSE': mse, 'R2': r2}
    
    def predict(self, X):
        X_scaled = self.preprocess_data(X)
        predictions = {}
        
        for target_name, model in self.models.items():
            predictions[target_name] = model.predict(X_scaled)
            
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
                    'R2': r2
                }
            }
            
            if target_name in self.feature_importances:
                top_features = self.feature_importances[target_name].head(5)
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
            }).head(5)
            results[target_name]['SamplePredictions'] = sample_predictions.to_dict(orient='records')
        
        return json.dumps(results, indent=4)
    
    def save_results(self, results, file_name="model_results.json"):
        with open(file_name, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    df = pd.read_csv('super_model_synthetic_dataset.csv')
    data = df.dropna()

    target_columns = ['linear_performance', 'nonlinear_performance', 'tree_performance']
    feature_columns = [col for col in data.columns if col not in target_columns]

    X = data[feature_columns]
    y_dict = {target: data[target] for target in target_columns}

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42, shuffle=False)

    y_train_dict = {
        target: data[target].loc[X_train.index] for target in target_columns
    }
    y_test_dict = {
        target: data[target].loc[X_test.index] for target in target_columns
    }


    pipeline = AdvancedRegressionPipeline()
    pipeline.fit(X_train, y_train_dict)

    results = pipeline.evaluate(X_test, y_test_dict)

    print("Model Evaluation Results:")
    print(results)

    pipeline.save_results(
        results=results,
        file_name="model_results.json"
    )