import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import spearmanr
from scipy.optimize import minimize
import warnings

class EnhancedSyntheticDataGenerator:
    def __init__(self, input_file):
        self.df = pd.read_csv(input_file)
        self._calculate_statistics()
        self._fit_models()
        
    def _calculate_statistics(self):
        self.means = self.df.mean()
        self.stds = self.df.std()
        self.mins = self.df.min()
        self.maxs = self.df.max()
        self.correlations = self.df.corr()
        
        self.feature_groups = {
            'samples': ['num_samples_original', 'num_samples_after_preprocessing'],
            'features': ['num_features_original', 'num_features_after_preprocessing',
                        'binary_features', 'categorical_features', 'numeric_features'],
            'performance': ['linear_performance', 'nonlinear_performance', 'tree_performance'],
            'correlations': ['avg_feature_correlation', 'max_correlation']
        }
        
    def _fit_models(self):
        self.gmms = {}
        for group_name, features in self.feature_groups.items():
            group_data = self.df[features].values
            n_components = min(len(self.df) // 10, 5)
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42
            )
            gmm.fit(group_data)
            self.gmms[group_name] = gmm
            
    def _adjust_feature_relationships(self, synthetic_data):
        synthetic_data.loc[synthetic_data['num_samples_after_preprocessing'] > 
                         synthetic_data['num_samples_original'], 
                         'num_samples_after_preprocessing'] = \
            synthetic_data['num_samples_original']
            
        features_cols = ['binary_features', 'categorical_features', 'numeric_features']
        total_features = synthetic_data['num_features_after_preprocessing']
        
        features_sum = synthetic_data[features_cols].sum(axis=1)
        for col in features_cols:
            synthetic_data[col] = np.round(
                synthetic_data[col] * total_features / features_sum
            )
            
        synthetic_data[features_cols] = synthetic_data[features_cols].clip(lower=0)
        synthetic_data[features_cols] = synthetic_data[features_cols].round()
        
        performance_cols = ['linear_performance', 'nonlinear_performance', 'tree_performance']
        synthetic_data[performance_cols] = synthetic_data[performance_cols].clip(0, 1)
        
        synthetic_data['max_correlation'] = synthetic_data['max_correlation'].clip(
            synthetic_data['avg_feature_correlation'], 1
        )
        synthetic_data['avg_feature_correlation'] = synthetic_data['avg_feature_correlation'].clip(0, 1)
        
        return synthetic_data
        
    def _generate_synthetic_samples(self, n_samples):
        synthetic_data = pd.DataFrame()
        
        for group_name, features in self.feature_groups.items():
            group_samples, _ = self.gmms[group_name].sample(n_samples)
            group_df = pd.DataFrame(group_samples, columns=features)
            
            for col in features:
                noise = np.random.normal(0, self.stds[col] * 0.05, n_samples)
                group_df[col] += noise
                
                group_df[col] = np.clip(
                    group_df[col],
                    self.mins[col],
                    self.maxs[col]
                )
                
                if np.issubdtype(self.df[col].dtype, np.integer):
                    group_df[col] = np.round(group_df[col])
            
            synthetic_data = pd.concat([synthetic_data, group_df], axis=1)
        
        remaining_features = [col for col in self.df.columns 
                            if not any(col in group 
                                     for group in self.feature_groups.values())]
        
        if remaining_features:
            remaining_gmm = GaussianMixture(
                n_components=min(len(self.df) // 10, 5),
                covariance_type='full',
                random_state=42
            )
            remaining_gmm.fit(self.df[remaining_features])
            remaining_samples, _ = remaining_gmm.sample(n_samples)
            remaining_df = pd.DataFrame(remaining_samples, columns=remaining_features)
            
            for col in remaining_features:
                noise = np.random.normal(0, self.stds[col] * 0.05, n_samples)
                remaining_df[col] += noise
                remaining_df[col] = np.clip(
                    remaining_df[col],
                    self.mins[col],
                    self.maxs[col]
                )
                if np.issubdtype(self.df[col].dtype, np.integer):
                    remaining_df[col] = np.round(remaining_df[col])
            
            synthetic_data = pd.concat([synthetic_data, remaining_df], axis=1)
        
        synthetic_data = synthetic_data[self.df.columns]
        
        synthetic_data = self._adjust_feature_relationships(synthetic_data)
        
        return synthetic_data
    
    def generate_and_save(self, n_samples, output_file, include_original=True):
        synthetic_df = self._generate_synthetic_samples(n_samples)
        
        if include_original:
            final_df = pd.concat([self.df, synthetic_df], ignore_index=True)
        else:
            final_df = synthetic_df
            
        final_df = final_df.dropna()  
            
        final_df.to_csv(output_file, index=False)
        
        print(f"Generated {n_samples} synthetic samples")
        print(f"Original dataset size: {len(self.df)}")
        print(f"Final dataset size: {len(final_df)}")
        
        self._print_validation_metrics(synthetic_df)
        
        return final_df
    
    def _print_validation_metrics(self, synthetic_df):
        print("\nValidation Metrics:")
        
        for col in self.df.columns:
            orig_mean = self.df[col].mean()
            syn_mean = synthetic_df[col].mean()
            mean_diff_pct = abs(orig_mean - syn_mean) / orig_mean * 100
            
            orig_std = self.df[col].std()
            syn_std = synthetic_df[col].std()
            std_diff_pct = abs(orig_std - syn_std) / orig_std * 100
            
            print(f"\n{col}:")
            print(f"  Mean difference: {mean_diff_pct:.2f}%")
            print(f"  Std difference: {std_diff_pct:.2f}%")

if __name__ == "__main__":
    input_file = "super_model_processed_dataset.csv"
    output_file = "super_model_synthetic_dataset.csv"
    
    generator = EnhancedSyntheticDataGenerator(input_file)
    synthetic_data = generator.generate_and_save(
        n_samples=300,
        output_file=output_file,
        include_original=True
    )