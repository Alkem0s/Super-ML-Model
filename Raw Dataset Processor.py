import ast
import numpy as np
import pandas as pd


def safe_dict_convert(dict_str):
    try:
        return ast.literal_eval(dict_str) if isinstance(dict_str, str) else dict_str
    except:
        return {}

def get_model_groups(columns):
    model_groups = {}
    
    metric_cols = [col for col in columns if 'f1_macro_mean' in col]
    
    for col in metric_cols:
        model_prefix = col.split('_f1_macro_mean')[0]
        
        model_type = model_prefix.split('_')[0]
        
        if model_type not in model_groups:
            model_groups[model_type] = []
        model_groups[model_type].append(model_prefix)
    
    return model_groups

def calculate_model_performance(row, model_prefixes):
    f1_mean = []
    precision_mean = []
    recall_mean = []
    
    for prefix in model_prefixes:
        metrics = {
            'f1': row.get(f'{prefix}_f1_macro_mean', np.nan),
            'precision': row.get(f'{prefix}_precision_macro_mean', np.nan),
            'recall': row.get(f'{prefix}_recall_macro_mean', np.nan),
        }
        
        if not pd.isna(metrics['f1']) and not pd.isna(metrics['precision']) and not pd.isna(metrics['recall']):
            f1_mean.append(metrics['f1'])
            precision_mean.append(metrics['precision'])
            recall_mean.append(metrics['recall'])
    
    if f1_mean and precision_mean and recall_mean:
        aggregated_f1 = np.mean(f1_mean)
        aggregated_precision = np.mean(precision_mean)
        aggregated_recall = np.mean(recall_mean)
        
        aggregated_performance = (aggregated_f1 + aggregated_precision + aggregated_recall) / 3
        return aggregated_performance
    else:
        return None

def clean_column_value(value):
    try:
        return ast.literal_eval(value)[0] if isinstance(value, str) and value.startswith("(") and value.endswith(",)") else value
    except Exception:
        return value

def organize_ml_data(input_file):
    df = pd.read_csv(input_file)
    
    model_groups = get_model_groups(df.columns)
    
    all_metadata = []
    
    for _, row in df.iterrows():
        num_samples_original = row['num_samples_original']
        num_samples_after_preprocessing = row['num_samples_after_preprocessing']
        num_features_original = row['num_features_original']
        num_features_after_preprocessing = row['num_features_after_preprocessing']
        num_classes = row['num_classes']
        
        feature_types_dict = safe_dict_convert(row['feature_types'])
        binary_features = feature_types_dict.get('binary', 0)
        categorical_features = feature_types_dict.get('categorical', 0)
        numeric_features = feature_types_dict.get('numeric', 0)
        
        class_dist = safe_dict_convert(row['class_distribution'])
        class_dist_values = list(class_dist.values())
        class_imbalance = max(class_dist_values) / min(class_dist_values) if min(class_dist_values) > 0 else float('inf')
        
        complexity_metrics = {
            'variance_explained': row.get('variance_explained_5_components', 0),
            'avg_feature_correlation': row.get('avg_feature_correlation', 0),
            'max_correlation': row.get('max_correlation', 0),
            'intrinsic_dimensionality': row.get('intrinsic_dimensionality', 0),
            'pca_variance': row.get('pca_variance', 0),
            'outlier_fraction': row.get('outlier_fraction', 0)
        }
        
        performance_scores = {
            f"{model_type}_performance": calculate_model_performance(row, prefixes)
            for model_type, prefixes in model_groups.items()
        }
        
        if any(pd.isna(score) or score == 0.0 for score in performance_scores.values()):
            continue
        
        metadata = {
            'num_samples_original': num_samples_original,
            'num_features_original': num_features_original,
            'num_samples_after_preprocessing': num_samples_after_preprocessing,
            'num_features_after_preprocessing': num_features_after_preprocessing,
            'num_classes': num_classes,
            'binary_features': binary_features,
            'categorical_features': categorical_features,
            'numeric_features': numeric_features,
            'class_imbalance': class_imbalance,
            
            **complexity_metrics,
            **performance_scores
        }
        
        if any(pd.isna(value) for value in metadata.values()):
            continue
        
        all_metadata.append(metadata)
    
    metadata_df = pd.DataFrame(all_metadata)
    
    metadata_df.dropna()
    
    if 'num_samples_original' in df.columns:
        df['num_samples_original'] = df['num_samples_original'].apply(clean_column_value)

    if 'num_features_original' in df.columns:
        df['num_features_original'] = df['num_features_original'].apply(clean_column_value)
    
    metadata_df.to_csv('super_model_processed_dataset.csv', index=False)
    
    return metadata_df

if __name__ == "__main__":
    openml_datasets = "super_model_raw_dataset_openml.csv"
    kaggle_datasets = "super_model_raw_dataset_kaggle.csv"

    df1 = pd.read_csv(openml_datasets)
    df2 = pd.read_csv(kaggle_datasets)

    df = pd.concat([df1, df2], ignore_index=True)

    output_file = "super_model_raw_dataset.csv"
    df.to_csv(output_file, index=False)
    
    metadata = organize_ml_data('super_model_raw_dataset.csv')