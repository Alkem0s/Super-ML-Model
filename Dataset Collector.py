import time
import tracemalloc
import warnings
import pandas as pd
import numpy as np
import openml
import os
import math
import chardet
import shutil
from typing import Dict, Any, List
from collections import Counter
import re
import psutil

from imblearn.over_sampling import ADASYN
from scipy import stats
from sklearn.calibration import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from catboost import CatBoostClassifier

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    try:
        from kaggle import KaggleApi
    except ImportError:
        KaggleApi = None

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

import shutil
import os

class FolderAlternator:
    def __init__(self, base_dir):
        self.folder1 = os.path.join(base_dir, "kaggle_temp_1")
        self.folder2 = os.path.join(base_dir, "kaggle_temp_2")
        self.current_folder = self.folder1

        os.makedirs(self.folder1, exist_ok=True)
        os.makedirs(self.folder2, exist_ok=True)

    def get_current_folder(self):
        return self.current_folder

    def switch_and_cleanup(self):
        next_folder = self.folder2 if self.current_folder == self.folder1 else self.folder1

        try:
            shutil.rmtree(next_folder)
            os.makedirs(next_folder)
        except Exception as e:
            print(f"Error cleaning folder {next_folder}: {e}")

        self.current_folder = next_folder

class SuperModelDatasetCollector:
    def __init__(self, 
                 num_datasets=15, 
                 output_dir='super_model_datasets',
                 sources=None,
                 kaggle_config_path='C:\\Users\\Alkem0s\\.kaggle\\kaggle.json'):
        
        self.num_datasets = num_datasets
        self.output_dir = output_dir
        self.sources = sources
        os.makedirs(output_dir, exist_ok=True)
        
        self.kaggle_api = self._setup_kaggle_api(kaggle_config_path) if KaggleApi else None
    
    
    def _setup_kaggle_api(self, kaggle_config_path=None):
        if KaggleApi is None:
            print("Kaggle API is not available.")
            return None
        
        possible_locations = [
            kaggle_config_path,
            os.path.expanduser('C:\\Users\\Alkem0s\\.kaggle\\kaggle.json'),
            os.path.join(os.getcwd(), 'kaggle.json')
        ]
        
        for location in possible_locations:
            if location and os.path.exists(location):
                try:
                    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(location)
                    
                    api = KaggleApi()
                    api.authenticate()
                    
                    print(f"Kaggle API successfully configured using {location}")
                    return api
                except Exception as e:
                    print(f"Failed to configure Kaggle API with {location}: {e}")
        
        print("Could not find or configure Kaggle API. Kaggle datasets will be skipped.")
        return None
    
    
    def smart_downsample(self, X, y, target_samples=25000, min_samples_per_class=10):
        try:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
                
            class_counts = y.value_counts()
            n_classes = len(class_counts)
            
            total_samples = len(y)
            sampling_ratios = class_counts / total_samples
            
            target_per_class = {
                cls: max(
                    min_samples_per_class,
                    min(
                        int(target_samples * ratio),
                        class_counts[cls],
                        int(target_samples * 0.5)
                    )
                )
                for cls, ratio in sampling_ratios.items()
            }
            
            total_allocated = sum(target_per_class.values())
            if total_allocated < target_samples:
                remaining = target_samples - total_allocated
                for cls in target_per_class:
                    additional = int(remaining * sampling_ratios[cls])
                    target_per_class[cls] = min(
                        target_per_class[cls] + additional,
                        class_counts[cls]
                    )
            
            sampled_indices = []
            
            for cls in target_per_class:
                cls_indices = y[y == cls].index
                
                if len(cls_indices) <= target_per_class[cls]:
                    sampled_indices.extend(cls_indices)
                else:
                    cls_X = X.loc[cls_indices]

                    if len(cls_indices) > 100:
                        try:
                            clf = DecisionTreeClassifier(
                                max_depth=5,
                                random_state=42
                            )
                            binary_y = (y == cls).astype(int)
                            clf.fit(X, binary_y)
                            importance = clf.feature_importances_
                            
                            weighted_samples = np.abs(
                                cls_X.values @ importance
                            )
                            weights = weighted_samples / weighted_samples.sum()
                        except:
                            weights = None
                    else:
                        weights = None
                    
                    sampled_cls_indices = np.random.choice(
                        cls_indices,
                        size=target_per_class[cls],
                        replace=False,
                        p=weights
                    )
                    sampled_indices.extend(sampled_cls_indices)
            
            X_downsampled = X.loc[sampled_indices]
            y_downsampled = y.loc[sampled_indices]
            
            original_ratios = sampling_ratios
            new_ratios = y_downsampled.value_counts() / len(y_downsampled)
            
            max_ratio_diff = max(
                abs(original_ratios[cls] - new_ratios[cls])
                for cls in original_ratios.index
            )
            
            if max_ratio_diff > 0.1:
                print("Warning: Class distribution changed significantly after downsampling")
                
            return X_downsampled, y_downsampled
        
        except Exception as e:
            print(f"Error in smart_downsample: {e}")
            return X, y
    
    
    def extract_target_column(self, X):
        exact_columns = ['class', 'target', 'response', 'output', 'y']
        partial_columns = ['label', 'result', 'flag', 'quality', 'symbol', 'symboling']
        
        for col in X.columns:
            if col.lower() in exact_columns:
                unique_vals = X[col].nunique()
                if unique_vals > 1:
                    print(f"Using '{col}' as target column with {unique_vals} unique values.")
                    return X[col], X.drop(columns=[col])
        
        for col in X.columns:
            if any(keyword in col.lower() for keyword in partial_columns):
                unique_vals = X[col].nunique()
                if unique_vals > 1 and unique_vals <= 50:
                    print(f"Using '{col}' as target column with {unique_vals} unique values.")
                    return X[col], X.drop(columns=[col])
        
        if X.shape[1] > 1:
            last_col = X.columns[-1]
            if (isinstance(last_col, str) and
                len(last_col) <= 63 and
                last_col.replace('_','').isalnum() and
                not last_col[0].isdigit()):
                
                unique_vals = X[last_col].nunique()
                if unique_vals > 1 and unique_vals <= 50:
                    print(f"Using last column '{last_col}' as target with {unique_vals} unique values.")
                    return X[last_col], X.drop(columns=[last_col])
        
        print("No suitable target column found.")
        return None, None
    
    
    def is_imbalanced(self, y, threshold=0.99):
        try:
            if isinstance(y, list):
                y = pd.Series(y)

            if isinstance(y.dtype, pd.CategoricalDtype) or not np.issubdtype(y.dtype, np.number):
                y = y.astype(str)
                le = LabelEncoder()
                y = le.fit_transform(y)

            class_counts = np.bincount(y)
            
            if len(class_counts) == 0:
                return False
            
            max_ratio = np.max(class_counts) / np.sum(class_counts)
            return max_ratio > threshold
        except Exception as e:
            print(f"Error in is_imbalanced: {e}")
            return False


    def is_dataset_suitable(self, X, y) -> bool:
        try:
            if X is None or y is None:
                print("Dataset rejected: Missing features or target")
                return X, y, False

            if len(X) != len(y):
                print("Dataset rejected: Mismatched lengths between features and target")
                return X, y, False

            if not isinstance(y, pd.Series):
                y = pd.Series(y)

            initial_features = X.shape[1]

            min_samples = 50
            max_features = 500

            if len(X) < min_samples:
                print(f"Dataset rejected: Too few samples ({len(X)} < {min_samples})")
                return X, y, False

            if X.shape[1] > max_features:
                print(f"Dataset rejected: Too many features ({X.shape[1]} > {max_features})")
                return X, y, False

            value_counts = y.value_counts()
            unique_classes = len(value_counts)

            if unique_classes < 2:
                print("Dataset rejected: Less than 2 classes")
                return X, y, False

            if unique_classes > 50:
                print(f"Dataset rejected: Too many classes ({unique_classes} > 50)")
                return X, y, False

            min_samples_per_class = max(5, len(X) // 1000)
            valid_classes = value_counts[value_counts >= min_samples_per_class].index

            if len(valid_classes) < 2:
                print("Dataset rejected: Less than 2 valid classes after filtering rare classes")
                return X, y, False

            valid_samples_mask = y.isin(valid_classes)
            X = X[valid_samples_mask]
            y = y[valid_samples_mask]

            filtered_counts = y.value_counts()
            
            majority_class_ratio = filtered_counts.iloc[0] / len(y)
            if majority_class_ratio > 0.99:
                print(f"Dataset rejected: Extreme class imbalance (majority class ratio: {majority_class_ratio:.3f})")
                return X, y, False

            target_samples = 25000
            if len(X) > target_samples:
                print(f"Downsampling dataset from {len(X)} to {target_samples} samples...")
                try:
                    X, y = self.smart_downsample(X, y, target_samples=target_samples)
                except Exception as e:
                    print(f"Downsampling failed: {e}")
                    return X, y, False

            try:
                missing_threshold = 0.3
                missing_values_count = X.isna().mean()
                columns_to_drop = missing_values_count[missing_values_count > missing_threshold].index
                
                if len(columns_to_drop) > 0:
                    X = X.drop(columns=columns_to_drop)

                if X.shape[1] == 0:
                    print("Dataset rejected: No features left after dropping columns with too many missing values")
                    return X, y, False

                numeric_cols = X.select_dtypes(include=[np.number]).columns
                categorical_cols = X.select_dtypes(exclude=[np.number]).columns

                if len(numeric_cols) > 0:
                    X.loc[:, numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                
                if len(categorical_cols) > 0:
                    for col in categorical_cols:
                        X.loc[:, col] = X[col].fillna(X[col].mode().iloc[0])

            except Exception as e:
                print(f"Warning: Missing value handling failed: {e}")
                return X, y, False

            try:
                if len(numeric_cols) > 0:
                    variance_threshold = VarianceThreshold(threshold=0.01)
                    numeric_data = X[numeric_cols]
                    variance_mask = variance_threshold.fit(numeric_data).get_support()
                    kept_numeric_cols = numeric_cols[variance_mask]
                    
                    dropped_features = len(numeric_cols) - len(kept_numeric_cols)
                    if dropped_features > 0:
                        print(f"Dropped {dropped_features} numeric features due to low variance")
                    
                    final_columns = list(kept_numeric_cols) + list(categorical_cols)
                    X = X[final_columns]

            except Exception as e:
                print(f"Warning: Variance thresholding failed: {e}")

            if X.shape[1] < 2:
                print(f"Dataset rejected: Insufficient features ({X.shape[1]} < 2)")
                return X, y, False

            estimated_memory_mb = (X.memory_usage().sum() + y.memory_usage()) / (1024 * 1024)
            if estimated_memory_mb > 1024:
                print(f"Dataset rejected: Memory footprint too large ({estimated_memory_mb:.1f} MB)")
                return False

            features_kept = X.shape[1]
            features_dropped = initial_features - features_kept
            
            print(f"Dataset accepted: {len(X)} samples, {features_kept} features kept "
                  f"({features_dropped} dropped), {len(filtered_counts)} classes")

            return X, y, True

        except Exception as e:
            print(f"Error in dataset validation: {e}")
            return X, y, False


    def compute_knn_intrinsic_dimensionality(self, X, k_min=5, k_max=50, max_samples=5000):
        try:
            X_numeric = X.select_dtypes(include=[np.number]) if isinstance(X, pd.DataFrame) else X
            if isinstance(X_numeric, pd.DataFrame) and X_numeric.empty:
                return None
            
            if isinstance(X_numeric, pd.DataFrame):
                X_numeric = X_numeric.fillna(X_numeric.mean())
                X_numeric = X_numeric.loc[:, X_numeric.std() > 0]
                if X_numeric.empty:
                    return None
            
            if len(X_numeric) > max_samples:
                if isinstance(X_numeric, pd.DataFrame):
                    X_numeric = X_numeric.sample(max_samples, random_state=42)
                else:
                    indices = np.random.RandomState(42).choice(len(X_numeric), max_samples, replace=False)
                    X_numeric = X_numeric[indices]
            
            X_scaled = StandardScaler().fit_transform(X_numeric)
            
            n_samples = len(X_scaled)
            k_max = min(k_max, n_samples - 1)
            k_min = min(k_min, k_max - 1)
            
            if k_min >= k_max or k_max <= 1:
                return None
            
            k_values = np.unique(np.logspace(np.log10(k_min), np.log10(k_max), num=min(10, k_max - k_min + 1)).astype(int))
            
            knn = NearestNeighbors(n_neighbors=k_max + 1, n_jobs=-1)
            knn.fit(X_scaled)
            distances, _ = knn.kneighbors(X_scaled)
            
            distances = distances[:, 1:]
            mean_distances = np.array([np.mean(distances[:, :k]) for k in k_values])
            mean_distances[mean_distances == 0] = np.min(mean_distances[mean_distances > 0]) / 10
            
            log_k = np.log(k_values)
            log_distances = np.log(mean_distances)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_distances)
            
            if not np.isfinite(slope) or p_value > 0.05 or r_value ** 2 < 0.8:
                return None
                
            intrinsic_dim = max(1.0, min(X_numeric.shape[1], 2.0 / slope))
            
            return float(intrinsic_dim) if 0.1 <= intrinsic_dim <= X_numeric.shape[1] * 2 else None
                
        except Exception as e:
            return None


    def compute_pca_variance(self, X, explained_variance_threshold=0.95):
        try:
            X_numeric = X.select_dtypes(include=[np.number])
            pca = PCA()
            pca.fit(X_numeric)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
            explained_variance = cumulative_variance[n_components - 1]
            return n_components, explained_variance
        except Exception as e:
            print(f"Error computing PCA variance: {e}")
            return None, None


    def compute_outlier_metrics(self, X):
        try:
            X_numeric = X.select_dtypes(include=[np.number])
            z_scores = np.abs((X_numeric - X_numeric.mean()) / X_numeric.std())
            outlier_fraction = (z_scores > 3).sum().sum() / X_numeric.size
            return outlier_fraction
        except Exception as e:
            print(f"Error computing outlier metrics: {e}")
            return None
    
    
    def get_feature_types(self, X):
        feature_types_map = Counter()

        for col in X.columns:
            unique_vals = X[col].nunique()
            dtype = X[col].dtype

            if unique_vals == 2:
                feature_types_map['binary'] += 1
                continue

            if isinstance(dtype, pd.CategoricalDtype):
                feature_types_map['categorical'] += 1
                continue

            elif np.issubdtype(dtype, np.number):
                feature_types_map['numeric'] += 1

            elif dtype == 'object' or dtype.name == 'category':
                feature_types_map['categorical'] += 1

            else:
                feature_types_map['unknown'] += 1

        return feature_types_map
    
    
    def preprocess_target_column(self, y):
        try:
            if y is None or len(y) == 0 or y.isnull().all():
                print("Warning: Target column is empty or contains only NaN values, skipping dataset.")
                return None

            if y.dtype.kind in ['O', 'U', 'S'] or any(isinstance(val, (str, bytes)) for val in y.dropna().head()):
                y = y.astype(str)

            elif y.dtype.kind in ['i', 'f']:
                y = pd.to_numeric(y, errors='coerce')

            if y.isnull().any():
                print("Warning: Target column contains NaNs after processing, skipping dataset.")
                return None

            return y

        except Exception as e:
            print(f"Error processing target column: {e}")
            return None
    
    
    def fetch_data(self, dataset):
        try:
            X, y, _, _ = dataset.get_data()
            return X, y
        except Exception as e:
            return None, str(e)
    
    
    def collect_openml_datasets(self, num_datasets):
        datasets = openml.datasets.list_datasets(output_format='dataframe')

        datasets = datasets[
            (datasets['status'] == 'active') &
            (datasets['NumberOfClasses'].notna()) &
            (datasets['NumberOfClasses'] > 1)
        ]

        suitable_datasets = []

        for dataset_id in datasets['did'][:num_datasets * 3]:
            try:
                dataset = openml.datasets.get_dataset(dataset_id)
                X, y = self.fetch_data(dataset)
                if X is None:
                    continue

                y, X = self.extract_target_column(X)
                if y is None:
                    continue

                y = self.preprocess_target_column(y)
                if y is None:
                    continue

                num_samples = X.shape[0]
                num_features = X.shape[1]

                X, y, suitable = self.is_dataset_suitable(X, y)
                if not suitable:
                    continue

                if y.isnull().any():
                    print(f"Target variable contains missing values for dataset {dataset_id}.")
                    continue
                
                unique_classes, class_counts = np.unique(y, return_counts=True)
                num_classes = len(unique_classes)

                if num_classes > 50:
                    print(f"Dataset {dataset_id} has too many classes ({num_classes}).")
                    continue
                
                if hasattr(X, 'dtypes'):
                   feature_types_map = self.get_feature_types(X)

                if hasattr(y, 'name'):
                    X_without_target = X.drop(columns=[y.name]) if y.name in X.columns else X
                else:
                    X_without_target = X

                results = self.extract_advanced_metadata(X_without_target, y)
                if len(results) == 0:
                    continue

                metrics = results['metrics']
                complexity_features = results['complexity_features']
                
                model_metrics = {}
                for model_type in ['linear_models', 'nonlinear_models', 'tree_models']:
                    for model, metrics_dict in metrics.get(model_type, {}).items():
                        for metric_name, value in metrics_dict.items():
                            column_name = f"{model_type}_{model}_{metric_name}"
                            model_metrics[column_name] = value

                complexity_metrics = {
                    "variance_explained_5_components": complexity_features.get('variance_explained_5_components', None),
                    "avg_feature_correlation": complexity_features.get('avg_feature_correlation', None),
                    "max_correlation": complexity_features.get('max_correlation', None),
                }

                intrinsic_dimensionality = complexity_features.get('intrinsic_dimensionality', None)
                pca_variance = complexity_features.get('pca_variance', None)
                outlier_fraction = complexity_features.get('outlier_fraction', None)

                metadata = {
                    'source': 'openml',
                    'name': dataset.name,
                    'num_samples_original': num_samples,
                    'num_features_original': num_features,
                    'num_samples_after_preprocessing': X.shape[0],
                    'num_features_after_preprocessing': X.shape[1],
                    'num_classes': len(np.unique(y)),
                    'class_distribution': dict(Counter(y)),
                    'feature_types': dict(feature_types_map),
                    **model_metrics,
                    **complexity_metrics,
                    'intrinsic_dimensionality': intrinsic_dimensionality,
                    'pca_variance': str(pca_variance) if pca_variance else None,
                    'outlier_fraction': outlier_fraction,
                }
                
                
                suitable_datasets.append(metadata)
                
                df = pd.DataFrame(suitable_datasets)
                            
                df.to_csv('super_model_raw_dataset_openml.csv', index=False)

                if len(suitable_datasets) >= num_datasets:
                    break

            except Exception as e:
                print(f"Error processing OpenML dataset {dataset_id}: {e}")

        return suitable_datasets

    def read_csv(self, file_path, encoding_fallbacks=['utf-8', 'ISO-8859-1', 'latin1', 'Windows-1252']):
        for encoding in encoding_fallbacks:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return df
            except UnicodeDecodeError:
                print(f"Encoding {encoding} failed for {file_path}")
        raise ValueError(f"Unable to read {file_path} with any encoding.")
    
    def process_single_dataset(self, num_datasets) -> Dict[str, Any]:
        try:
            file_path = 'data.csv'
            df = self.read_csv(file_path)
            
            y, X = self.extract_target_column(df)
            if y is None:
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
            
            y = self.preprocess_target_column(y)
            if y is None:
                y = df.iloc[:, -1]
            
            num_samples = X.shape[0]
            num_features = X.shape[1]
            
            unique_classes = np.unique(y)
            class_distribution = dict(Counter(y))
            
            if hasattr(X, 'dtypes'):
                feature_types_map = self.get_feature_types(X)

            if hasattr(y, 'name'):
                X_without_target = X.drop(columns=[y.name]) if y.name in X.columns else X
            else:
                X_without_target = X

            results = self.extract_advanced_metadata(X_without_target, y)
            
            metrics = results['metrics']
            complexity_features = results['complexity_features']

            model_metrics = {}
            for model_type in ['linear_models', 'nonlinear_models', 'tree_models']:
                for model, metrics_dict in metrics.get(model_type, {}).items():
                    for metric_name, value in metrics_dict.items():
                        column_name = f"{model_type}_{model}_{metric_name}"
                        model_metrics[column_name] = value
            
            complexity_metrics = {
                "variance_explained_5_components": complexity_features.get('variance_explained_5_components', None),
                "avg_feature_correlation": complexity_features.get('avg_feature_correlation', None),
                "max_correlation": complexity_features.get('max_correlation', None),
            }

            intrinsic_dimensionality = complexity_features.get('intrinsic_dimensionality', None)
            pca_variance = complexity_features.get('pca_variance', None)
            outlier_fraction = complexity_features.get('outlier_fraction', None)
            
            metadata = {
                'source': 'local_file',
                'name': os.path.basename(file_path),
                'num_samples_original': num_samples,
                'num_features_original': num_features,
                'num_samples_after_preprocessing': X.shape[0],
                'num_features_after_preprocessing': X.shape[1],
                'num_classes': len(unique_classes),
                'class_distribution': str(class_distribution),
                'feature_types': dict(feature_types_map),
                'dataset_type': results.get('dataset_type', 'unknown'),
                'evaluation_status': results.get('evaluation_status', 'unknown'),
                **model_metrics,
                **complexity_metrics,
                'intrinsic_dimensionality': intrinsic_dimensionality,
                'pca_variance': str(pca_variance) if pca_variance else None,
                'outlier_fraction': outlier_fraction,
            }
            
            dataset = [metadata]
            
            return dataset
            
        except Exception as e:
            return {
                'source': 'local_file',
                'name': os.path.basename(file_path),
                'error': str(e),
                'evaluation_status': 'Failed'
            }

    def collect_kaggle_datasets(self, num_datasets):
        if not self.kaggle_api:
            print("Kaggle API not configured. Skipping Kaggle datasets.")
            return []
        
        folder_manager = FolderAlternator(self.output_dir)
        
        suitable_datasets = []
        
        DATASET_BLACKLIST = {
            'aaron7sun/stocknews',
            'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
            'nicapotato/womens-ecommerce-clothing-reviews',
            'arashnic/hr-analytics-job-change-of-data-scientists',
            'olistbr/brazilian-ecommerce',
            'agileteam/bigdatacertificationkr',
            'rounakbanik/pokemon',
            'arnabchaki/data-science-salaries-2023',
            'jsphyg/weather-dataset-rattle-package',
            'datatattle/covid-19-nlp-text-classification',
            'shivamb/real-or-fake-fake-jobposting-prediction',
            'nelgiriyewithana/apple-quality',
            'ahsan81/hotel-reservations-classification-dataset',
            'mansoordaku/ckdisease',
            'yasserh/wine-quality-dataset',
            'uciml/red-wine-quality-cortez-et-al-2009',
            'nelgiriyewithana/top-spotify-songs-2023',
            'carrie1/ecommerce-data',
            'nelgiriyewithana/global-youtube-statistics-2023',
            'nelgiriyewithana/most-streamed-spotify-songs-2024,'
        }   
        
        try:
            datasets = []
            page = 1
            while len(datasets) < num_datasets:
                try:
                    page_datasets = self.kaggle_api.dataset_list(
                        search='classification',
                        file_type='csv',
                        sort_by="votes",
                        page=page,
                    )
                    
                    if not page_datasets:
                        break
                    
                    page_datasets = [
                        dataset for dataset in page_datasets 
                        if dataset.ref not in DATASET_BLACKLIST
                    ]
                    
                    datasets.extend(page_datasets)
                    page += 1
                    
                except Exception as e:
                    print(f"Error fetching Kaggle datasets on page {page}: {e}")
                    break
            
            for dataset in datasets:        
                try:            
                    folder_manager.switch_and_cleanup()
                    
                    download_dir = folder_manager.get_current_folder()
                                            
                    if dataset.totalBytes > 100 * 1024 * 1024:
                        continue
                    
                    self.kaggle_api.dataset_download_files(
                        f"{dataset.ref}",
                        path=download_dir, 
                        unzip=True
                    )
                    
                    csv_files = [f for f in os.listdir(download_dir) if f.lower().endswith('.csv')]
                    
                    for csv_file in csv_files:
                        csv_file_path = os.path.join(download_dir, csv_file)
                        try:
                            df = self.read_csv_fully(csv_file_path)

                            y, X = self.extract_target_column(df)
                            if y is None:
                                continue
                            
                            y = self.preprocess_target_column(y)
                            if y is None:
                                continue
                            
                            num_samples = X.shape[0],
                            num_features = X.shape[1],
                            
                            X, y, suitable = self.is_dataset_suitable(X, y)
                            if not suitable:
                                print(f"Dataset {dataset.title} is not suitable.")
                                continue

                            if y.isnull().any():
                                print(f"Target variable contains missing values for dataset {dataset.ref}.")
                                continue
                            
                            unique_classes, class_counts = np.unique(y, return_counts=True)
                            num_classes = len(unique_classes)
                            class_distribution = dict(zip(unique_classes, class_counts))

                            if num_classes > 50:
                                print(f"Dataset {dataset.ref} has too many classes ({num_classes}).")
                                continue

                            feature_types_map = self.get_feature_types(df)
                        
                            results = self.extract_advanced_metadata(df.drop(columns=y.name), y)
                            if len(results) == 0:
                                continue
                                
                            metrics = results['metrics']
                            complexity_features = results['complexity_features']

                            model_metrics = {}
                            for model_type in ['linear_models', 'nonlinear_models', 'tree_models']:
                                for model, metrics_dict in metrics.get(model_type, {}).items():
                                    for metric_name, value in metrics_dict.items():
                                        column_name = f"{model_type}_{model}_{metric_name}"
                                        model_metrics[column_name] = value

                            complexity_metrics = {
                                "variance_explained_5_components": complexity_features.get('variance_explained_5_components', None),
                                "avg_feature_correlation": complexity_features.get('avg_feature_correlation', None),
                                "max_correlation": complexity_features.get('max_correlation', None),
                            }

                            intrinsic_dimensionality = complexity_features.get('intrinsic_dimensionality', None)
                            pca_variance = complexity_features.get('pca_variance', None)
                            outlier_fraction = complexity_features.get('outlier_fraction', None)

                            metadata = {
                                'source': 'kaggle',
                                'name': dataset.title,
                                'num_samples_original': num_samples,
                                'num_features_original': num_features,
                                'num_samples_after_preprocessing': len(df),
                                'num_features_after_preprocessing': df.shape[1],
                                'num_classes': num_classes,
                                'class_distribution': str(class_distribution),
                                'feature_types': dict(feature_types_map),
                                **model_metrics,
                                **complexity_metrics,
                                'intrinsic_dimensionality': intrinsic_dimensionality,
                                'pca_variance': str(pca_variance) if pca_variance else None,
                                'outlier_fraction': outlier_fraction,
                            }
                            
                            
                            suitable_datasets.append(metadata)
                            
                            df = pd.DataFrame(suitable_datasets)
                            
                            df.to_csv('super_model_raw_dataset_kaggle.csv', index=False)
                            
                            if len(suitable_datasets) >= num_datasets:
                                folder_manager.switch_and_cleanup()
                                folder_manager.switch_and_cleanup()
                                break
                        
                        except Exception as e:
                            print(f"Error processing Kaggle dataset {csv_file}: {e}")
                    
                except Exception as e:
                    print(f"Error processing Kaggle dataset {dataset.ref}: {e}")
            
        except Exception as e:
            print(f"Error collecting Kaggle datasets: {e}")
        
        return suitable_datasets
    

    def preprocess_data(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        X_processed = X.copy()
        
        numeric_columns = []
        for col in X_processed.columns:
            series = X_processed[col]
            
            if series.dtype == 'object' or series.dtype.name == 'category':
                if series.nunique() < 10:
                    le = LabelEncoder()
                    X_processed[col] = pd.Series(series).astype(str).fillna('MISSING')
                    X_processed[col] = le.fit_transform(X_processed[col])
                    numeric_columns.append(col)
                else:
                    dummies = pd.get_dummies(series, prefix=col, drop_first=True)
                    X_processed = pd.concat([X_processed, dummies], axis=1)
                    X_processed = X_processed.drop(columns=[col])
                    numeric_columns.extend(dummies.columns)
            else:
                X_processed[col] = pd.to_numeric(series, errors='coerce')
                numeric_columns.append(col)
        
        X_processed = X_processed[numeric_columns]
        
        imputer = SimpleImputer(strategy='mean')
        X_processed = pd.DataFrame(
            imputer.fit_transform(X_processed),
            columns=X_processed.columns
        )
        
        return X_processed


    def compute_feature_complexity(self, X):
        complexity_metrics = {}
        
        try:
            X = X.select_dtypes(include=[np.number])
            if X.empty:
                return {'error': 'No numeric features available'}
            
            X_filled = X.fillna(X.median())
            q1 = X_filled.quantile(0.01)
            q3 = X_filled.quantile(0.99)
            X_filled = X_filled.clip(lower=q1, upper=q3, axis=1)
            
            scaler = StandardScaler()
            chunk_size = len(X_filled) // 10
            chunks = [X_filled.iloc[i:i + chunk_size] for i in range(0, len(X_filled), chunk_size)]

            X_scaled = pd.concat(
                [pd.DataFrame(scaler.partial_fit(chunk).transform(chunk), columns=chunk.columns) 
                for chunk in chunks]
            )
            
            try:
                n_samples, n_features = X_scaled.shape
                n_components = min(5, n_features, n_samples - 1)
                if n_components > 0:
                    svd = TruncatedSVD(
                        n_components=n_components,
                        random_state=42,
                        n_iter=max(2, min(5, n_samples // 100)),
                        tol=0.2
                    )
                    svd.fit(X_scaled)
                    variance_ratio = svd.explained_variance_ratio_
                    complexity_metrics['variance_explained_5_components'] = float(
                        np.sum(variance_ratio[:5])
                    )
            except Exception as e:
                print(f"SVD computation failed: {e}")
                complexity_metrics['variance_explained_5_components'] = None
            
            try:
                if X_scaled.shape[1] > 5000:
                    selected_features = X_scaled.iloc[:, :5000]
                else:
                    selected_features = X_scaled
                corr_matrix = selected_features.corr()
                upper_tri = np.triu(corr_matrix, k=1)
                complexity_metrics['avg_feature_correlation'] = float(
                    np.nanmean(np.abs(upper_tri))
                )
                complexity_metrics['max_correlation'] = float(
                    np.nanmax(np.abs(upper_tri))
                )
            except Exception as e:
                print(f"Error in correlation computation: {e}")
                complexity_metrics['avg_feature_correlation'] = None
                complexity_metrics['max_correlation'] = None
            
            return complexity_metrics
        
        except Exception as e:
            print(f"Feature complexity computation error: {e}")
            return {
                'error': str(e),
                'variance_explained_5_components': None,
                'avg_feature_correlation': None,
                'max_correlation': None
            }


    def balance_data(self, X, y, max_ratio=3):
        try:
            y = pd.Series(y) if not isinstance(y, pd.Series) else y
            
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)
            
            if not np.issubdtype(y.dtype, np.integer):
                y = y.astype(int)
            
            class_counts = np.bincount(y)
            classes = np.arange(len(class_counts))
            
            non_empty_classes = [cls for cls in classes if class_counts[cls] > 0]
            
            if len(non_empty_classes) != len(classes):
                print(f"Warning: Removing {len(classes) - len(non_empty_classes)} empty classes from the dataset.")
            
            mask = y.isin(non_empty_classes)
            
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.loc[mask.index[mask]]
                y = y.loc[mask.index[mask]]
            else:
                X = X[mask.to_numpy()]
                y = y[mask.to_numpy()]

            class_counts = np.bincount(y)
            max_class_size = np.max(class_counts)
            min_target_size = max(5, max_class_size // max_ratio)

            if np.min(class_counts) > 0 and max_class_size / np.min(class_counts) <= max_ratio:
                return X, y
            
            sampling_strategy = {}
            for cls in non_empty_classes:
                current_size = class_counts[cls]
                if current_size < min_target_size:
                    sampling_strategy[cls] = min_target_size
                    
            if not sampling_strategy:
                return X, y
            
            min_class_size = np.min(class_counts[class_counts > 0])
            k_neighbors = max(1, min(5, min_class_size - 1)) 
            
            if k_neighbors <= 0:
                print("Warning: Invalid k_neighbors value computed, skipping resampling.")
                return X, y
            
            try:
                adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
                X_balanced, y_balanced = adasyn.fit_resample(X, y)
                
                new_counts = np.bincount(y_balanced)
                if np.min(new_counts) >= 5 and max(new_counts) / min(new_counts) <= max_ratio * 1.5:
                    return X_balanced, y_balanced
            except Exception as e:
                print(f"ADASYN failed: {e}")
                pass
            
            try:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
                return smote.fit_resample(X, y)
            except Exception as e:
                print(f"SMOTE failed: {e}")
                return X, y
            
        except Exception as e:
            print(f"Error in balance_data: {e}")
            return X, y
        

    def adaptive_stratified_split(self, X, y):
        try:
            class_counts = np.bincount(y)
            min_samples = np.min(class_counts)
            
            if min_samples < 5:
                n_splits = 2
            elif min_samples < 15:
                n_splits = 3
            elif min_samples < 30:
                n_splits = 4
            else:
                n_splits = 5
                
            if hasattr(X, 'index') and isinstance(X.index, pd.DatetimeIndex):
                from sklearn.model_selection import TimeSeriesSplit
                return TimeSeriesSplit(n_splits=n_splits)
                
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
        except Exception as e:
            print(f"Error in adaptive split: {e}. Falling back to default CV.")
            return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


    def process_features(self, X_processed, n_components=None):
        try:
            scaler = StandardScaler()
            X_standardized = scaler.fit_transform(X_processed)
            
            if n_components is None:
                n_components = min(X_processed.shape[0], X_processed.shape[1])
            
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_standardized)
            
            return X_standardized, X_pca, scaler, pca
        except Exception as e:
            print(f"Error in feature processing: {e}")
            return None, None, None, None


    def extract_advanced_metadata(self, X, y) -> Dict[str, Any]:
        performance_metrics = {
            'dataset_type': None,
            'evaluation_status': 'Not Evaluated',
            'metrics': {
                'linear_models': {},
                'nonlinear_models': {},
                'tree_models': {},
            },
            'complexity_features': {}
        }

        try:
            if y is None or X is None:
                performance_metrics['evaluation_status'] = 'Skipped - Invalid Input Data'
                return performance_metrics

            if isinstance(y, pd.DataFrame):
                if y.shape[1] > 1:
                    print(f"Warning: Multi-dimensional target detected ({y.shape[1]} dimensions). Using first column.")
                    y = y.iloc[:, 0]
                else:
                    y = y.iloc[:, 0]
            elif isinstance(y, np.ndarray):
                if y.ndim > 1:
                    print(f"Warning: Multi-dimensional target detected ({y.ndim} dimensions). Using first column.")
                    y = y[:, 0]

            y = np.asarray(y).ravel()

            if not np.issubdtype(y.dtype, np.number):
                label_encoder = LabelEncoder()
                try:
                    y = pd.Series(y).astype(str).fillna('MISSING')
                    y = label_encoder.fit_transform(y)
                except Exception as e:
                    print(f"Error encoding target column: {e}")
                    performance_metrics['evaluation_status'] = 'Failed - Encoding Error'
                    return performance_metrics

            if y.ndim != 1:
                print(f"Error: Target variable has incorrect dimensions after processing: {y.ndim}")
                performance_metrics['evaluation_status'] = 'Failed - Invalid Target Dimensions'
                return performance_metrics

            try:
                X_processed = self.preprocess_data(X, y)
            except Exception as e:
                print(f"Error in data preprocessing: {e}")
                performance_metrics['evaluation_status'] = 'Failed - Preprocessing Error'
                return performance_metrics

            try:
                X_standardized, X_pca, scaler, pca = self.process_features(X_processed)
                if X_standardized is None or X_pca is None:
                    print("Error in feature processing")
                    performance_metrics['evaluation_status'] = 'Failed - Feature Processing Error'
                    return performance_metrics
            except Exception as e:
                print(f"Error in feature processing: {e}")
                performance_metrics['evaluation_status'] = 'Failed - Feature Processing Error'
                return performance_metrics

            unique_values = len(np.unique(y))
            if unique_values <= 10 or unique_values / len(y) < 0.1:
                dataset_type = 'classification'
            else:
                performance_metrics['evaluation_status'] = 'Skipped - Regression Dataset'
                return performance_metrics

            performance_metrics['dataset_type'] = dataset_type

            if len(X_processed) > 25000:
                X_processed, y = self.smart_downsample(X_standardized, y)
            
            try:
                X_balanced, y_balanced = self.balance_data(X_processed, y)
            except Exception as e:
                print(f"Warning: Data balancing failed, using original data: {e}")
                X_balanced, y_balanced = X_processed, y
            
            try:
                complexity_metrics = self.compute_feature_complexity(X_processed)
                performance_metrics['complexity_features'].update({
                    'variance_explained_5_components': complexity_metrics.get('variance_explained_5_components'),
                    'avg_feature_correlation': complexity_metrics.get('avg_feature_correlation'),
                    'max_correlation': complexity_metrics.get('max_correlation')
                })
            except Exception as e:
                print(f"Error computing feature complexity: {e}")
                performance_metrics['complexity_features'] = {'error': str(e)}

            performance_metrics['complexity_features'].update({
                'intrinsic_dimensionality': self.compute_knn_intrinsic_dimensionality(
                    X_processed, k_min=5, k_max=50, max_samples=5000
                ),
                'pca_variance': self.compute_pca_variance(X_processed)[0] if self.compute_pca_variance(X_processed) else None,
                'outlier_fraction': self.compute_outlier_metrics(X_processed)
            })

            linear_models = {
                'pa_classifier': Pipeline([
                    ('scaler', StandardScaler()),
                    ('pa', PassiveAggressiveClassifier(
                        max_iter=2000,
                        tol=1e-3,
                        C=1.0,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=-1
                    ))
                ]),
                'ridge_classifier': Pipeline([
                    ('scaler', StandardScaler()),
                    ('ridge', RidgeClassifier(
                        random_state=42,
                        max_iter=2000,
                        solver='auto',
                        tol=0.001,
                        alpha=1.0,
                        class_weight='balanced',
                        positive=False
                    ))
                ]),
            }

            nonlinear_models = {
                'gnb_classifier': GaussianNB(var_smoothing=1e-5),
                'rbf_svc' : Pipeline([
                    ('rbf_feature', RBFSampler(gamma='scale', n_components=100, random_state=42)),
                    ('linear_svc', LinearSVC(class_weight='balanced', random_state=42))
                ])
            }
            
            tree_models = {
                'catboost' : CatBoostClassifier(
                    iterations=100,
                    depth=5,
                    learning_rate=0.1,
                    random_seed=42,
                    devices='0',
                    early_stopping_rounds=10,
                    verbose=False
                ),
                'extra_trees': ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=7,
                    random_state=42,
                    max_features='sqrt',
                    min_samples_leaf=10,
                    n_jobs=-1,
                    class_weight='balanced',
                    bootstrap=True,
                    max_samples=0.7
                )
            }
            
            scoring_metrics = {
                'f1_macro': 'f1_macro',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro'
            }

            class_counts = np.bincount(y_balanced)
            min_samples = np.min(class_counts)
            
            if min_samples < 3:
                valid_classes = np.where(class_counts >= 3)[0]
                tiny_classes = np.where((class_counts > 0) & (class_counts < 3))[0]
                
                if len(valid_classes) >= 2:
                    mask = np.isin(y_balanced, valid_classes)
                    X_balanced = X_balanced[mask]
                    y_balanced = y_balanced[mask]
                    
                    class_counts = np.bincount(y_balanced)
                    min_samples = np.min(class_counts)
                else:
                    classes_to_oversample = np.where(class_counts == 2)[0]
                    if len(classes_to_oversample) + len(valid_classes) >= 2:
                        X_list = [X_balanced]
                        y_list = [y_balanced]
                        
                        for cls in classes_to_oversample:
                            mask = y_balanced == cls
                            X_cls = X_balanced[mask]
                            y_cls = y_balanced[mask]
                            
                            synthetic_X = np.mean(X_cls, axis=0).reshape(1, -1)
                            synthetic_y = np.array([cls])
                            
                            X_list.append(synthetic_X)
                            y_list.append(synthetic_y)
                        
                        X_balanced = np.vstack(X_list)
                        y_balanced = np.concatenate(y_list)
                        
                        class_counts = np.bincount(y_balanced)
                        min_samples = np.min(class_counts)
                    else:
                        print("Warning: Unable to salvage dataset - insufficient samples even with oversampling")
                        performance_metrics['evaluation_status'] = 'Skipped - Insufficient samples per class'
                        return performance_metrics

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                
                for name, model in linear_models.items():
                    try:
                        start_time = time.time()
                        process = psutil.Process()
                        start_memory = process.memory_info().rss / 1024 / 1024
                        tracemalloc.start()
                        
                        skf = self.adaptive_stratified_split(X_balanced, y_balanced)
                        
                        cv_results = cross_validate(
                            model, X_balanced, y_balanced,
                            cv=skf,
                            scoring=scoring_metrics,
                            error_score='raise'
                        )
                        
                        cv_scores = {}
                        for metric_name in scoring_metrics.keys():
                            test_scores = cv_results[f'test_{metric_name}']
                            cv_scores[f'{metric_name}_mean'] = float(np.mean(test_scores))
                            cv_scores[f'{metric_name}_std'] = float(np.std(test_scores))

                        performance_metrics['metrics']['linear_models'][name] = cv_scores
                        
                        end_time = time.time()
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        #print(f"\nPerformance measurements for {name}:")
                        #print(f"Execution time: {end_time - start_time:.2f} seconds")
                        #print(f"Memory increase: {end_memory - start_memory:.2f} MB")
                        #print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
                        #print(f"Current memory: {current / 1024 / 1024:.2f} MB")
                        #print("-" * 50)
                        
                    except Exception as e:
                        print(f"Linear model evaluation error for {name}: {e}")
                        performance_metrics['metrics']['linear_models'][name] = None
                        
                        if tracemalloc.is_tracing():
                            tracemalloc.stop()

                for name, model in nonlinear_models.items():
                    try:
                        start_time = time.time()
                        process = psutil.Process()
                        start_memory = process.memory_info().rss / 1024 / 1024
                        tracemalloc.start()
                        
                        skf = self.adaptive_stratified_split(X_balanced, y_balanced)
                        
                        cv_results = cross_validate(
                            model, X_balanced, y_balanced,
                            cv=skf,
                            scoring=scoring_metrics,
                            error_score='raise'
                        )
                        
                        cv_scores = {}
                        for metric_name in scoring_metrics.keys():
                            test_scores = cv_results[f'test_{metric_name}']
                            cv_scores[f'{metric_name}_mean'] = float(np.mean(test_scores))
                            cv_scores[f'{metric_name}_std'] = float(np.std(test_scores))

                        performance_metrics['metrics']['nonlinear_models'][name] = cv_scores
                        
                        end_time = time.time()
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        #print(f"\nPerformance measurements for {name}:")
                        #print(f"Execution time: {end_time - start_time:.2f} seconds")
                        #print(f"Memory increase: {end_memory - start_memory:.2f} MB")
                        #print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
                        #print(f"Current memory: {current / 1024 / 1024:.2f} MB")
                        #print("-" * 50)
                        
                    except Exception as e:
                        print(f"Nonlinear model evaluation error for {name}: {e}")
                        performance_metrics['metrics']['nonlinear_models'][name] = None
                        
                        if tracemalloc.is_tracing():
                            tracemalloc.stop()

                for name, model in tree_models.items():
                    try:
                        start_time = time.time()
                        process = psutil.Process()
                        start_memory = process.memory_info().rss / 1024 / 1024
                        tracemalloc.start()
                        
                        skf = self.adaptive_stratified_split(X_balanced, y_balanced)

                        cv_results = cross_validate(
                            model, X_balanced, y_balanced,
                            cv=skf,
                            scoring=scoring_metrics,
                            error_score='raise'
                        )

                        cv_scores = {}
                        for metric_name in scoring_metrics.keys():
                            test_scores = cv_results[f'test_{metric_name}']
                            cv_scores[f'{metric_name}_mean'] = float(np.mean(test_scores))
                            cv_scores[f'{metric_name}_std'] = float(np.std(test_scores))

                        performance_metrics['metrics']['tree_models'][name] = cv_scores
                        
                        end_time = time.time()
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        #print(f"\nPerformance measurements for {name}:")
                        #print(f"Execution time: {end_time - start_time:.2f} seconds")
                        #print(f"Memory increase: {end_memory - start_memory:.2f} MB")
                        #print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
                        #print(f"Current memory: {current / 1024 / 1024:.2f} MB")
                        #print("-" * 50)
                        
                    except Exception as e:
                        print(f"Tree model evaluation error for {name}: {e}")
                        performance_metrics['metrics']['tree_models'][name] = None
                        
                        if tracemalloc.is_tracing():
                            tracemalloc.stop()
            
            performance_metrics['evaluation_status'] = 'Successful'
            return performance_metrics

        except Exception as e:
            print(f"Metadata extraction error: {e}")
            performance_metrics['evaluation_status'] = 'Failed - ' + str(e)
            return performance_metrics
        

    def collect_suitable_datasets(self, weights=None) -> pd.DataFrame:
        source_methods = {
            'openml': self.collect_openml_datasets,
            'kaggle': self.collect_kaggle_datasets,
            'local': self.process_single_dataset,
        }
        
        if weights is None:
            weights = {'openml': 0.5, 'kaggle': 0.5, 'local': 0}
        
        if not any(weights.values()):
            raise ValueError("At least one source must have a non-zero weight")
        
        active_sources = {k: v for k, v in weights.items() if v > 0}
        total_weight = sum(active_sources.values())
        normalized_weights = {source: weight / total_weight 
                            for source, weight in weights.items()}
        
        num_datasets_per_source = {
            source: math.ceil(self.num_datasets * weight)
            for source, weight in normalized_weights.items()
            if weight > 0
        }
        
        all_datasets = []
        
        for source, method in source_methods.items():
            if source not in num_datasets_per_source:
                continue
                
            try:
                count = num_datasets_per_source[source]
                print(f"Collecting {count} datasets from {source}...")
                source_datasets = method(count)
                
                for dataset in source_datasets:
                    try:
                        all_datasets.append(dataset)
                    except Exception as e:
                        print(f"Error processing dataset: {e}")
                        continue
                    
            except Exception as e:
                print(f"Error collecting datasets from {source}: {e}")

        df = pd.DataFrame(all_datasets)
        
        return df


if __name__ == '__main__':
    collector = SuperModelDatasetCollector(
        num_datasets=300,
    )
    
    weights = {'openml': 0.0, 'kaggle': 0.0, 'local': 0.0}
    datasets = collector.collect_suitable_datasets(weights)
    
    datasets.to_csv('super_model_raw_data.csv', index=False)