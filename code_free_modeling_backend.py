from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# Azure OpenAI (assuming it's already configured in your main app)
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeFreeModelingEngine:
    """
    Advanced Code-Free Modeling Engine similar to Dataiku/Alteryx
    """
    
    def __init__(self, azure_client=None):
        self.client = azure_client
        self.models = {
            'classification': {
                'Logistic Regression': {
                    'model': LogisticRegression,
                    'params': {'max_iter': 1000, 'random_state': 42},
                    'tuning_params': {
                        'C': [0.1, 1, 10],
                        'solver': ['liblinear', 'lbfgs']
                    }
                },
                'Random Forest': {
                    'model': RandomForestClassifier,
                    'params': {'n_estimators': 100, 'random_state': 42},
                    'tuning_params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingClassifier,
                    'params': {'n_estimators': 100, 'random_state': 42},
                    'tuning_params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                },
                'Support Vector Machine': {
                    'model': SVC,
                    'params': {'random_state': 42, 'probability': True},
                    'tuning_params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'Neural Network': {
                    'model': MLPClassifier,
                    'params': {'random_state': 42, 'max_iter': 500},
                    'tuning_params': {
                        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate': ['constant', 'adaptive']
                    }
                }
            },
            'regression': {
                'Linear Regression': {
                    'model': LinearRegression,
                    'params': {},
                    'tuning_params': {}
                },
                'Ridge Regression': {
                    'model': Ridge,
                    'params': {'random_state': 42},
                    'tuning_params': {
                        'alpha': [0.1, 1, 10, 100]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor,
                    'params': {'n_estimators': 100, 'random_state': 42},
                    'tuning_params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor,
                    'params': {'n_estimators': 100, 'random_state': 42},
                    'tuning_params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                },
                'Support Vector Regression': {
                    'model': SVR,
                    'params': {},
                    'tuning_params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                }
            }
        }
    
    def analyze_dataset(self, df):
        """Comprehensive dataset analysis for modeling recommendations"""
        analysis = {
            'shape': df.shape,
            'columns': [],
            'data_quality': {},
            'recommendations': []
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                'unique_count': int(df[col].nunique()),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_categorical': pd.api.types.is_object_dtype(df[col]),
                'suitable_for_target': False,
                'suitable_for_feature': True
            }
            
            # Determine suitability for target variable
            if col_info['is_numeric']:
                if col_info['unique_count'] <= 20 and col_info['unique_count'] >= 2:
                    col_info['suitable_for_target'] = True
                    col_info['problem_type'] = 'classification'
                elif col_info['unique_count'] > 20:
                    col_info['suitable_for_target'] = True
                    col_info['problem_type'] = 'regression'
            elif col_info['is_categorical']:
                if col_info['unique_count'] <= 10 and col_info['unique_count'] >= 2:
                    col_info['suitable_for_target'] = True
                    col_info['problem_type'] = 'classification'
            
            # Sample values
            try:
                sample_values = df[col].dropna().head(5).tolist()
                col_info['sample_values'] = [str(val) for val in sample_values]
            except:
                col_info['sample_values'] = []
            
            analysis['columns'].append(col_info)
        
        # Data quality assessment
        analysis['data_quality'] = {
            'total_missing': int(df.isnull().sum().sum()),
            'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': float((df.duplicated().sum() / len(df)) * 100)
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis):
        """Generate intelligent recommendations for modeling"""
        recommendations = []
        
        # Data quality recommendations
        if analysis['data_quality']['missing_percentage'] > 10:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'title': 'Handle Missing Values',
                'description': f"Dataset has {analysis['data_quality']['missing_percentage']:.1f}% missing values. Consider imputation or removal strategies."
            })
        
        if analysis['data_quality']['duplicate_percentage'] > 5:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'medium',
                'title': 'Remove Duplicates',
                'description': f"Found {analysis['data_quality']['duplicate_rows']} duplicate rows ({analysis['data_quality']['duplicate_percentage']:.1f}%)."
            })
        
        # Feature engineering recommendations
        numeric_cols = [col for col in analysis['columns'] if col['is_numeric']]
        categorical_cols = [col for col in analysis['columns'] if col['is_categorical']]
        
        if len(numeric_cols) > 1:
            recommendations.append({
                'type': 'feature_engineering',
                'priority': 'medium',
                'title': 'Feature Interactions',
                'description': 'Consider creating interaction features between numeric variables.'
            })
        
        if len(categorical_cols) > 0:
            high_cardinality_cols = [col for col in categorical_cols if col['unique_count'] > 50]
            if high_cardinality_cols:
                recommendations.append({
                    'type': 'preprocessing',
                    'priority': 'high',
                    'title': 'High Cardinality Features',
                    'description': f"Columns {[col['name'] for col in high_cardinality_cols]} have high cardinality. Consider grouping or encoding strategies."
                })
        
        return recommendations
    
    def preprocess_data(self, df, target_column, feature_columns, preprocessing_options):
        """Advanced data preprocessing pipeline"""
        # Create feature matrix and target vector
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        preprocessing_steps = []
        
        # Handle missing values
        if preprocessing_options.get('handle_missing', True):
            numeric_features = X.select_dtypes(include=['number']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            for col in numeric_features:
                if X[col].isnull().any():
                    if preprocessing_options.get('numeric_strategy') == 'median':
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        X[col].fillna(X[col].mean(), inplace=True)
            
            for col in categorical_features:
                if X[col].isnull().any():
                    X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
            
            preprocessing_steps.append("Handled missing values")
        
        # Feature scaling
        if preprocessing_options.get('scale_features', True):
            numeric_features = X.select_dtypes(include=['number']).columns
            if len(numeric_features) > 0:
                scaler = StandardScaler()
                X[numeric_features] = scaler.fit_transform(X[numeric_features])
                preprocessing_steps.append("Scaled numeric features")
        
        # Encode categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            for col in categorical_features:
                if X[col].nunique() <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            preprocessing_steps.append("Encoded categorical variables")
        
        # Feature engineering
        if preprocessing_options.get('create_interactions', False):
            numeric_cols = X.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                # Create interaction features for top numeric columns
                top_cols = numeric_cols[:3]
                for i, col1 in enumerate(top_cols):
                    for col2 in top_cols[i+1:]:
                        X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                preprocessing_steps.append("Created interaction features")
        
        # Encode target variable if needed
        label_encoder = None
        if pd.api.types.is_object_dtype(y):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        return X, y_encoded, label_encoder, preprocessing_steps
    
    def train_models(self, X, y, problem_type, selected_models, use_hyperparameter_tuning=False):
        """Train multiple models and compare performance"""
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if problem_type == 'classification' else None
        )
        
        for model_name in selected_models:
            if model_name not in self.models[problem_type]:
                continue
            
            model_config = self.models[problem_type][model_name]
            
            try:
                # Create model
                if use_hyperparameter_tuning and model_config['tuning_params']:
                    # Use GridSearchCV for hyperparameter tuning
                    base_model = model_config['model'](**model_config['params'])
                    grid_search = GridSearchCV(
                        base_model, 
                        model_config['tuning_params'],
                        cv=3,
                        scoring='accuracy' if problem_type == 'classification' else 'r2',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    model = model_config['model'](**model_config['params'])
                    model.fit(X_train, y_train)
                    best_params = model_config['params']
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba') and problem_type == 'classification':
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, problem_type)
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X, y, cv=5,
                    scoring='accuracy' if problem_type == 'classification' else 'r2'
                )
                
                # Feature importance
                feature_importance = self._get_feature_importance(model, X.columns)
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'cv_scores': {
                        'mean': float(cv_scores.mean()),
                        'std': float(cv_scores.std()),
                        'scores': cv_scores.tolist()
                    },
                    'best_params': best_params,
                    'feature_importance': feature_importance,
                    'predictions': {
                        'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                        'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
                    }
                }
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results, X_test, y_test
    
    def _calculate_metrics(self, y_test, y_pred, y_pred_proba, problem_type):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        if problem_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['f1_score'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                from sklearn.metrics import roc_auc_score
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
        
        else:  # regression
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['r2_score'] = float(r2_score(y_test, y_pred))
            metrics['mae'] = float(np.mean(np.abs(y_test - y_pred)))
        
        return metrics
    
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = [
                    {'feature': name, 'importance': float(imp)}
                    for name, imp in zip(feature_names, importances)
                ]
                return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]
                feature_importance = [
                    {'feature': name, 'importance': float(abs(coef_val))}
                    for name, coef_val in zip(feature_names, coef)
                ]
                return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
        
        return []
    
    def create_visualizations(self, results, problem_type):
        """Create comprehensive visualizations"""
        visualizations = {}
        
        # Model comparison chart
        model_names = list(results.keys())
        if problem_type == 'classification':
            scores = [results[name]['metrics']['accuracy'] for name in model_names if 'metrics' in results[name]]
            metric_name = 'Accuracy'
        else:
            scores = [results[name]['metrics']['r2_score'] for name in model_names if 'metrics' in results[name]]
            metric_name = 'R² Score'
        
        if scores:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(model_names, scores, color=['#3366FF', '#6366F1', '#8B5CF6', '#EC4899', '#F59E0B'])
            plt.title(f'Model Comparison - {metric_name}', fontsize=16, fontweight='bold')
            plt.ylabel(metric_name)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            visualizations['model_comparison'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
        
        # Feature importance for best model
        best_model_name = max(results.keys(), key=lambda x: scores[list(results.keys()).index(x)] if x in results and 'metrics' in results[x] else 0)
        if best_model_name in results and 'feature_importance' in results[best_model_name]:
            feature_imp = results[best_model_name]['feature_importance'][:10]  # Top 10 features
            
            if feature_imp:
                plt.figure(figsize=(10, 8))
                features = [f['feature'] for f in feature_imp]
                importances = [f['importance'] for f in feature_imp]
                
                plt.barh(features, importances, color='#3366FF')
                plt.title(f'Top 10 Feature Importances - {best_model_name}', fontsize=16, fontweight='bold')
                plt.xlabel('Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                visualizations['feature_importance'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
        
        return visualizations
    
    def generate_insights(self, results, problem_type, dataset_info):
        """Generate AI-powered insights about the modeling results"""
        insights = []
        
        try:
            if not self.client:
                return self._generate_fallback_insights(results, problem_type)
            
            # Prepare summary for LLM
            model_performance = {}
            for model_name, result in results.items():
                if 'metrics' in result:
                    if problem_type == 'classification':
                        model_performance[model_name] = {
                            'accuracy': result['metrics']['accuracy'],
                            'f1_score': result['metrics']['f1_score'],
                            'cv_mean': result['cv_scores']['mean']
                        }
                    else:
                        model_performance[model_name] = {
                            'r2_score': result['metrics']['r2_score'],
                            'rmse': result['metrics']['rmse'],
                            'cv_mean': result['cv_scores']['mean']
                        }
            
            summary = {
                'problem_type': problem_type,
                'dataset_shape': dataset_info['shape'],
                'model_performance': model_performance,
                'data_quality': dataset_info['data_quality']
            }
            
            prompt = f"""
            You are an expert data scientist analyzing automated machine learning results. Provide strategic insights and recommendations.
            
            Analysis Summary:
            {json.dumps(summary, indent=2)}
            
            Provide 5-7 insights covering:
            1. Model performance assessment and comparison
            2. Best model recommendation with justification
            3. Data quality impact on results
            4. Feature engineering opportunities
            5. Deployment readiness and considerations
            6. Business value and ROI potential
            7. Next steps for model improvement
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed analysis and recommendation",
                        "category": "Performance|Recommendation|Data Quality|Feature Engineering|Deployment|Business Value|Improvement",
                        "priority": "High|Medium|Low",
                        "actionable": true/false
                    }}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert ML engineer and data scientist. Provide strategic insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            insights = self._generate_fallback_insights(results, problem_type)
        
        return insights
    
    def _generate_fallback_insights(self, results, problem_type):
        """Generate fallback insights when AI is not available"""
        insights = []
        
        # Find best performing model
        if problem_type == 'classification':
            best_model = max(results.keys(), 
                           key=lambda x: results[x]['metrics']['accuracy'] if 'metrics' in results[x] else 0)
            best_score = results[best_model]['metrics']['accuracy'] if 'metrics' in results[best_model] else 0
            metric_name = 'accuracy'
        else:
            best_model = max(results.keys(), 
                           key=lambda x: results[x]['metrics']['r2_score'] if 'metrics' in results[x] else 0)
            best_score = results[best_model]['metrics']['r2_score'] if 'metrics' in results[best_model] else 0
            metric_name = 'R² score'
        
        # Performance assessment
        if best_score > 0.9:
            insights.append({
                'title': 'Excellent Model Performance',
                'description': f'{best_model} achieved outstanding {metric_name} of {best_score:.3f}, indicating excellent predictive capability.',
                'category': 'Performance',
                'priority': 'High',
                'actionable': True
            })
        elif best_score > 0.8:
            insights.append({
                'title': 'Good Model Performance',
                'description': f'{best_model} shows good performance with {metric_name} of {best_score:.3f}. Consider hyperparameter tuning for improvement.',
                'category': 'Performance',
                'priority': 'Medium',
                'actionable': True
            })
        else:
            insights.append({
                'title': 'Model Performance Needs Improvement',
                'description': f'Best model ({best_model}) achieved {metric_name} of {best_score:.3f}. Consider feature engineering or data quality improvements.',
                'category': 'Improvement',
                'priority': 'High',
                'actionable': True
            })
        
        # Model recommendation
        insights.append({
            'title': f'Recommended Model: {best_model}',
            'description': f'{best_model} is the top performer and recommended for deployment based on {metric_name} and cross-validation results.',
            'category': 'Recommendation',
            'priority': 'High',
            'actionable': True
        })
        
        # Deployment readiness
        insights.append({
            'title': 'Deployment Readiness',
            'description': 'Models are trained and ready for deployment. Consider A/B testing and monitoring setup.',
            'category': 'Deployment',
            'priority': 'Medium',
            'actionable': True
        })
        
        return insights

# Flask route integration (add this to your main app.py)
def add_code_free_modeling_routes(app, data_store, client):
    """Add Code-Free Modeling routes to the Flask app"""
    
    modeling_engine = CodeFreeModelingEngine(client)
    
    @app.route('/code-free-modeling')
    def code_free_modeling():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            if not session_id or session_id not in data_store:
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            return render_template('code-free-modeling.html')
        except Exception as e:
            logger.error(f"Error in code_free_modeling route: {str(e)}")
            return redirect(url_for('index'))
    
    @app.route('/api/code-free-modeling/analyze-dataset', methods=['GET'])
    def analyze_dataset_for_modeling():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Analyze dataset
            analysis = modeling_engine.analyze_dataset(df)
            analysis['filename'] = filename
            
            return jsonify(analysis)
        
        except Exception as e:
            logger.error(f"Error in analyze_dataset_for_modeling: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/api/code-free-modeling/train', methods=['POST'])
    def train_code_free_models():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400
            
            df = data_store[session_id]['df']
            
            # Extract parameters
            target_column = data.get('target_column')
            feature_columns = data.get('feature_columns', [])
            problem_type = data.get('problem_type')
            selected_models = data.get('selected_models', [])
            preprocessing_options = data.get('preprocessing_options', {})
            use_hyperparameter_tuning = data.get('use_hyperparameter_tuning', False)
            
            if not target_column or not feature_columns or not problem_type:
                return jsonify({'error': 'Missing required parameters'}), 400
            
            # Start modeling process
            start_time = time.time()
            
            # Preprocess data
            X, y, label_encoder, preprocessing_steps = modeling_engine.preprocess_data(
                df, target_column, feature_columns, preprocessing_options
            )
            
            # Train models
            results, X_test, y_test = modeling_engine.train_models(
                X, y, problem_type, selected_models, use_hyperparameter_tuning
            )
            
            # Create visualizations
            visualizations = modeling_engine.create_visualizations(results, problem_type)
            
            # Generate insights
            dataset_info = modeling_engine.analyze_dataset(df)
            insights = modeling_engine.generate_insights(results, problem_type, dataset_info)
            
            processing_time = round(time.time() - start_time, 2)
            
            # Store results
            modeling_id = str(uuid.uuid4())
            data_store[f"modeling_{modeling_id}"] = {
                'results': results,
                'dataset_info': dataset_info,
                'preprocessing_steps': preprocessing_steps,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Prepare response
            response = {
                'modeling_id': modeling_id,
                'results': {name: {k: v for k, v in result.items() if k != 'model'} 
                           for name, result in results.items()},
                'visualizations': visualizations,
                'insights': insights,
                'preprocessing_steps': preprocessing_steps,
                'processing_time': processing_time,
                'dataset_info': {
                    'shape': dataset_info['shape'],
                    'data_quality': dataset_info['data_quality']
                }
            }
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error in train_code_free_models: {str(e)}")
            return jsonify({'error': f'Training failed: {str(e)}'}), 500
    
    @app.route('/api/code-free-modeling/download', methods=['POST'])
    def download_modeling_results():
        try:
            data = request.json
            modeling_id = data.get('modeling_id')
            
            if not modeling_id:
                return jsonify({'error': 'Missing modeling_id'}), 400
            
            modeling_key = f"modeling_{modeling_id}"
            if modeling_key not in data_store:
                return jsonify({'error': 'Modeling results not found'}), 404
            
            # Create enhanced dataset with predictions
            modeling_data = data_store[modeling_key]
            session_id = modeling_data['session_id']
            original_df = data_store[session_id]['df']
            
            # Add model predictions to dataset
            enhanced_df = original_df.copy()
            enhanced_df['modeling_session'] = modeling_id
            enhanced_df['analysis_timestamp'] = modeling_data['timestamp']
            
            # Create temporary file
            temp_filename = f"code_free_modeling_results_{modeling_id[:8]}.csv"
            temp_path = os.path.join('static/temp', temp_filename)
            
            # Ensure temp directory exists
            import os
            os.makedirs('static/temp', exist_ok=True)
            
            # Save to CSV
            enhanced_df.to_csv(temp_path, index=False)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in download_modeling_results: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

# Add this to your main app.py file:
# add_code_free_modeling_routes(app, data_store, client)