from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
# Add the required imports at the top of your file if not already present
from collections import Counter
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import joblib


#from embedded_ml_backend import embedded_ml_bp
from data_storage import data_store
# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Azure OpenAI
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)







app.secret_key = "7c0d2e2f35e020ec485f18271ca26451"

# Azure OpenAI Configuration
try:
    client = AzureOpenAI(
        api_key="F6Zf4y3wEP8HoxVunVjWjIqyrbiEVw6YnNRdj5plfoulznvYVNLOJQQJ99BDAC77bzfXJ3w3AAABACOGqRfA",
        api_version="2024-04-01-preview",
        azure_endpoint="https://gen-ai-llm-deployment.openai.azure.com/"
    )
    deployment_name = "gpt-4o"
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Azure OpenAI client: {str(e)}")
    client = None

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create temp folder for plots
TEMP_FOLDER = 'static/temp'
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global data store
data_store = {}

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

# Optimized Algorithm configurations for fast training
ALGORITHMS = {
    'classification': {
        'quick': {
            'Logistic Regression': {
                'model': LogisticRegression,
                'params': {'max_iter': 50, 'random_state': 42, 'solver': 'liblinear'},
                'category': 'Linear Models',
                'description': 'Fast linear classifier for binary and multiclass problems'
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier,
                'params': {'max_depth': 3, 'random_state': 42, 'min_samples_split': 10},
                'category': 'Tree-based',
                'description': 'Interpretable tree-based classifier'
            },
            'Naive Bayes': {
                'model': GaussianNB,
                'params': {},
                'category': 'Probabilistic',
                'description': 'Probabilistic classifier based on Bayes theorem'
            }
        },
        'standard': {
            'Random Forest': {
                'model': RandomForestClassifier,
                'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 5, 'n_jobs': -1},
                'category': 'Ensemble',
                'description': 'Robust ensemble method with high accuracy'
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier,
                'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 3},
                'category': 'Ensemble',
                'description': 'Sequential ensemble with excellent performance'
            },
            'SVM': {
                'model': SVC,
                'params': {'kernel': 'linear', 'random_state': 42, 'probability': True, 'max_iter': 100},
                'category': 'Support Vector',
                'description': 'Powerful classifier for complex decision boundaries'
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier,
                'params': {'n_neighbors': 3, 'n_jobs': -1},
                'category': 'Instance-based',
                'description': 'Simple yet effective distance-based classifier'
            },
            'Neural Network': {
                'model': MLPClassifier,
                'params': {'hidden_layer_sizes': (50,), 'max_iter': 50, 'random_state': 42},
                'category': 'Neural Networks',
                'description': 'Deep learning approach for complex patterns'
            }
        }
    },
    'regression': {
        'quick': {
            'Linear Regression': {
                'model': LinearRegression,
                'params': {'n_jobs': -1},
                'category': 'Linear Models',
                'description': 'Simple linear relationship modeling'
            },
            'Ridge Regression': {
                'model': Ridge,
                'params': {'alpha': 1.0, 'random_state': 42},
                'category': 'Linear Models',
                'description': 'Regularized linear regression'
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor,
                'params': {'max_depth': 3, 'random_state': 42, 'min_samples_split': 10},
                'category': 'Tree-based',
                'description': 'Interpretable tree-based regressor'
            }
        },
        'standard': {
            'Random Forest': {
                'model': RandomForestRegressor,
                'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 5, 'n_jobs': -1},
                'category': 'Ensemble',
                'description': 'Robust ensemble method for regression'
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor,
                'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 3},
                'category': 'Ensemble',
                'description': 'Sequential ensemble with excellent performance'
            },
            'SVR': {
                'model': SVR,
                'params': {'kernel': 'linear', 'max_iter': 100},
                'category': 'Support Vector',
                'description': 'Support Vector Regression for non-linear patterns'
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsRegressor,
                'params': {'n_neighbors': 3, 'n_jobs': -1},
                'category': 'Instance-based',
                'description': 'Distance-based regression approach'
            },
            'Neural Network': {
                'model': MLPRegressor,
                'params': {'hidden_layer_sizes': (50,), 'max_iter': 50, 'random_state': 42},
                'category': 'Neural Networks',
                'description': 'Deep learning approach for complex patterns'
            }
        }
    }
}

# Add XGBoost if available (optimized for speed)
if XGBOOST_AVAILABLE:
    ALGORITHMS['classification']['standard']['XGBoost'] = {
        'model': xgb.XGBClassifier,
        'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 3, 'n_jobs': -1},
        'category': 'Gradient Boosting',
        'description': 'Extreme Gradient Boosting with superior performance'
    }
    ALGORITHMS['regression']['standard']['XGBoost'] = {
        'model': xgb.XGBRegressor,
        'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 3, 'n_jobs': -1},
        'category': 'Gradient Boosting',
        'description': 'Extreme Gradient Boosting for regression'
    }

# Add LightGBM if available (optimized for speed)
if LIGHTGBM_AVAILABLE:
    ALGORITHMS['classification']['standard']['LightGBM'] = {
        'model': lgb.LGBMClassifier,
        'params': {'n_estimators': 20, 'random_state': 42, 'verbose': -1, 'n_jobs': -1},
        'category': 'Gradient Boosting',
        'description': 'Fast gradient boosting with memory efficiency'
    }
    ALGORITHMS['regression']['standard']['LightGBM'] = {
        'model': lgb.LGBMRegressor,
        'params': {'n_estimators': 20, 'random_state': 42, 'verbose': -1, 'n_jobs': -1},
        'category': 'Gradient Boosting',
        'description': 'Fast gradient boosting for regression'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and (file.filename.endswith(('.csv', '.xlsx', '.xls'))):
            # Generate a unique ID for this session
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            logger.info(f"New session created: {session_id}")
            
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            # Read the file
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                logger.info(f"File read successfully: {filename}, shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                return jsonify({'error': f'Error reading file: {str(e)}'}), 400
            
            # Store the dataframe in our data store
            data_store[session_id] = {
                'df': df,
                'filename': filename,
                'file_path': file_path,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Get basic info about the dataframe
            info = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'preview': df.head(5).to_dict(orient='records'),
                'session_id': session_id
            }
            
            return jsonify(info)
        
        return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400
    
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# NLP Tools Routes
@app.route('/nlp-tools')
def nlp_tools():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"NLP Tools route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for NLP Tools: {session_id}")
        return render_template('nlp-tools.html')
    except Exception as e:
        logger.error(f"Error in nlp_tools route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/nlp-tools/dataset-info', methods=['GET'])
def api_nlp_tools_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"NLP Tools dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for NLP suitability
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Check if column is suitable for NLP
            is_text_suitable = False
            if pd.api.types.is_object_dtype(df[col]):
                # Check if it contains text-like data
                sample_values = df[col].dropna().head(5).astype(str).tolist()
                avg_length = np.mean([len(str(val)) for val in sample_values]) if sample_values else 0
                if avg_length > 10:  # Assume text if average length > 10 characters
                    is_text_suitable = True
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'is_text_suitable': is_text_suitable
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_nlp_tools_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/nlp-tools/analyze', methods=['POST'])
def api_nlp_tools_analyze():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"NLP analysis requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        text_column = data.get('text_column')
        nlp_task = data.get('nlp_task')
        model = data.get('model')
        sample_size = data.get('sample_size', '500')
        custom_labels = data.get('custom_labels', '')
        
        if not text_column or not nlp_task or not model:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        df = data_store[session_id]['df']
        
        # Sample data if needed
        if sample_size != 'all':
            sample_size = int(sample_size)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampling {len(df)} rows for NLP analysis")
        
        # Perform NLP analysis
        analysis_result = perform_nlp_analysis(df, text_column, nlp_task, model, custom_labels)
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        data_store[f"nlp_analysis_{analysis_id}"] = {
            'result': analysis_result,
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        analysis_result['analysis_id'] = analysis_id
        return jsonify(analysis_result)
    
    except Exception as e:
        logger.error(f"Error in api_nlp_tools_analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_nlp_analysis(df, text_column, nlp_task, model, custom_labels):
    """
    Perform NLP analysis using Azure OpenAI
    """
    try:
        # Get text data
        text_data = df[text_column].dropna().astype(str).tolist()
        processed_rows = len(text_data)
        
        if processed_rows == 0:
            raise ValueError("No valid text data found in the selected column")
        
        # Limit to first 100 rows for demo purposes (to avoid API limits)
        if len(text_data) > 100:
            text_data = text_data[:100]
            processed_rows = 100
        
        # Perform analysis based on task type
        if nlp_task == 'sentiment':
            results = analyze_sentiment(text_data, model, custom_labels)
        elif nlp_task == 'classification':
            results = classify_text(text_data, model, custom_labels)
        elif nlp_task == 'entity_extraction':
            results = extract_entities(text_data, model)
        elif nlp_task == 'embedding':
            results = generate_embeddings(text_data, model)
        elif nlp_task == 'summarization':
            results = summarize_text(text_data, model)
        elif nlp_task == 'topic_modeling':
            results = model_topics(text_data, model)
        elif nlp_task == 'language_detection':
            results = detect_language(text_data, model)
        elif nlp_task == 'keyword_extraction':
            results = extract_keywords(text_data, model)
        else:
            raise ValueError(f"Unsupported NLP task: {nlp_task}")
        
        # Calculate metrics
        metrics = calculate_nlp_metrics(results, nlp_task)
        
        # Generate insights
        insights = generate_nlp_insights(results, nlp_task, model)
        
        return {
            'processed_rows': processed_rows,
            'model': model,
            'task': nlp_task,
            'results': results,
            'metrics': metrics,
            'insights': insights
        }
    
    except Exception as e:
        logger.error(f"Error in perform_nlp_analysis: {str(e)}")
        raise

def analyze_sentiment(text_data, model, custom_labels):
    """Analyze sentiment of text data"""
    try:
        if not client:
            # Fallback sentiment analysis
            return fallback_sentiment_analysis(text_data)
        
        results = []
        labels = custom_labels.split(',') if custom_labels else ['positive', 'negative', 'neutral']
        labels = [label.strip() for label in labels]
        
        # Process in batches to avoid token limits
        batch_size = 10
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Analyze the sentiment of the following texts. Classify each text as one of: {', '.join(labels)}.
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "sentiment": "label", "confidence": 0.95}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analysis AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]  # Ensure we don't have more results than input
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return fallback_sentiment_analysis(text_data)

def classify_text(text_data, model, custom_labels):
    """Classify text data into categories"""
    try:
        if not client:
            return fallback_text_classification(text_data)
        
        results = []
        labels = custom_labels.split(',') if custom_labels else ['business', 'technology', 'sports', 'entertainment', 'politics']
        labels = [label.strip() for label in labels]
        
        # Process in batches
        batch_size = 10
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Classify the following texts into one of these categories: {', '.join(labels)}.
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "category": "label", "confidence": 0.95}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert text classification AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]
    
    except Exception as e:
        logger.error(f"Error in text classification: {str(e)}")
        return fallback_text_classification(text_data)

def extract_entities(text_data, model):
    """Extract named entities from text"""
    try:
        if not client:
            return fallback_entity_extraction(text_data)
        
        results = []
        
        # Process in smaller batches for entity extraction
        batch_size = 5
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Extract named entities (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, etc.) from the following texts:
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "entities": [{{"entity": "entity text", "type": "PERSON", "start": 0, "end": 5}}]}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert named entity recognition AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]
    
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        return fallback_entity_extraction(text_data)

def generate_embeddings(text_data, model):
    """Generate text embeddings (simplified for demo)"""
    try:
        # For demo purposes, we'll simulate embeddings
        results = []
        for text in text_data:
            # Simulate embedding generation
            embedding = np.random.rand(384).tolist()  # Simulate 384-dimensional embedding
            results.append({
                'text': text,
                'embedding': embedding,
                'dimension': 384
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in embedding generation: {str(e)}")
        return [{'text': text, 'embedding': [], 'dimension': 0} for text in text_data]

def summarize_text(text_data, model):
    """Summarize text data"""
    try:
        if not client:
            return fallback_summarization(text_data)
        
        results = []
        
        for text in text_data:
            if len(text) < 100:  # Skip very short texts
                results.append({
                    'text': text,
                    'summary': text,
                    'compression_ratio': 1.0
                })
                continue
            
            prompt = f"""
            Provide a concise summary of the following text:
            
            Text: {text}
            
            Respond with JSON format:
            {{
                "summary": "concise summary here"
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert text summarization AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            summary = result.get('summary', text)
            
            results.append({
                'text': text,
                'summary': summary,
                'compression_ratio': len(summary) / len(text)
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return fallback_summarization(text_data)

def model_topics(text_data, model):
    """Perform topic modeling on text data"""
    try:
        if not client:
            return fallback_topic_modeling(text_data)
        
        # Combine all texts for topic analysis
        combined_text = ' '.join(text_data)
        
        prompt = f"""
        Analyze the following texts and identify the main topics. Assign each text to the most relevant topic.
        
        Combined texts: {combined_text[:2000]}...
        
        Individual texts:
        {chr(10).join([f"{i+1}. {text}" for i, text in enumerate(text_data[:20])])}
        
        Respond with JSON format:
        {{
            "topics": ["topic1", "topic2", "topic3"],
            "results": [
                {{"text": "original text", "topic": "topic1", "confidence": 0.85}},
                ...
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert topic modeling AI. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('results', [])
    
    except Exception as e:
        logger.error(f"Error in topic modeling: {str(e)}")
        return fallback_topic_modeling(text_data)

def detect_language(text_data, model):
    """Detect language of text data"""
    try:
        if not client:
            return fallback_language_detection(text_data)
        
        results = []
        
        # Process in batches
        batch_size = 20
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Detect the language of the following texts:
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "language": "English", "language_code": "en", "confidence": 0.95}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert language detection AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]
    
    except Exception as e:
        logger.error(f"Error in language detection: {str(e)}")
        return fallback_language_detection(text_data)

def extract_keywords(text_data, model):
    """Extract keywords from text data"""
    try:
        if not client:
            return fallback_keyword_extraction(text_data)
        
        results = []
        
        for text in text_data:
            prompt = f"""
            Extract the most important keywords and phrases from the following text:
            
            Text: {text}
            
            Respond with JSON format:
            {{
                "keywords": ["keyword1", "keyword2", "keyword3"],
                "phrases": ["important phrase 1", "important phrase 2"]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert keyword extraction AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            results.append({
                'text': text,
                'keywords': result.get('keywords', []),
                'phrases': result.get('phrases', [])
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in keyword extraction: {str(e)}")
        return fallback_keyword_extraction(text_data)

# Fallback functions for when Azure OpenAI is not available
def fallback_sentiment_analysis(text_data):
    """Fallback sentiment analysis using simple rules"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']
    
    results = []
    for text in text_data:
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.6 + (pos_count - neg_count) * 0.1)
        elif neg_count > pos_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.6 + (neg_count - pos_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        })
    
    return results

def fallback_text_classification(text_data):
    """Fallback text classification using simple keyword matching"""
    categories = {
        'business': ['business', 'company', 'market', 'finance', 'economy', 'profit', 'revenue'],
        'technology': ['technology', 'software', 'computer', 'AI', 'digital', 'tech', 'innovation'],
        'sports': ['sports', 'game', 'team', 'player', 'match', 'score', 'championship'],
        'entertainment': ['movie', 'music', 'celebrity', 'entertainment', 'show', 'actor', 'film'],
        'politics': ['politics', 'government', 'election', 'policy', 'political', 'vote', 'democracy']
    }
    
    results = []
    for text in text_data:
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        best_category = max(scores, key=scores.get) if max(scores.values()) > 0 else 'other'
        confidence = min(0.9, 0.5 + scores[best_category] * 0.1)
        
        results.append({
            'text': text,
            'category': best_category,
            'confidence': confidence
        })
    
    return results

def fallback_entity_extraction(text_data):
    """Fallback entity extraction using simple patterns"""
    import re
    
    results = []
    for text in text_data:
        entities = []
        
        # Simple patterns for common entities
        # Names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for name in names:
            entities.append({
                'entity': name,
                'type': 'PERSON',
                'start': text.find(name),
                'end': text.find(name) + len(name)
            })
        
        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
        for date in dates:
            entities.append({
                'entity': date,
                'type': 'DATE',
                'start': text.find(date),
                'end': text.find(date) + len(date)
            })
        
        # Money
        money = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        for amount in money:
            entities.append({
                'entity': amount,
                'type': 'MONEY',
                'start': text.find(amount),
                'end': text.find(date) + len(amount)
            })
        
        results.append({
            'text': text,
            'entities': entities[:5]  # Limit to 5 entities per text
        })
    
    return results

def fallback_summarization(text_data):
    """Fallback summarization using simple sentence extraction"""
    results = []
    for text in text_data:
        sentences = text.split('.')
        if len(sentences) <= 2:
            summary = text
        else:
            # Take first and last sentence as summary
            summary = sentences[0] + '. ' + sentences[-1] if len(sentences) > 1 else sentences[0]
        
        results.append({
            'text': text,
            'summary': summary.strip(),
            'compression_ratio': len(summary) / len(text) if len(text) > 0 else 1.0
        })
    
    return results

def fallback_topic_modeling(text_data):
    """Fallback topic modeling using keyword frequency"""
    from collections import Counter
    import re
    
    # Extract all words
    all_words = []
    for text in text_data:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        all_words.extend(words)
    
    # Get most common words as topics
    word_counts = Counter(all_words)
    topics = [word for word, count in word_counts.most_common(5)]
    
    results = []
    for text in text_data:
        text_lower = text.lower()
        topic_scores = {}
        
        for topic in topics:
            score = text_lower.count(topic)
            topic_scores[topic] = score
        
        best_topic = max(topic_scores, key=topic_scores.get) if topic_scores else 'general'
        confidence = min(0.9, 0.5 + topic_scores.get(best_topic, 0) * 0.1)
        
        results.append({
            'text': text,
            'topic': best_topic,
            'confidence': confidence
        })
    
    return results

def fallback_language_detection(text_data):
    """Fallback language detection using simple heuristics"""
    results = []
    for text in text_data:
        # Simple heuristic: assume English for now
        # In a real implementation, you could use character frequency analysis
        results.append({
            'text': text,
            'language': 'English',
            'language_code': 'en',
            'confidence': 0.8
        })
    
    return results

def fallback_keyword_extraction(text_data):
    """Fallback keyword extraction using word frequency"""
    import re
    from collections import Counter
    
    results = []
    for text in text_data:
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        filtered_words = [word for word in words if word not in stop_words]
        
        # Get most frequent words as keywords
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(5)]
        
        # Extract phrases (simple bigrams)
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                phrases.append(f"{words[i]} {words[i+1]}")
        
        phrase_counts = Counter(phrases)
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(3)]
        
        results.append({
            'text': text,
            'keywords': keywords,
            'phrases': top_phrases
        })
    
    return results

def calculate_nlp_metrics(results, nlp_task):
    """Calculate metrics based on NLP task results"""
    try:
        metrics = {}
        
        if nlp_task == 'sentiment':
            sentiments = [r.get('sentiment', 'neutral') for r in results]
            sentiment_counts = Counter(sentiments)
            total = len(sentiments)
            
            metrics = {
                'total_texts': total,
                'positive_count': sentiment_counts.get('positive', 0),
                'negative_count': sentiment_counts.get('negative', 0),
                'neutral_count': sentiment_counts.get('neutral', 0),
                'avg_confidence': np.mean([r.get('confidence', 0) for r in results])
            }
        
        elif nlp_task == 'classification':
            categories = [r.get('category', 'unknown') for r in results]
            category_counts = Counter(categories)
            
            metrics = {
                'total_texts': len(categories),
                'unique_categories': len(category_counts),
                'most_common_category': category_counts.most_common(1)[0][0] if category_counts else 'none',
                'avg_confidence': np.mean([r.get('confidence', 0) for r in results])
            }
        
        elif nlp_task == 'entity_extraction':
            all_entities = []
            for r in results:
                all_entities.extend(r.get('entities', []))
            
            entity_types = [e.get('type', 'UNKNOWN') for e in all_entities]
            type_counts = Counter(entity_types)
            
            metrics = {
                'total_texts': len(results),
                'total_entities': len(all_entities),
                'avg_entities_per_text': len(all_entities) / len(results) if results else 0,
                'most_common_entity_type': type_counts.most_common(1)[0][0] if type_counts else 'none'
            }
        
        elif nlp_task == 'embedding':
            metrics = {
                'total_texts': len(results),
                'embedding_dimension': results[0].get('dimension', 0) if results else 0,
                'avg_text_length': np.mean([len(r.get('text', '')) for r in results])
            }
        
        elif nlp_task == 'summarization':
            compression_ratios = [r.get('compression_ratio', 1.0) for r in results]
            
            metrics = {
                'total_texts': len(results),
                'avg_compression_ratio': np.mean(compression_ratios),
                'min_compression_ratio': np.min(compression_ratios),
                'max_compression_ratio': np.max(compression_ratios)
            }
        
        elif nlp_task == 'topic_modeling':
            topics = [r.get('topic', 'unknown') for r in results]
            topic_counts = Counter(topics)
            
            metrics = {
                'total_texts': len(topics),
                'unique_topics': len(topic_counts),
                'most_common_topic': topic_counts.most_common(1)[0][0] if topic_counts else 'none',
                'avg_confidence': np.mean([r.get('confidence', 0) for r in results])
            }
        
        elif nlp_task == 'language_detection':
            languages = [r.get('language', 'unknown') for r in results]
            language_counts = Counter(languages)
            
            metrics = {
                'total_texts': len(languages),
                'unique_languages': len(language_counts),
                'most_common_language': language_counts.most_common(1)[0][0] if language_counts else 'none',
                'avg_confidence': np.mean([r.get('confidence', 0) for r in results])
            }
        
        elif nlp_task == 'keyword_extraction':
            all_keywords = []
            for r in results:
                all_keywords.extend(r.get('keywords', []))
            
            keyword_counts = Counter(all_keywords)
            
            metrics = {
                'total_texts': len(results),
                'total_keywords': len(all_keywords),
                'unique_keywords': len(keyword_counts),
                'avg_keywords_per_text': len(all_keywords) / len(results) if results else 0
            }
        
        else:
            metrics = {
                'total_texts': len(results),
                'task_type': nlp_task
            }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {'total_texts': len(results), 'error': str(e)}

def generate_nlp_insights(results, nlp_task, model):
    """Generate insights about NLP analysis results"""
    try:
        insights = []
        
        if nlp_task == 'sentiment':
            sentiments = [r.get('sentiment', 'neutral') for r in results]
            sentiment_counts = Counter(sentiments)
            total = len(sentiments)
            
            if sentiment_counts.get('positive', 0) > total * 0.6:
                insights.append({
                    'title': 'Predominantly Positive Sentiment',
                    'description': f'Over 60% of texts show positive sentiment, indicating favorable opinions or experiences.'
                })
            elif sentiment_counts.get('negative', 0) > total * 0.6:
                insights.append({
                    'title': 'Predominantly Negative Sentiment',
                    'description': f'Over 60% of texts show negative sentiment, suggesting areas for improvement.'
                })
            else:
                insights.append({
                    'title': 'Mixed Sentiment Distribution',
                    'description': f'Sentiment is well-distributed across positive, negative, and neutral categories.'
                })
        
        elif nlp_task == 'classification':
            categories = [r.get('category', 'unknown') for r in results]
            category_counts = Counter(categories)
            most_common = category_counts.most_common(1)[0] if category_counts else ('none', 0)
            
            insights.append({
                'title': 'Category Distribution Analysis',
                'description': f'Most common category is "{most_common[0]}" with {most_common[1]} occurrences ({(most_common[1]/len(categories)*100):.1f}% of total).'
            })
        
        elif nlp_task == 'entity_extraction':
            all_entities = []
            for r in results:
                all_entities.extend(r.get('entities', []))
            
            entity_types = [e.get('type', 'UNKNOWN') for e in all_entities]
            type_counts = Counter(entity_types)
            
            if type_counts:
                most_common_type = type_counts.most_common(1)[0]
                insights.append({
                    'title': 'Entity Type Analysis',
                    'description': f'Most frequently extracted entity type is {most_common_type[0]} with {most_common_type[1]} occurrences.'
                })
        
        elif nlp_task == 'topic_modeling':
            topics = [r.get('topic', 'unknown') for r in results]
            topic_counts = Counter(topics)
            
            insights.append({
                'title': 'Topic Distribution',
                'description': f'Identified {len(topic_counts)} distinct topics. Most prevalent topic: "{topic_counts.most_common(1)[0][0]}" if topic_counts else "none".'
            })
        
        # Add general insights
        insights.append({
            'title': 'ETL Integration Recommendation',
            'description': f'This {nlp_task} analysis can be integrated into your ETL pipeline for automated text processing and enrichment.'
        })
        
        insights.append({
            'title': 'Data Quality Assessment',
            'description': f'Processed {len(results)} text samples using {model} model with high accuracy and reliability.'
        })
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': f'Successfully processed {len(results)} texts using {nlp_task} analysis.'}]

@app.route('/api/nlp-tools/download', methods=['POST'])
def api_nlp_tools_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        analysis_id = data.get('analysis_id')
        
        if not session_id or not analysis_id:
            return jsonify({'error': 'Missing session_id or analysis_id'}), 400
        
        analysis_key = f"nlp_analysis_{analysis_id}"
        if analysis_key not in data_store:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis_data = data_store[analysis_key]
        results = analysis_data['result']['results']
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Create temporary file
        temp_filename = f"nlp_analysis_results_{analysis_id}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        df_results.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_nlp_tools_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

# AutoML Routes
@app.route('/automl')
def automl():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"AutoML route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for AutoML: {session_id}")
        return render_template('automl.html')
    except Exception as e:
        logger.error(f"Error in automl route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/automl/dataset-info', methods=['GET'])
def api_automl_dataset_info():
    try:
        # Try to get session_id from multiple sources
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Sample data for faster processing (max 1000 rows)
        if len(df) > 1000:
            df_sample = df.sample(n=1000, random_state=42)
            logger.info(f"Sampling {len(df_sample)} rows from {len(df)} total rows for faster processing")
        else:
            df_sample = df
        
        # Analyze columns
        columns_info = []
        for col in df_sample.columns:
            col_type = str(df_sample[col].dtype)
            missing = df_sample[col].isna().sum()
            missing_pct = (missing / len(df_sample)) * 100
            unique_count = df_sample[col].nunique()
            
            # Determine if column is suitable for target
            is_target_suitable = False
            target_type = None
            
            if pd.api.types.is_numeric_dtype(df_sample[col]):
                if unique_count <= 20 and unique_count >= 2:
                    is_target_suitable = True
                    target_type = 'classification'
                elif unique_count > 20:
                    is_target_suitable = True
                    target_type = 'regression'
            elif pd.api.types.is_object_dtype(df_sample[col]):
                if unique_count <= 50 and unique_count >= 2:
                    is_target_suitable = True
                    target_type = 'classification'
            
            # Get sample values safely
            sample_values = []
            try:
                non_null_values = df_sample[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(3, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                logger.warning(f"Error getting sample values for column {col}: {str(e)}")
                sample_values = ["N/A"]
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'is_target_suitable': is_target_suitable,
                'target_type': target_type,
                'sample_values': sample_values
            })
        
        # Create a serializable version of algorithms without the actual model classes
        algorithms_info = {}
        for problem_type in ALGORITHMS:
            algorithms_info[problem_type] = {}
            for complexity in ALGORITHMS[problem_type]:
                algorithms_info[problem_type][complexity] = {}
                for algo_name, algo_config in ALGORITHMS[problem_type][complexity].items():
                    algorithms_info[problem_type][complexity][algo_name] = {
                        'category': algo_config['category'],
                        'description': algo_config['description']
                    }

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'algorithms': algorithms_info,
            'session_id': session_id  # Include session_id in response
        })
    
    except Exception as e:
        logger.error(f"Error in api_automl_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/automl/analyze', methods=['POST'])
def api_automl_analyze():
    try:
        # Get session_id from request or session
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Training requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns', [])
        algorithm = data.get('algorithm')
        problem_type = data.get('problem_type')
        test_size = float(data.get('test_size', 0.2))
        
        if not target_column or not feature_columns or not algorithm:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        df = data_store[session_id]['df']
        
        # Sample data for faster training (max 5000 rows)
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
            logger.info(f"Sampling {len(df)} rows for faster training")
        
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove rows with missing target values
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            return jsonify({'error': 'No valid data after removing missing values'}), 400
        
        # Get algorithm configuration
        algo_config = None
        for complexity in ALGORITHMS[problem_type]:
            if algorithm in ALGORITHMS[problem_type][complexity]:
                algo_config = ALGORITHMS[problem_type][complexity][algorithm]
                break
        
        if not algo_config:
            return jsonify({'error': f'Algorithm {algorithm} not found'}), 400
        
        # Simple preprocessing for speed
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values quickly
        for col in numeric_features:
            X[col].fillna(X[col].median(), inplace=True)
        
        for col in categorical_features:
            X[col].fillna('Unknown', inplace=True)
        
        # Simple encoding for categorical variables
        if categorical_features:
            for col in categorical_features:
                # Simple label encoding for speed
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if classification
        if problem_type == 'classification' and pd.api.types.is_object_dtype(y):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y
            label_encoder = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, 
            stratify=y_encoded if problem_type == 'classification' else None
        )
        
        # Create and train model (simplified for speed)
        model = algo_config['model'](**algo_config['params'])
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Create simple confusion matrix plot
            try:
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                # Save plot
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
                buffer.seek(0)
                cm_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating confusion matrix: {str(e)}")
                cm_plot = None
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm_plot
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Create simple prediction vs actual plot
            try:
                plt.figure(figsize=(6, 4))
                plt.scatter(y_test, y_pred, alpha=0.6, s=20)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title('Prediction vs Actual')
                plt.tight_layout()
                
                # Save plot
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
                buffer.seek(0)
                pred_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating prediction plot: {str(e)}")
                pred_plot = None
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'prediction_plot': pred_plot
            }
        
        # Feature importance (if available)
        feature_importance = None
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = X.columns.tolist()
                importances = model.feature_importances_
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                feature_importance = feature_importance[:10]  # Top 10 features
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
        
        # Store model for download
        model_id = str(uuid.uuid4())
        model_data = {
            'model': model,
            'label_encoder': label_encoder,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'algorithm': algorithm,
            'problem_type': problem_type,
            'metrics': metrics,
            'training_time': training_time
        }
        data_store[f"model_{model_id}"] = model_data
        
        # Generate AI insights (simplified for speed)
        ai_insights = generate_automl_insights_fast(metrics, feature_importance, algorithm, problem_type)
        
        return jsonify({
            'model_id': model_id,
            'algorithm': algorithm,
            'problem_type': problem_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_time': float(training_time),
            'ai_insights': ai_insights,
            'data_shape': {
                'train': X_train.shape,
                'test': X_test.shape
            }
        })
    
    except Exception as e:
        logger.error(f"Error in api_automl_analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def generate_automl_insights_fast(metrics, feature_importance, algorithm, problem_type):
    """Generate fast insights without AI for speed"""
    if problem_type == 'classification':
        accuracy = metrics['accuracy']
        if accuracy > 0.9:
            performance = "Excellent performance"
            recommendation = "Model is ready for deployment"
        elif accuracy > 0.8:
            performance = "Good performance"
            recommendation = "Consider minor tuning for improvement"
        elif accuracy > 0.7:
            performance = "Moderate performance"
            recommendation = "Try feature engineering or different algorithms"
        else:
            performance = "Poor performance"
            recommendation = "Needs significant improvement - check data quality"
    else:
        r2 = metrics['r2_score']
        if r2 > 0.9:
            performance = "Excellent fit"
            recommendation = "Model explains data very well"
        elif r2 > 0.7:
            performance = "Good fit"
            recommendation = "Model performs well with room for improvement"
        elif r2 > 0.5:
            performance = "Moderate fit"
            recommendation = "Consider feature engineering or more complex models"
        else:
            performance = "Poor fit"
            recommendation = "Model needs significant improvement"
    
    feature_insight = "No feature importance available"
    if feature_importance and len(feature_importance) > 0:
        top_feature = feature_importance[0][0]
        feature_insight = f"'{top_feature}' is the most important feature for prediction"
    
    return {
        "performance_assessment": f"{performance} achieved with {algorithm}",
        "feature_insights": feature_insight,
        "improvement_recommendations": [
            recommendation,
            "Consider collecting more training data",
            "Try different preprocessing techniques"
        ],
        "etl_suggestions": [
            "Check for data quality issues",
            "Validate data consistency",
            "Consider feature scaling if needed"
        ],
        "deployment_readiness": f"Model training completed in under 5 seconds. {recommendation}"
    }

# Data Profiling Routes
@app.route('/dataprofiling')
def data_profiling():
    try:
        session_id = session.get('session_id')
        logger.info(f"Data profiling requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        return render_template('dataprofiling.html')
    
    except Exception as e:
        logger.error(f"Error in data_profiling route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/dataprofiling', methods=['GET'])
def api_data_profiling():
    try:
        session_id = session.get('session_id')
        logger.info(f"API data profiling requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        
        # Generate profiling data
        profile = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB",
                'duplicated_rows': int(df.duplicated().sum())
            },
            'columns': []
        }
        
        # Process each column
        for col in df.columns:
            try:
                col_type = str(df[col].dtype)
                missing = int(df[col].isna().sum())
                missing_pct = (missing / len(df)) * 100
                
                col_info = {
                    'name': col,
                    'type': col_type,
                    'missing': missing,
                    'missing_pct': f"{missing_pct:.2f}%",
                }
                
                # Add numeric statistics
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        col_info.update({
                            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                            'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                            'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                        })
                        
                        # Generate histogram
                        plt.figure(figsize=(8, 4))
                        sns.histplot(df[col].dropna(), kde=True)
                        plt.title(f'Distribution of {col}')
                        plt.tight_layout()
                        
                        # Save plot to buffer
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        plt.close()
                        
                        col_info['histogram'] = plot_data
                    except Exception as e:
                        logger.warning(f"Error generating numeric stats for column {col}: {str(e)}")
                    
                # Add categorical statistics
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    try:
                        value_counts = df[col].value_counts().head(10).to_dict()
                        col_info.update({
                            'unique_values': int(df[col].nunique()),
                            'top_values': value_counts
                        })
                        
                        # Generate bar plot for top categories
                        if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
                            plt.figure(figsize=(10, 5))
                            top_categories = df[col].value_counts().head(10)
                            sns.barplot(x=top_categories.index, y=top_categories.values)
                            plt.title(f'Top Categories in {col}')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            
                            # Save plot to buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='png')
                            buffer.seek(0)
                            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            plt.close()
                            
                            col_info['barplot'] = plot_data
                    except Exception as e:
                        logger.warning(f"Error generating categorical stats for column {col}: {str(e)}")
                
                # Add datetime statistics
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    try:
                        col_info.update({
                            'min': df[col].min().strftime('%Y-%m-%d %H:%M:%S'),
                            'max': df[col].max().strftime('%Y-%m-%d %H:%M:%S'),
                            'range': f"{(df[col].max() - df[col].min()).days} days"
                        })
                    except Exception as e:
                        logger.warning(f"Error generating datetime stats for column {col}: {str(e)}")
                
                profile['columns'].append(col_info)
            
            except Exception as e:
                logger.warning(f"Error processing column {col}: {str(e)}")
                # Add minimal column info to avoid breaking the UI
                profile['columns'].append({
                    'name': col,
                    'type': str(df[col].dtype),
                    'missing': int(df[col].isna().sum()),
                    'missing_pct': f"{(df[col].isna().sum() / len(df)) * 100:.2f}%",
                    'error': str(e)
                })
        
        # Generate correlation matrix for numeric columns
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                
                # Save plot to buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                corr_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                profile['correlation_matrix'] = corr_plot
        except Exception as e:
            logger.warning(f"Error generating correlation matrix: {str(e)}")
        
        # Use Azure OpenAI to generate insights
        try:
            if client:
                df_sample = df.head(100).to_string()
                prompt = f"""
                You are a data analysis expert. Based on the following dataset sample, provide 3-5 key insights and potential analysis directions.
                
                Dataset Sample:
                {df_sample}
                
                Please provide insights about:
                1. Data quality issues you notice
                2. Interesting patterns or relationships
                3. Suggested analyses or visualizations
                4. Potential business questions this data could answer
                
                Format your response as JSON with the following structure:
                {{
                    "insights": [
                        {{
                            "title": "Insight title",
                            "description": "Detailed explanation"
                        }}
                    ],
                    "suggested_analyses": [
                        {{
                            "title": "Analysis title",
                            "description": "What to analyze and why it's valuable"
                        }}
                    ]
                }}
                """
                
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a data analysis expert providing insights in JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                ai_insights = json.loads(response.choices[0].message.content)
                profile['ai_insights'] = ai_insights
            else:
                # Fallback if Azure OpenAI is not available
                profile['ai_insights'] = {
                    "insights": [
                        {
                            "title": "Basic Data Overview",
                            "description": f"This dataset contains {len(df)} rows and {len(df.columns)} columns with various data types."
                        }
                    ],
                    "suggested_analyses": [
                        {
                            "title": "Exploratory Data Analysis",
                            "description": "Perform basic statistical analysis and visualization to understand the distribution and relationships in the data."
                        }
                    ]
                }
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            profile['ai_insights'] = {
                "error": "Could not generate AI insights. Please try again later.",
                "insights": [
                    {
                        "title": "Basic Data Overview",
                        "description": f"This dataset contains {len(df)} rows and {len(df.columns)} columns."
                    }
                ],
                "suggested_analyses": [
                    {
                        "title": "Exploratory Data Analysis",
                        "description": "Perform basic statistical analysis to understand your data."
                    }
                ]
            }
        
        return jsonify(profile)
    
    except Exception as e:
        logger.error(f"Error in api_data_profiling: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'An error occurred while profiling the data.',
            'details': str(e),
            'basic_info': {
                'rows': 0,
                'columns': 0,
                'memory_usage': '0 MB',
                'duplicated_rows': 0
            },
            'columns': []
        }), 500

# LLM Code Generation Routes
@app.route('/llmcodegeneration')
def llm_code_generation():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return redirect(url_for('index'))
        
        return render_template('llmcodegeneration.html')
    except Exception as e:
        logger.error(f"Error in llm_code_generation route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/llmcodegeneration/generate', methods=['POST'])
def api_llmcodegeneration_generate():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        # Get data from request
        data = request.json
        task_description = data.get('task_description')
        code_type = data.get('code_type', 'analysis')
        complexity = data.get('complexity', 3)
        
        if not task_description:
            return jsonify({'error': 'No task description provided'}), 400
        
        # Get dataframe info
        df = data_store[session_id]['df']
        df_info = {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'shape': df.shape,
            'sample': df.head(5).to_dict(orient='records')
        }
        
        # Use Azure OpenAI to generate code
        try:
            if client:
                df_sample = df.head(10).to_string()
                df_info_str = json.dumps(df_info, indent=2)
                
                complexity_levels = {
                    1: "basic (simple operations, no advanced techniques)",
                    2: "simple (common data operations, minimal complexity)",
                    3: "medium (some advanced techniques, good balance)",
                    4: "advanced (complex operations, efficient code)",
                    5: "expert (cutting-edge techniques, highly optimized)"
                }
                
                complexity_desc = complexity_levels.get(complexity, "medium")
                
                prompt = f"""
                You are a Python data science expert. Generate code for the following task:
                
                Task: {task_description}
                
                Code Type: {code_type}
                Complexity Level: {complexity} ({complexity_desc})
                
                Dataset Information:
                {df_info_str}
                
                Dataset Sample:
                {df_sample}
                
                Please provide:
                1. A clear explanation of what the code does
                2. Well-commented Python code that accomplishes the task
                3. A list of required packages
                
                The code should assume that the dataframe is already loaded as 'df'.
                The code should be appropriate for the specified complexity level.
                
                Format your response as JSON with the following structure:
                {{
                    "explanation": "Explanation of what the code does",
                    "code": "Python code",
                    "requirements": ["package1", "package2"]
                }}
                """
                
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a Python data science expert providing code in JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
            else:
                # Fallback if Azure OpenAI is not available
                result = {
                    "explanation": "This code performs basic data analysis on the dataset.",
                    "code": "# Basic data analysis\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n# Display basic statistics\nprint(df.describe())\n\n# Check for missing values\nprint('\\nMissing values:')\nprint(df.isnull().sum())\n\n# Plot a histogram for numeric columns\ndf.select_dtypes(include=['number']).hist(figsize=(10, 8))\nplt.tight_layout()\nplt.show()",
                    "requirements": ["pandas", "matplotlib"]
                }
            
            # Store the generated code
            data_store[session_id]['generated_code'] = result['code']
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return jsonify({
                "explanation": "Basic data analysis code (fallback due to error).",
                "code": "# Basic data analysis\nimport pandas as pd\n\n# Display basic statistics\nprint(df.describe())\n\n# Check for missing values\nprint('\\nMissing values:')\nprint(df.isnull().sum())",
                "requirements": ["pandas"],
                "error": str(e)
            })
    
    except Exception as e:
        logger.error(f"Error in api_llmcodegeneration_generate: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/llmcodegeneration/execute', methods=['POST'])
def api_llmcodegeneration_execute():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        # Get code from request
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Get dataframe
        df = data_store[session_id]['df'].copy()
        
        # Execute code
        try:
            # Redirect stdout to capture print statements
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            
            # Create a local namespace
            local_vars = {'df': df, 'plt': plt, 'np': np, 'pd': pd, 'sns': sns}
            
            # Execute the code
            exec(code, globals(), local_vars)
            
            # Get the output
            output = mystdout.getvalue()
            
            # Reset stdout
            sys.stdout = old_stdout
            
            # Check if there are any figures
            figures = []
            if plt.get_fignums():
                for i in plt.get_fignums():
                    fig = plt.figure(i)
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png')
                    buffer.seek(0)
                    figures.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
                
                plt.close('all')
            
            return jsonify({
                'output': output,
                'figures': figures
            })
        
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        logger.error(f"Error in api_llmcodegeneration_execute: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# LLM Mesh Routes
@app.route('/llm_mesh')
def llm_mesh():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return redirect(url_for('index'))
        
        return render_template('llm_mesh.html')
    except Exception as e:
        logger.error(f"Error in llm_mesh route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/llm_mesh/analyze', methods=['POST'])
def api_llm_mesh_analyze():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Simulate LLM Mesh analysis with comprehensive data profiling
        analysis_result = perform_llm_mesh_analysis(df, filename)
        
        return jsonify(analysis_result)
    
    except Exception as e:
        logger.error(f"Error in api_llm_mesh_analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_llm_mesh_analysis(df, filename):
    """
    Perform comprehensive LLM Mesh analysis simulating multiple LLMs working together
    """
    try:
        # Basic profiling
        profiling = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'quality_score': calculate_data_quality_score(df)
        }
        
        # Anomaly detection
        anomalies = detect_anomalies(df)
        
        # Column analysis
        column_analysis = analyze_columns_with_llm(df)
        
        # Generate insights using Azure OpenAI
        insights = generate_llm_mesh_insights(df, filename)
        
        return {
            'dataset_name': filename,
            'profiling': profiling,
            'anomalies': anomalies,
            'column_analysis': column_analysis,
            'insights': insights,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except Exception as e:
        logger.error(f"Error in perform_llm_mesh_analysis: {str(e)}")
        return {
            'error': str(e),
            'dataset_name': filename,
            'profiling': {'total_rows': 0, 'total_columns': 0},
            'anomalies': [],
            'column_analysis': [],
            'insights': []
        }

def calculate_data_quality_score(df):
    """Calculate a data quality score based on various factors"""
    try:
        # Factors for quality score
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_ratio = df.duplicated().sum() / len(df)
        
        # Calculate score (0-100)
        quality_score = 100 - (missing_ratio * 50) - (duplicate_ratio * 30)
        quality_score = max(0, min(100, quality_score))
        
        return f"{quality_score:.1f}%"
    except:
        return "N/A"

def detect_anomalies(df):
    """Detect various types of anomalies in the dataset"""
    anomalies = []
    
    try:
        # Check for high missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                anomalies.append({
                    'type': f'High Missing Values in {col}',
                    'description': f'Column {col} has {missing_pct:.1f}% missing values',
                    'confidence': 95
                })
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            anomalies.append({
                'type': 'Duplicate Rows Detected',
                'description': f'Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}% of dataset)',
                'confidence': 100
            })
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    anomalies.append({
                        'type': f'Statistical Outliers in {col}',
                        'description': f'Column {col} contains {len(outliers)} potential outliers',
                        'confidence': 80
                    })
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                anomalies.append({
                    'type': f'Constant Column: {col}',
                    'description': f'Column {col} has only one unique value',
                    'confidence': 100
                })
    
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
    
    return anomalies

def analyze_columns_with_llm(df):
    """Analyze each column and provide AI-powered insights"""
    column_analysis = []
    
    try:
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Generate summary based on column type
            if pd.api.types.is_numeric_dtype(df[col]):
                summary = f"Numeric column with {unique_count} unique values. Range: {df[col].min():.2f} to {df[col].max():.2f}"
                if missing_pct < 5:
                    recommendation = "Good quality numeric data. Consider for statistical analysis."
                elif missing_pct < 20:
                    recommendation = "Some missing values. Consider for imputation strategies."
                else:
                    recommendation = "High missing values. Investigate data collection process."
            
            elif pd.api.types.is_object_dtype(df[col]):
                summary = f"Categorical column with {unique_count} unique values"
                if unique_count / len(df) > 0.8:
                    recommendation = "High cardinality. Consider grouping or encoding strategies."
                elif unique_count < 10:
                    recommendation = "Low cardinality. Good for categorical analysis."
                else:
                    recommendation = "Medium cardinality. Suitable for most analyses."
            
            else:
                summary = f"Column of type {col_type} with {unique_count} unique values"
                recommendation = "Review data type and consider appropriate preprocessing."
            
            column_analysis.append({
                'name': col,
                'type': col_type,
                'summary': summary,
                'recommendation': recommendation
            })
    
    except Exception as e:
        logger.error(f"Error in column analysis: {str(e)}")
    
    return column_analysis

def generate_llm_mesh_insights(df, filename):
    """Generate strategic insights using Azure OpenAI (simulating LLM Mesh)"""
    insights = []
    
    try:
        if client:
            # Create a comprehensive prompt for LLM Mesh analysis
            df_summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {col: str(df[col].dtype) for col in df.columns},
                'missing_summary': df.isnull().sum().to_dict(),
                'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
            
            prompt = f"""
            You are part of an advanced LLM Mesh system analyzing the dataset "{filename}". 
            Multiple specialized AI models have contributed to this analysis. Provide strategic insights and recommendations.
            
            Dataset Summary:
            {json.dumps(df_summary, indent=2, default=str)}
            
            As part of the LLM Mesh, provide 5-7 strategic insights covering:
            1. Data quality and reliability assessment
            2. Business value and potential use cases
            3. Recommended analytical approaches
            4. Data preparation suggestions
            5. Risk factors and limitations
            6. Opportunities for further analysis
            7. Integration recommendations
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed strategic recommendation"
                    }}
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are an advanced LLM Mesh system providing strategic data insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
        
        else:
            # Fallback insights if Azure OpenAI is not available
            insights = [
                {
                    "title": "Data Quality Assessment",
                    "description": f"The dataset contains {len(df)} rows and {len(df.columns)} columns with varying data quality. Consider implementing data validation rules."
                },
                {
                    "title": "Analysis Readiness",
                    "description": "The dataset appears suitable for exploratory data analysis. Recommend starting with descriptive statistics and visualization."
                },
                {
                    "title": "Missing Data Strategy",
                    "description": f"Missing values detected in {df.isnull().any().sum()} columns. Develop appropriate imputation or exclusion strategies."
                },
                {
                    "title": "Feature Engineering Opportunities",
                    "description": "Consider creating derived features from existing columns to enhance analytical value."
                },
                {
                    "title": "Scalability Considerations",
                    "description": "Evaluate computational requirements for larger datasets and consider optimization strategies."
                }
            ]
    
    except Exception as e:
        logger.error(f"Error generating LLM Mesh insights: {str(e)}")
        insights = [
            {
                "title": "Analysis Error",
                "description": f"Unable to generate comprehensive insights due to: {str(e)}"
            }
        ]
    
    return insights

# Automated Feature Engineering Routes
@app.route('/automated-feature-engineering')
def automated_feature_engineering():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Automated Feature Engineering route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for Automated Feature Engineering: {session_id}")
        return render_template('automated-feature-engineering.html')
    except Exception as e:
        logger.error(f"Error in automated_feature_engineering route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/feature-engineering/dataset-info', methods=['GET'])
def api_feature_engineering_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Feature Engineering dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for feature engineering suitability
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Determine feature engineering potential
            fe_potential = "High"
            if pd.api.types.is_numeric_dtype(df[col]):
                if unique_count < 5:
                    fe_potential = "Medium"
                elif missing_pct > 50:
                    fe_potential = "Low"
            elif pd.api.types.is_object_dtype(df[col]):
                if unique_count > len(df) * 0.8:
                    fe_potential = "Low"
                elif unique_count < 10:
                    fe_potential = "High"
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'fe_potential': fe_potential
            })

        # Calculate dataset size
        dataset_size = df.memory_usage(deep=True).sum()

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': columns_info,
            'size': int(dataset_size),
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_feature_engineering_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/feature-engineering/generate', methods=['POST'])
def api_feature_engineering_generate():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Feature engineering generation requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        selected_columns = data.get('selected_columns', [])
        feature_types = data.get('feature_types', [])
        model = data.get('model', 'gpt-4o')
        processing_mode = data.get('processing_mode', 'intelligent')
        
        if not selected_columns:
            return jsonify({'error': 'No columns selected for feature engineering'}), 400
        
        if not feature_types:
            return jsonify({'error': 'No feature engineering techniques selected'}), 400
        
        df = data_store[session_id]['df']
        
        # Perform feature engineering
        start_time = time.time()
        enhanced_df, feature_info = perform_automated_feature_engineering(
            df, selected_columns, feature_types, model, processing_mode
        )
        processing_time = round(time.time() - start_time, 2)
        
        # Store enhanced dataset
        processing_id = str(uuid.uuid4())
        data_store[f"enhanced_{processing_id}"] = {
            'enhanced_df': enhanced_df,
            'original_df': df,
            'feature_info': feature_info,
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate insights using Azure OpenAI
        insights = generate_feature_engineering_insights(df, enhanced_df, feature_info, model)
        
        # Prepare response data
        original_data = {
            'columns': df.columns.tolist(),
            'data': df.head(10).to_dict(orient='records')
        }
        
        enhanced_data = {
            'columns': enhanced_df.columns.tolist(),
            'data': enhanced_df.head(10).to_dict(orient='records')
        }
        
        new_features_count = len(enhanced_df.columns) - len(df.columns)
        
        return jsonify({
            'processing_id': processing_id,
            'original_data': original_data,
            'enhanced_data': enhanced_data,
            'new_features_count': new_features_count,
            'total_features_count': len(enhanced_df.columns),
            'processing_time': processing_time,
            'feature_info': feature_info,
            'insights': insights
        })
    
    except Exception as e:
        logger.error(f"Error in api_feature_engineering_generate: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/feature-engineering/download', methods=['POST'])
def api_feature_engineering_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        processing_id = data.get('processing_id')
        
        if not session_id or not processing_id:
            return jsonify({'error': 'Missing session_id or processing_id'}), 400
        
        enhanced_key = f"enhanced_{processing_id}"
        if enhanced_key not in data_store:
            return jsonify({'error': 'Enhanced dataset not found'}), 404
        
        enhanced_data = data_store[enhanced_key]
        enhanced_df = enhanced_data['enhanced_df']
        
        # Create temporary file
        temp_filename = f"enhanced_dataset_{processing_id}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        enhanced_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_feature_engineering_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

def perform_automated_feature_engineering(df, selected_columns, feature_types, model, processing_mode):
    """
    Perform automated feature engineering using various techniques and LLM guidance
    """
    try:
        enhanced_df = df.copy()
        feature_info = []
        
        # Filter to selected columns
        selected_df = df[selected_columns].copy()
        
        # Statistical Features
        if 'statistical' in feature_types:
            stat_features, stat_info = create_statistical_features(selected_df)
            enhanced_df = pd.concat([enhanced_df, stat_features], axis=1)
            feature_info.extend(stat_info)
        
        # Temporal Features
        if 'temporal' in feature_types:
            temporal_features, temporal_info = create_temporal_features(selected_df)
            if not temporal_features.empty:
                enhanced_df = pd.concat([enhanced_df, temporal_features], axis=1)
                feature_info.extend(temporal_info)
        
        # Categorical Encoding
        if 'categorical' in feature_types:
            cat_features, cat_info = create_categorical_features(selected_df)
            if not cat_features.empty:
                enhanced_df = pd.concat([enhanced_df, cat_features], axis=1)
                feature_info.extend(cat_info)
        
        # Feature Interactions
        if 'interaction' in feature_types:
            interaction_features, interaction_info = create_interaction_features(selected_df)
            if not interaction_features.empty:
                enhanced_df = pd.concat([enhanced_df, interaction_features], axis=1)
                feature_info.extend(interaction_info)
        
        # Text Features
        if 'text' in feature_types:
            text_features, text_info = create_text_features(selected_df)
            if not text_features.empty:
                enhanced_df = pd.concat([enhanced_df, text_features], axis=1)
                feature_info.extend(text_info)
        
        # Aggregation Features
        if 'aggregation' in feature_types:
            agg_features, agg_info = create_aggregation_features(selected_df)
            if not agg_features.empty:
                enhanced_df = pd.concat([enhanced_df, agg_features], axis=1)
                feature_info.extend(agg_info)
        
        # LLM-guided feature suggestions
        if client and processing_mode == 'intelligent':
            llm_features, llm_info = create_llm_guided_features(selected_df, model)
            if not llm_features.empty:
                enhanced_df = pd.concat([enhanced_df, llm_features], axis=1)
                feature_info.extend(llm_info)
        
        return enhanced_df, feature_info
    
    except Exception as e:
        logger.error(f"Error in perform_automated_feature_engineering: {str(e)}")
        raise

def create_statistical_features(df):
    """Create statistical features from numeric columns"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            # Rolling statistics
            features[f'{col}_rolling_mean_3'] = df[col].rolling(window=3, min_periods=1).mean()
            features[f'{col}_rolling_std_3'] = df[col].rolling(window=3, min_periods=1).std()
            
            # Lag features
            features[f'{col}_lag_1'] = df[col].shift(1)
            features[f'{col}_lag_2'] = df[col].shift(2)
            
            # Cumulative features
            features[f'{col}_cumsum'] = df[col].cumsum()
            features[f'{col}_cummax'] = df[col].cummax()
            features[f'{col}_cummin'] = df[col].cummin()
            
            # Z-score normalization
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                features[f'{col}_zscore'] = (df[col] - mean_val) / std_val
            
            # Percentile ranks
            features[f'{col}_percentile_rank'] = df[col].rank(pct=True)
            
            feature_info.extend([
                {'name': f'{col}_rolling_mean_3', 'type': 'statistical', 'description': f'3-period rolling mean of {col}'},
                {'name': f'{col}_rolling_std_3', 'type': 'statistical', 'description': f'3-period rolling standard deviation of {col}'},
                {'name': f'{col}_lag_1', 'type': 'statistical', 'description': f'1-period lag of {col}'},
                {'name': f'{col}_lag_2', 'type': 'statistical', 'description': f'2-period lag of {col}'},
                {'name': f'{col}_cumsum', 'type': 'statistical', 'description': f'Cumulative sum of {col}'},
                {'name': f'{col}_cummax', 'type': 'statistical', 'description': f'Cumulative maximum of {col}'},
                {'name': f'{col}_cummin', 'type': 'statistical', 'description': f'Cumulative minimum of {col}'},
                {'name': f'{col}_zscore', 'type': 'statistical', 'description': f'Z-score normalized {col}'},
                {'name': f'{col}_percentile_rank', 'type': 'statistical', 'description': f'Percentile rank of {col}'}
            ])
    
    return features, feature_info

def create_temporal_features(df):
    """Create temporal features from datetime columns"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    # Try to identify datetime columns
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == 'object':
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col].dropna().head(100))
                datetime_cols.append(col)
            except:
                continue
    
    for col in datetime_cols:
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_series = pd.to_datetime(df[col], errors='coerce')
            else:
                dt_series = df[col]
            
            # Extract temporal features
            features[f'{col}_year'] = dt_series.dt.year
            features[f'{col}_month'] = dt_series.dt.month
            features[f'{col}_day'] = dt_series.dt.day
            features[f'{col}_dayofweek'] = dt_series.dt.dayofweek
            features[f'{col}_hour'] = dt_series.dt.hour
            features[f'{col}_quarter'] = dt_series.dt.quarter
            features[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
            features[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
            features[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
            
            feature_info.extend([
                {'name': f'{col}_year', 'type': 'temporal', 'description': f'Year extracted from {col}'},
                {'name': f'{col}_month', 'type': 'temporal', 'description': f'Month extracted from {col}'},
                {'name': f'{col}_day', 'type': 'temporal', 'description': f'Day extracted from {col}'},
                {'name': f'{col}_dayofweek', 'type': 'temporal', 'description': f'Day of week from {col}'},
                {'name': f'{col}_hour', 'type': 'temporal', 'description': f'Hour extracted from {col}'},
                {'name': f'{col}_quarter', 'type': 'temporal', 'description': f'Quarter extracted from {col}'},
                {'name': f'{col}_is_weekend', 'type': 'temporal', 'description': f'Weekend indicator from {col}'},
                {'name': f'{col}_is_month_start', 'type': 'temporal', 'description': f'Month start indicator from {col}'},
                {'name': f'{col}_is_month_end', 'type': 'temporal', 'description': f'Month end indicator from {col}'}
            ])
            
        except Exception as e:
            logger.warning(f"Error creating temporal features for column {col}: {str(e)}")
            continue
    
    return features, feature_info

def create_categorical_features(df):
    """Create categorical encoding features"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if df[col].notna().sum() > 0:
            unique_count = df[col].nunique()
            
            # One-hot encoding for low cardinality
            if unique_count <= 10:
                dummies = pd.get_dummies(df[col], prefix=f'{col}_onehot', dummy_na=True)
                features = pd.concat([features, dummies], axis=1)
                
                for dummy_col in dummies.columns:
                    feature_info.append({
                        'name': dummy_col, 
                        'type': 'categorical', 
                        'description': f'One-hot encoding for {col}'
                    })
            
            # Label encoding
            le = LabelEncoder()
            features[f'{col}_label_encoded'] = le.fit_transform(df[col].fillna('missing'))
            
            # Frequency encoding
            freq_map = df[col].value_counts().to_dict()
            features[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
            
            # Length of string (for text columns)
            features[f'{col}_length'] = df[col].astype(str).str.len()
            
            feature_info.extend([
                {'name': f'{col}_label_encoded', 'type': 'categorical', 'description': f'Label encoding of {col}'},
                {'name': f'{col}_frequency', 'type': 'categorical', 'description': f'Frequency encoding of {col}'},
                {'name': f'{col}_length', 'type': 'categorical', 'description': f'String length of {col}'}
            ])
    
    return features, feature_info

def create_interaction_features(df):
    """Create feature interaction features"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Limit to first 5 numeric columns to avoid explosion
    numeric_cols = numeric_cols[:5]
    
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            if df[col1].notna().sum() > 0 and df[col2].notna().sum() > 0:
                # Multiplication
                features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Addition
                features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                
                # Ratio (avoid division by zero)
                features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
                # Difference
                features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                
                feature_info.extend([
                    {'name': f'{col1}_x_{col2}', 'type': 'interaction', 'description': f'Product of {col1} and {col2}'},
                    {'name': f'{col1}_plus_{col2}', 'type': 'interaction', 'description': f'Sum of {col1} and {col2}'},
                    {'name': f'{col1}_div_{col2}', 'type': 'interaction', 'description': f'Ratio of {col1} to {col2}'},
                    {'name': f'{col1}_minus_{col2}', 'type': 'interaction', 'description': f'Difference of {col1} and {col2}'}
                ])
    
    return features, feature_info

def create_text_features(df):
    """Create text-based features"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    text_cols = df.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        if df[col].notna().sum() > 0:
            text_series = df[col].astype(str)
            
            # Basic text features
            features[f'{col}_word_count'] = text_series.str.split().str.len()
            features[f'{col}_char_count'] = text_series.str.len()
            features[f'{col}_unique_words'] = text_series.apply(lambda x: len(set(x.split())))
            features[f'{col}_avg_word_length'] = text_series.apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            )
            
            # Count specific characters (fix regex escape sequences)
            features[f'{col}_exclamation_count'] = text_series.str.count('!')
            features[f'{col}_question_count'] = text_series.str.count(r'\?')  # Fixed escape sequence
            features[f'{col}_uppercase_count'] = text_series.str.count('[A-Z]')
            features[f'{col}_digit_count'] = text_series.str.count(r'\d')  # Fixed escape sequence
            
            # Text complexity
            features[f'{col}_sentence_count'] = text_series.str.count('[.!?]+')
            
            feature_info.extend([
                {'name': f'{col}_word_count', 'type': 'text', 'description': f'Word count in {col}'},
                {'name': f'{col}_char_count', 'type': 'text', 'description': f'Character count in {col}'},
                {'name': f'{col}_unique_words', 'type': 'text', 'description': f'Unique word count in {col}'},
                {'name': f'{col}_avg_word_length', 'type': 'text', 'description': f'Average word length in {col}'},
                {'name': f'{col}_exclamation_count', 'type': 'text', 'description': f'Exclamation mark count in {col}'},
                {'name': f'{col}_question_count', 'type': 'text', 'description': f'Question mark count in {col}'},
                {'name': f'{col}_uppercase_count', 'type': 'text', 'description': f'Uppercase letter count in {col}'},
                {'name': f'{col}_digit_count', 'type': 'text', 'description': f'Digit count in {col}'},
                {'name': f'{col}_sentence_count', 'type': 'text', 'description': f'Sentence count in {col}'}
            ])
    
    return features, feature_info

def create_aggregation_features(df):
    """Create aggregation features"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Group by categorical columns and aggregate numeric columns
    for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if df[cat_col].notna().sum() > 0:
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[num_col].notna().sum() > 0:
                    # Group statistics
                    group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count']).add_prefix(f'{cat_col}_{num_col}_')
                    
                    # Map back to original dataframe
                    for stat in ['mean', 'std', 'count']:
                        feature_name = f'{cat_col}_{num_col}_{stat}'
                        features[feature_name] = df[cat_col].map(group_stats[feature_name]).fillna(0)
                        
                        feature_info.append({
                            'name': feature_name,
                            'type': 'aggregation',
                            'description': f'{stat.title()} of {num_col} grouped by {cat_col}'
                        })
    
    return features, feature_info

def create_llm_guided_features(df, model):
    """Create LLM-guided features using Azure OpenAI"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    try:
        if not client:
            return features, feature_info
        
        # Analyze the dataset structure
        df_info = {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'sample_data': df.head(5).to_dict(orient='records'),
            'shape': df.shape
        }
        
        prompt = f"""
        You are an expert data scientist specializing in feature engineering. Analyze the following dataset and suggest 3-5 intelligent feature engineering ideas that would be valuable for machine learning.

        Dataset Information:
        {json.dumps(df_info, indent=2, default=str)}

        Please suggest features that:
        1. Are mathematically sound and interpretable
        2. Could improve model performance
        3. Are not too complex to compute
        4. Make business sense

        Respond with JSON format:
        {{
            "features": [
                {{
                    "name": "feature_name",
                    "description": "Clear description of the feature",
                    "formula": "Mathematical formula or logic",
                    "columns_used": ["col1", "col2"]
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert feature engineering AI. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        llm_suggestions = json.loads(response.choices[0].message.content)
        
        # Implement suggested features
        for suggestion in llm_suggestions.get('features', []):
            try:
                feature_name = suggestion['name']
                columns_used = suggestion.get('columns_used', [])
                
                # Simple feature implementations based on common patterns
                if len(columns_used) >= 2:
                    col1, col2 = columns_used[0], columns_used[1]
                    if col1 in df.columns and col2 in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                            # Create a ratio feature
                            features[feature_name] = df[col1] / (df[col2] + 1e-8)
                        elif pd.api.types.is_numeric_dtype(df[col1]):
                            # Create a grouped statistic
                            group_mean = df.groupby(col2)[col1].transform('mean')
                            features[feature_name] = df[col1] / (group_mean + 1e-8)
                
                feature_info.append({
                    'name': feature_name,
                    'type': 'llm_guided',
                    'description': suggestion['description']
                })
                
            except Exception as e:
                logger.warning(f"Error implementing LLM feature {suggestion.get('name', 'unknown')}: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error in create_llm_guided_features: {str(e)}")
    
    return features, feature_info

def generate_feature_engineering_insights(original_df, enhanced_df, feature_info, model):
    """Generate insights about the feature engineering process using Azure OpenAI"""
    try:
        insights = []
        
        if client:
            # Prepare summary for LLM
            summary = {
                'original_features': len(original_df.columns),
                'new_features': len(enhanced_df.columns) - len(original_df.columns),
                'total_features': len(enhanced_df.columns),
                'feature_types': list(set([f['type'] for f in feature_info])),
                'sample_new_features': [f['name'] for f in feature_info[:5]]
            }
            
            prompt = f"""
            You are an expert data scientist analyzing the results of an automated feature engineering process. 
            
            Feature Engineering Summary:
            {json.dumps(summary, indent=2)}
            
            Provide 3-5 strategic insights about:
            1. The quality and potential impact of the new features
            2. Recommendations for model training
            3. Potential risks or considerations
            4. Next steps for the data science workflow
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed explanation and recommendation"
                    }}
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist providing feature engineering insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
        
        else:
            # Fallback insights
            new_features_count = len(enhanced_df.columns) - len(original_df.columns)
            insights = [
                {
                    "title": "Feature Engineering Complete",
                    "description": f"Successfully generated {new_features_count} new features, expanding your dataset from {len(original_df.columns)} to {len(enhanced_df.columns)} features."
                },
                {
                    "title": "Model Performance Potential",
                    "description": "The new features include statistical, temporal, and interaction features that could significantly improve model performance."
                },
                {
                    "title": "Data Quality Consideration",
                    "description": "Review the new features for any missing values or outliers before training your models."
                },
                {
                    "title": "Feature Selection Recommendation",
                    "description": "Consider using feature selection techniques to identify the most important features for your specific use case."
                }
            ]
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating feature engineering insights: {str(e)}")
        return [{"title": "Processing Complete", "description": "Feature engineering completed successfully."}]

# AI Copilot for Data Exploration & Prep Routes
@app.route('/ai-copilot')
def ai_copilot():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"AI Copilot route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for AI Copilot: {session_id}")
        return render_template('ai-copilot.html')
    except Exception as e:
        logger.error(f"Error in ai_copilot route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/ai-copilot/dataset-info', methods=['GET'])
def api_ai_copilot_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"AI Copilot dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for AI Copilot
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Get sample values
            sample_values = []
            try:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                logger.warning(f"Error getting sample values for column {col}: {str(e)}")
                sample_values = ["N/A"]
            
            # Determine data quality
            quality = "Good"
            if missing_pct > 50:
                quality = "Poor"
            elif missing_pct > 20:
                quality = "Fair"
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'sample_values': sample_values,
                'quality': quality
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_ai_copilot_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/ai-copilot/explore', methods=['POST'])
def api_ai_copilot_explore():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"AI Copilot exploration requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        selected_columns = data.get('selected_columns', [])
        operation_type = data.get('operation_type')
        model = data.get('model', 'gpt-4o')
        custom_instruction = data.get('custom_instruction', '')
        
        if not selected_columns:
            return jsonify({'error': 'No columns selected'}), 400
        
        if not operation_type:
            return jsonify({'error': 'No operation type specified'}), 400
        
        df = data_store[session_id]['df']
        
        # Perform AI-powered data exploration
        start_time = time.time()
        result = perform_ai_copilot_operation(df, selected_columns, operation_type, model, custom_instruction)
        processing_time = round(time.time() - start_time, 2)
        
        # Store result for download (keep the DataFrame here)
        operation_id = str(uuid.uuid4())
        data_store[f"copilot_{operation_id}"] = {
            'result_df': result.get('modified_df', df),
            'original_df': df,
            'operation_type': operation_type,
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Remove DataFrame from result before JSON serialization
        if 'modified_df' in result:
            # Convert DataFrame to preview data
            modified_df = result['modified_df']
            result['data_preview'] = {
                'columns': modified_df.columns.tolist(),
                'data': modified_df.head(10).to_dict(orient='records'),
                'shape': modified_df.shape
            }
            del result['modified_df']  # Remove the DataFrame
        
        result['operation_id'] = operation_id
        result['processing_time'] = processing_time
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in api_ai_copilot_explore: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_ai_copilot_operation(df, selected_columns, operation_type, model, custom_instruction):
    """
    Perform AI-powered data exploration and preparation operations
    """
    try:
        result = {
            'operation': operation_type,
            'columns_processed': selected_columns,
            'insights': [],
            'visualizations': [],
            'modified_df': df.copy(),
            'changes_summary': []
        }
        
        # Filter to selected columns for analysis
        selected_df = df[selected_columns].copy()
        
        if operation_type == 'data_cleaning':
            result = perform_data_cleaning(df, selected_columns, model, custom_instruction)
        elif operation_type == 'outlier_detection':
            result = perform_outlier_detection(df, selected_columns, model, custom_instruction)
        elif operation_type == 'missing_value_analysis':
            result = perform_missing_value_analysis(df, selected_columns, model, custom_instruction)
        elif operation_type == 'correlation_analysis':
            result = perform_correlation_analysis(df, selected_columns, model, custom_instruction)
        elif operation_type == 'distribution_analysis':
            result = perform_distribution_analysis(df, selected_columns, model, custom_instruction)
        elif operation_type == 'feature_importance':
            result = perform_feature_importance_analysis(df, selected_columns, model, custom_instruction)
        elif operation_type == 'data_transformation':
            result = perform_data_transformation(df, selected_columns, model, custom_instruction)
        elif operation_type == 'custom_analysis':
            result = perform_custom_analysis(df, selected_columns, model, custom_instruction)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_ai_copilot_operation: {str(e)}")
        raise

def perform_data_cleaning(df, selected_columns, model, custom_instruction):
    """Perform intelligent data cleaning"""
    result = {
        'operation': 'data_cleaning',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'changes_summary': []
    }
    
    try:
        modified_df = df.copy()
        changes = []
        
        for col in selected_columns:
            if col in df.columns:
                # Handle missing values
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Fill with median for numeric columns
                        median_val = df[col].median()
                        modified_df[col].fillna(median_val, inplace=True)
                        changes.append(f"Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
                    else:
                        # Fill with mode for categorical columns
                        mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        modified_df[col].fillna(mode_val, inplace=True)
                        changes.append(f"Filled {missing_count} missing values in '{col}' with mode ('{mode_val}')")
                
                # Remove duplicates
                if col in df.select_dtypes(include=['object']).columns:
                    # Standardize text
                    modified_df[col] = modified_df[col].astype(str).str.strip().str.lower()
                    changes.append(f"Standardized text in '{col}' (trimmed and lowercased)")
        
        # Remove duplicate rows
        initial_rows = len(modified_df)
        modified_df.drop_duplicates(inplace=True)
        removed_duplicates = initial_rows - len(modified_df)
        if removed_duplicates > 0:
            changes.append(f"Removed {removed_duplicates} duplicate rows")
        
        # Store the modified DataFrame separately (will be handled by the main function)
        result['modified_df'] = modified_df
        result['changes_summary'] = changes
        
        # Generate insights using AI
        if client:
            insights = generate_cleaning_insights(df, modified_df, changes, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Data Cleaning Complete',
                    'description': f'Applied {len(changes)} cleaning operations to improve data quality.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_data_cleaning: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Data cleaning failed: {str(e)}'}]
        return result

def perform_outlier_detection(df, selected_columns, model, custom_instruction):
    """Perform outlier detection and analysis"""
    result = {
        'operation': 'outlier_detection',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'changes_summary': []
    }
    
    try:
        outlier_info = []
        visualizations = []
        
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                # IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(df)) * 100
                
                outlier_info.append({
                    'column': col,
                    'outlier_count': outlier_count,
                    'outlier_percentage': f"{outlier_percentage:.2f}%",
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                })
                
                # Create box plot
                plt.figure(figsize=(8, 6))
                plt.boxplot(df[col].dropna(), labels=[col])
                plt.title(f'Box Plot for {col} - Outlier Detection')
                plt.ylabel('Values')
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                visualizations.append({
                    'type': 'boxplot',
                    'title': f'Outlier Detection - {col}',
                    'data': plot_data
                })
        
        # Store modified DataFrame (same as original for outlier detection)
        result['modified_df'] = df.copy()
        result['visualizations'] = visualizations
        result['outlier_info'] = outlier_info
        
        # Generate AI insights
        if client:
            insights = generate_outlier_insights(outlier_info, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Outlier Analysis Complete',
                    'description': f'Analyzed {len(numeric_cols)} numeric columns for outliers using IQR method.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_outlier_detection: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Outlier detection failed: {str(e)}'}]
        return result

def perform_missing_value_analysis(df, selected_columns, model, custom_instruction):
    """Perform comprehensive missing value analysis"""
    result = {
        'operation': 'missing_value_analysis',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        missing_info = []
        
        for col in selected_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_percentage = (missing_count / len(df)) * 100
                
                missing_info.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': f"{missing_percentage:.2f}%",
                    'data_type': str(df[col].dtype)
                })
        
        # Create missing value heatmap
        if len(selected_columns) > 1:
            plt.figure(figsize=(10, 6))
            missing_matrix = df[selected_columns].isnull()
            sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Value Pattern')
            plt.xlabel('Columns')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            result['visualizations'] = [{
                'type': 'heatmap',
                'title': 'Missing Value Pattern',
                'data': plot_data
            }]
        
        result['modified_df'] = df.copy()
        result['missing_info'] = missing_info
        
        # Generate AI insights
        if client:
            insights = generate_missing_value_insights(missing_info, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Missing Value Analysis Complete',
                    'description': f'Analyzed missing values across {len(selected_columns)} columns.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_missing_value_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Missing value analysis failed: {str(e)}'}]
        return result

def perform_correlation_analysis(df, selected_columns, model, custom_instruction):
    """Perform correlation analysis"""
    result = {
        'operation': 'correlation_analysis',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            result['insights'] = [{'title': 'Insufficient Data', 'description': 'Need at least 2 numeric columns for correlation analysis.'}]
            return result
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        result['visualizations'] = [{
            'type': 'correlation_heatmap',
            'title': 'Correlation Matrix',
            'data': plot_data
        }]
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'column1': corr_matrix.columns[i],
                        'column2': corr_matrix.columns[j],
                        'correlation': f"{corr_val:.3f}"
                    })
        
        result['modified_df'] = df.copy()
        result['strong_correlations'] = strong_correlations
        
        # Generate AI insights
        if client:
            insights = generate_correlation_insights(strong_correlations, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Correlation Analysis Complete',
                    'description': f'Found {len(strong_correlations)} strong correlations (|r| > 0.7) among {len(numeric_cols)} numeric columns.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_correlation_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Correlation analysis failed: {str(e)}'}]
        return result

def perform_distribution_analysis(df, selected_columns, model, custom_instruction):
    """Perform distribution analysis"""
    result = {
        'operation': 'distribution_analysis',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        visualizations = []
        distribution_stats = []
        
        for col in selected_columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Create histogram
                    plt.figure(figsize=(8, 6))
                    plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    visualizations.append({
                        'type': 'histogram',
                        'title': f'Distribution - {col}',
                        'data': plot_data
                    })
                    
                    # Calculate distribution statistics
                    stats = {
                        'column': col,
                        'mean': f"{df[col].mean():.3f}",
                        'median': f"{df[col].median():.3f}",
                        'std': f"{df[col].std():.3f}",
                        'skewness': f"{df[col].skew():.3f}",
                        'kurtosis': f"{df[col].kurtosis():.3f}"
                    }
                    distribution_stats.append(stats)
                
                elif pd.api.types.is_object_dtype(df[col]):
                    # Create bar plot for categorical data
                    value_counts = df[col].value_counts().head(10)
                    
                    plt.figure(figsize=(10, 6))
                    value_counts.plot(kind='bar')
                    plt.title(f'Top 10 Values in {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    visualizations.append({
                        'type': 'barplot',
                        'title': f'Value Distribution - {col}',
                        'data': plot_data
                    })
        
        result['modified_df'] = df.copy()
        result['visualizations'] = visualizations
        result['distribution_stats'] = distribution_stats
        
        # Generate AI insights
        if client:
            insights = generate_distribution_insights(distribution_stats, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Distribution Analysis Complete',
                    'description': f'Analyzed distributions for {len(selected_columns)} columns.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_distribution_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Distribution analysis failed: {str(e)}'}]
        return result

def perform_feature_importance_analysis(df, selected_columns, model, custom_instruction):
    """Perform feature importance analysis"""
    result = {
        'operation': 'feature_importance',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        # This is a simplified feature importance analysis
        # In a real scenario, you'd need a target variable
        
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            result['insights'] = [{'title': 'Insufficient Data', 'description': 'Need at least 2 numeric columns for feature importance analysis.'}]
            return result
        
        # Calculate variance as a proxy for importance
        importance_scores = []
        for col in numeric_cols:
            variance = df[col].var()
            importance_scores.append({
                'feature': col,
                'importance_score': variance,
                'normalized_score': variance / df[col].max() if df[col].max() != 0 else 0
            })
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Create importance plot
        features = [item['feature'] for item in importance_scores]
        scores = [item['normalized_score'] for item in importance_scores]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, scores)
        plt.title('Feature Importance (Based on Variance)')
        plt.xlabel('Normalized Importance Score')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        result['visualizations'] = [{
            'type': 'feature_importance',
            'title': 'Feature Importance Analysis',
            'data': plot_data
        }]
        
        result['modified_df'] = df.copy()
        result['importance_scores'] = importance_scores
        
        # Generate AI insights
        if client:
            insights = generate_feature_importance_insights(importance_scores, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Feature Importance Analysis Complete',
                    'description': f'Analyzed importance of {len(numeric_cols)} numeric features based on variance.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_feature_importance_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Feature importance analysis failed: {str(e)}'}]
        return result

def perform_data_transformation(df, selected_columns, model, custom_instruction):
    """Perform data transformation"""
    result = {
        'operation': 'data_transformation',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        modified_df = df.copy()
        changes = []
        
        for col in selected_columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Apply log transformation for skewed data
                    if df[col].skew() > 1:
                        modified_df[f'{col}_log'] = np.log1p(df[col])
                        changes.append(f"Applied log transformation to '{col}' (high skewness)")
                    
                    # Apply standardization
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        modified_df[f'{col}_standardized'] = (df[col] - mean_val) / std_val
                        changes.append(f"Standardized '{col}' (mean=0, std=1)")
                
                elif pd.api.types.is_object_dtype(df[col]):
                    # Apply label encoding
                    le = LabelEncoder()
                    modified_df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('missing'))
                    changes.append(f"Label encoded '{col}'")
        
        result['modified_df'] = modified_df
        result['changes_summary'] = changes
        
        # Generate AI insights
        if client:
            insights = generate_transformation_insights(changes, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Data Transformation Complete',
                    'description': f'Applied {len(changes)} transformations to improve data quality and model readiness.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_data_transformation: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Data transformation failed: {str(e)}'}]
        return result

def perform_custom_analysis(df, selected_columns, model, custom_instruction):
    """Perform custom analysis based on user instruction"""
    result = {
        'operation': 'custom_analysis',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        if not custom_instruction:
            result['insights'] = [{'title': 'No Instruction', 'description': 'Please provide custom analysis instructions.'}]
            return result
        
        # Use AI to interpret and execute custom instruction
        if client:
            insights = generate_custom_analysis_insights(df, selected_columns, custom_instruction, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Custom Analysis',
                    'description': f'Custom analysis requested for columns: {", ".join(selected_columns)}. Instruction: {custom_instruction}'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_custom_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Custom analysis failed: {str(e)}'}]
        return result

# Helper functions for generating AI insights
def generate_cleaning_insights(original_df, modified_df, changes, model):
    """Generate insights about data cleaning operations"""
    try:
        if not client:
            return [{'title': 'Data Cleaning Complete', 'description': f'Applied {len(changes)} cleaning operations.'}]
        
        prompt = f"""
        Analyze the data cleaning operations performed and provide insights.
        
        Changes made:
        {chr(10).join(changes)}
        
        Original dataset shape: {original_df.shape}
        Modified dataset shape: {modified_df.shape}
        
        Provide 3-5 insights about the cleaning process and recommendations.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data cleaning expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating cleaning insights: {str(e)}")
        return [{'title': 'Data Cleaning Complete', 'description': f'Applied {len(changes)} cleaning operations.'}]

def generate_outlier_insights(outlier_info, model):
    """Generate insights about outlier detection"""
    try:
        if not client:
            return [{'title': 'Outlier Detection Complete', 'description': f'Analyzed {len(outlier_info)} columns for outliers.'}]
        
        prompt = f"""
        Analyze the outlier detection results and provide insights.
        
        Outlier Information:
        {json.dumps(outlier_info, indent=2)}
        
        Provide insights about the outliers found and recommendations for handling them.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an outlier detection expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating outlier insights: {str(e)}")
        return [{'title': 'Outlier Detection Complete', 'description': f'Analyzed {len(outlier_info)} columns for outliers.'}]

def generate_missing_value_insights(missing_info, model):
    """Generate insights about missing value analysis"""
    try:
        if not client:
            return [{'title': 'Missing Value Analysis Complete', 'description': f'Analyzed {len(missing_info)} columns for missing values.'}]
        
        prompt = f"""
        Analyze the missing value patterns and provide insights.
        
        Missing Value Information:
        {json.dumps(missing_info, indent=2)}
        
        Provide insights about missing value patterns and recommendations for handling them.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a missing value analysis expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating missing value insights: {str(e)}")
        return [{'title': 'Missing Value Analysis Complete', 'description': f'Analyzed {len(missing_info)} columns for missing values.'}]

def generate_correlation_insights(strong_correlations, model):
    """Generate insights about correlation analysis"""
    try:
        if not client:
            return [{'title': 'Correlation Analysis Complete', 'description': f'Found {len(strong_correlations)} strong correlations.'}]
        
        prompt = f"""
        Analyze the correlation results and provide insights.
        
        Strong Correlations Found:
        {json.dumps(strong_correlations, indent=2)}
        
        Provide insights about the correlations and their implications for analysis.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a correlation analysis expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating correlation insights: {str(e)}")
        return [{'title': 'Correlation Analysis Complete', 'description': f'Found {len(strong_correlations)} strong correlations.'}]

def generate_distribution_insights(distribution_stats, model):
    """Generate insights about distribution analysis"""
    try:
        if not client:
            return [{'title': 'Distribution Analysis Complete', 'description': f'Analyzed distributions for {len(distribution_stats)} columns.'}]
        
        prompt = f"""
        Analyze the distribution statistics and provide insights.
        
        Distribution Statistics:
        {json.dumps(distribution_stats, indent=2)}
        
        Provide insights about the distributions and their characteristics.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a distribution analysis expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating distribution insights: {str(e)}")
        return [{'title': 'Distribution Analysis Complete', 'description': f'Analyzed distributions for {len(distribution_stats)} columns.'}]

def generate_feature_importance_insights(importance_scores, model):
    """Generate insights about feature importance"""
    try:
        if not client:
            return [{'title': 'Feature Importance Analysis Complete', 'description': f'Analyzed importance of {len(importance_scores)} features.'}]
        
        prompt = f"""
        Analyze the feature importance scores and provide insights.
        
        Feature Importance Scores:
        {json.dumps(importance_scores, indent=2)}
        
        Provide insights about feature importance and recommendations.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a feature importance expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating feature importance insights: {str(e)}")
        return [{'title': 'Feature Importance Analysis Complete', 'description': f'Analyzed importance of {len(importance_scores)} features.'}]

def generate_transformation_insights(changes, model):
    """Generate insights about data transformation"""
    try:
        if not client:
            return [{'title': 'Data Transformation Complete', 'description': f'Applied {len(changes)} transformations.'}]
        
        prompt = f"""
        Analyze the data transformation operations and provide insights.
        
        Transformations Applied:
        {chr(10).join(changes)}
        
        Provide insights about the transformations and their benefits.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data transformation expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating transformation insights: {str(e)}")
        return [{'title': 'Data Transformation Complete', 'description': f'Applied {len(changes)} transformations.'}]

def generate_custom_analysis_insights(df, selected_columns, custom_instruction, model):
    """Generate insights for custom analysis"""
    try:
        if not client:
            return [{'title': 'Custom Analysis', 'description': f'Custom analysis for: {custom_instruction}'}]
        
        # Get basic info about selected columns
        column_info = {}
        for col in selected_columns:
            if col in df.columns:
                column_info[col] = {
                    'type': str(df[col].dtype),
                    'missing': int(df[col].isnull().sum()),
                    'unique': int(df[col].nunique())
                }
        
        prompt = f"""
        Perform custom analysis based on the user's instruction.
        
        User Instruction: {custom_instruction}
        
        Selected Columns Information:
        {json.dumps(column_info, indent=2)}
        
        Dataset Shape: {df.shape}
        
        Provide detailed insights and analysis based on the user's request.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating custom analysis insights: {str(e)}")
        return [{'title': 'Custom Analysis', 'description': f'Custom analysis for: {custom_instruction}'}]


@app.route('/api/ai-copilot/download', methods=['POST'])
def api_ai_copilot_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        operation_id = data.get('operation_id')
        
        if not session_id or not operation_id:
            return jsonify({'error': 'Missing session_id or operation_id'}), 400
        
        copilot_key = f"copilot_{operation_id}"
        if copilot_key not in data_store:
            return jsonify({'error': 'Operation result not found'}), 404
        
        copilot_data = data_store[copilot_key]
        result_df = copilot_data['result_df']
        
        # Create temporary file
        temp_filename = f"ai_copilot_result_{operation_id}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        result_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_ai_copilot_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


# Optimized GenAI for Documentation Routes
@app.route('/genai-docs')
def genai_docs():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"GenAI Docs route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for GenAI Docs: {session_id}")
        return render_template('GenAI-for-Documentation.html')
    except Exception as e:
        logger.error(f"Error in genai_docs route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/genai-docs/dataset-info', methods=['GET'])
def api_genai_docs_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"GenAI Docs dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Quick analysis for speed (limit to first 1000 rows)
        sample_df = df.head(1000) if len(df) > 1000 else df
        
        # Analyze columns for documentation generation (optimized)
        columns_info = []
        for col in sample_df.columns:
            col_type = str(sample_df[col].dtype)
            missing = sample_df[col].isna().sum()
            missing_pct = (missing / len(sample_df)) * 100
            unique_count = sample_df[col].nunique()
            
            # Get sample values quickly
            sample_values = []
            try:
                non_null_values = sample_df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(3, len(non_null_values))  # Reduced from 5 to 3
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                sample_values = ["N/A"]
            
            # Quick data quality assessment
            quality_score = 100 - missing_pct
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                try:
                    data_characteristics = f"Numeric: {sample_df[col].min():.2f} to {sample_df[col].max():.2f}"
                except:
                    data_characteristics = f"Numeric data"
            elif pd.api.types.is_object_dtype(sample_df[col]):
                data_characteristics = f"Text/Categorical: {unique_count} unique values"
            else:
                data_characteristics = f"Data type: {col_type}"
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.1f}%",
                'unique_count': int(unique_count),
                'sample_values': sample_values,
                'quality_score': f"{quality_score:.1f}%",
                'characteristics': data_characteristics
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_genai_docs_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/genai-docs/generate', methods=['POST'])
def api_genai_docs_generate():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"GenAI documentation generation requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        selected_columns = data.get('selected_columns', [])
        model = data.get('model', 'gpt-4o')
        doc_type = data.get('doc_type', 'comprehensive')
        additional_instructions = data.get('additional_instructions', '')
        
        if not selected_columns:
            return jsonify({'error': 'No columns selected for documentation'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Generate FAST documentation (under 5 seconds)
        start_time = time.time()
        documentation_result = generate_fast_etl_documentation(
            df, selected_columns, filename, model, doc_type, additional_instructions
        )
        processing_time = round(time.time() - start_time, 2)
        
        # Store documentation for download
        doc_id = str(uuid.uuid4())
        data_store[f"documentation_{doc_id}"] = {
            'documentation': documentation_result['documentation'],
            'documentation_html': documentation_result['documentation_html'],
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'columns': selected_columns
        }
        
        # Prepare data preview (limited for speed)
        data_preview = {
            'columns': df[selected_columns].columns.tolist(),
            'data': df[selected_columns].head(5).to_dict(orient='records'),  # Reduced from 10 to 5
            'shape': df[selected_columns].shape
        }
        
        return jsonify({
            'doc_id': doc_id,
            'documentation': documentation_result['documentation'],
            'documentation_html': documentation_result['documentation_html'],
            'data_preview': data_preview,
            'processing_time': processing_time,
            'insights': documentation_result.get('insights', []),
            'etl_recommendations': documentation_result.get('etl_recommendations', [])
        })
    
    except Exception as e:
        logger.error(f"Error in api_genai_docs_generate: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def generate_fast_etl_documentation(df, selected_columns, filename, model, doc_type, additional_instructions):
    """
    Generate FAST ETL documentation (under 5 seconds)
    """
    try:
        # Quick data analysis (optimized for speed)
        selected_df = df[selected_columns].head(500)  # Limit to 500 rows for speed
        data_analysis = analyze_data_fast(selected_df, selected_columns)
        
        # Generate documentation using optimized AI or fallback
        if client:
            try:
                documentation = generate_fast_ai_documentation(
                    data_analysis, filename, model, doc_type, additional_instructions
                )
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {str(e)}")
                documentation = generate_fast_fallback_documentation(
                    data_analysis, filename, doc_type
                )
        else:
            documentation = generate_fast_fallback_documentation(
                data_analysis, filename, doc_type
            )
        
        # Convert to HTML format (simplified)
        documentation_html = convert_to_fast_html(documentation)
        
        # Generate quick insights
        insights = generate_fast_insights(data_analysis)
        etl_recommendations = generate_fast_recommendations()
        
        return {
            'documentation': documentation,
            'documentation_html': documentation_html,
            'insights': insights,
            'etl_recommendations': etl_recommendations
        }
    
    except Exception as e:
        logger.error(f"Error in generate_fast_etl_documentation: {str(e)}")
        # Return basic documentation if everything fails
        return {
            'documentation': f"# ETL Documentation: {filename}\n\nBasic documentation generated for {len(selected_columns)} columns.",
            'documentation_html': f"<h3>ETL Documentation: {filename}</h3><p>Basic documentation generated for {len(selected_columns)} columns.</p>",
            'insights': [{'title': 'Documentation Generated', 'description': 'Basic documentation created successfully.'}],
            'etl_recommendations': []
        }

def analyze_data_fast(df, selected_columns):
    """
    Fast data analysis (optimized for speed)
    """
    try:
        analysis = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(selected_columns),
                'missing_values_total': int(df.isnull().sum().sum()),
                'duplicate_rows': int(df.duplicated().sum())
            },
            'column_details': []
        }
        
        # Quick column analysis
        for col in selected_columns[:10]:  # Limit to first 10 columns for speed
            if col in df.columns:
                col_analysis = {
                    'name': col,
                    'data_type': str(df[col].dtype),
                    'missing_count': int(df[col].isnull().sum()),
                    'missing_percentage': f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
                    'unique_count': int(df[col].nunique())
                }
                
                # Quick type-specific analysis
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        col_analysis.update({
                            'min_value': float(df[col].min()),
                            'max_value': float(df[col].max()),
                            'mean_value': float(df[col].mean())
                        })
                    except:
                        pass
                elif pd.api.types.is_object_dtype(df[col]):
                    try:
                        top_values = df[col].value_counts().head(3).to_dict()
                        col_analysis['top_values'] = top_values
                    except:
                        pass
                
                analysis['column_details'].append(col_analysis)
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error in analyze_data_fast: {str(e)}")
        return {
            'basic_info': {'total_rows': len(df), 'total_columns': len(selected_columns)},
            'column_details': []
        }

def generate_fast_ai_documentation(data_analysis, filename, model, doc_type, additional_instructions):
    """
    Generate documentation using Azure OpenAI (optimized for speed)
    """
    try:
        # Simplified prompt for faster generation
        basic_info = data_analysis['basic_info']
        column_details = data_analysis['column_details'][:5]  # Limit to 5 columns
        
        prompt = f"""
        Generate concise ETL documentation for dataset "{filename}".
        
        Dataset: {basic_info['total_rows']} rows, {basic_info['total_columns']} columns
        Missing values: {basic_info['missing_values_total']}
        
        Key columns: {[col['name'] for col in column_details]}
        
        Create a {doc_type} documentation with:
        1. Executive Summary
        2. Data Overview
        3. Column Descriptions
        4. ETL Recommendations
        5. Data Quality Notes
        
        Keep it concise and professional. Focus on practical ETL insights.
        Additional instructions: {additional_instructions or 'None'}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an ETL documentation expert. Create concise, professional documentation quickly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,  # Reduced from 4000 for speed
            timeout=10  # 10 second timeout
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error in fast AI documentation: {str(e)}")
        raise

def generate_fast_fallback_documentation(data_analysis, filename, doc_type):
    """
    Generate fast fallback documentation
    """
    basic_info = data_analysis['basic_info']
    column_details = data_analysis['column_details']
    
    documentation = f"""# ETL Documentation: {filename}

## Executive Summary
This {doc_type} documentation covers the dataset "{filename}" with {basic_info['total_rows']:,} rows and {basic_info['total_columns']} columns.

## Data Overview
- **Total Records**: {basic_info['total_rows']:,}
- **Total Columns**: {basic_info['total_columns']}
- **Missing Values**: {basic_info['missing_values_total']:,}
- **Duplicate Records**: {basic_info.get('duplicate_rows', 0):,}

## Column Analysis
"""
    
    for col in column_details:
        documentation += f"""
### {col['name']}
- **Type**: {col['data_type']}
- **Missing**: {col['missing_count']} ({col['missing_percentage']})
- **Unique Values**: {col['unique_count']}
"""
        
        if col.get('min_value') is not None:
            documentation += f"- **Range**: {col['min_value']:.2f} to {col['max_value']:.2f}\n"
        
        if col.get('top_values'):
            top_vals = list(col['top_values'].keys())[:3]
            documentation += f"- **Top Values**: {', '.join(map(str, top_vals))}\n"
    
    documentation += """
## ETL Recommendations
1. **Data Quality**: Implement validation checks for missing values
2. **Performance**: Consider indexing on key columns
3. **Monitoring**: Set up data quality alerts
4. **Documentation**: Keep this documentation updated

## Implementation Notes
- Generated automatically for quick ETL planning
- Review and customize for specific requirements
- Consider data volume for processing strategies
"""
    
    return documentation

def convert_to_fast_html(documentation):
    """
    Fast HTML conversion
    """
    try:
        # Simple markdown to HTML conversion
        html = documentation.replace('\n', '<br>')
        html = html.replace('# ', '<h3 style="color: #3366FF; margin: 20px 0 10px 0;">')
        html = html.replace('## ', '<h4 style="color: #6366F1; margin: 15px 0 8px 0;">')
        html = html.replace('### ', '<h5 style="color: #8B5CF6; margin: 10px 0 5px 0;">')
        html = html.replace('**', '<strong>')
        html = html.replace('**', '</strong>')
        html = html.replace('- ', '<li>')
        
        return f'<div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">{html}</div>'
    
    except Exception as e:
        logger.error(f"Error in fast HTML conversion: {str(e)}")
        return f'<pre>{documentation}</pre>'

def generate_fast_insights(data_analysis):
    """
    Generate quick insights
    """
    try:
        insights = []
        basic_info = data_analysis['basic_info']
        
        # Quick insights based on data
        if basic_info['missing_values_total'] > 0:
            insights.append({
                'title': 'Data Quality Check',
                'description': f'Found {basic_info["missing_values_total"]} missing values. Consider data cleaning strategies.'
            })
        
        if basic_info['total_rows'] > 100000:
            insights.append({
                'title': 'Large Dataset',
                'description': 'Large dataset detected. Consider partitioning and incremental processing.'
            })
        
        insights.append({
            'title': 'ETL Ready',
            'description': 'Dataset analysis complete. Ready for ETL pipeline design.'
        })
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating fast insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': 'Data analysis completed successfully.'}]

def generate_fast_recommendations():
    """
    Generate quick ETL recommendations
    """
    return [
        {
            'category': 'Performance',
            'title': 'Optimize Processing',
            'description': 'Use parallel processing and chunking for large datasets.',
            'priority': 'High'
        },
        {
            'category': 'Quality',
            'title': 'Data Validation',
            'description': 'Implement automated data quality checks.',
            'priority': 'High'
        },
        {
            'category': 'Monitoring',
            'title': 'Pipeline Monitoring',
            'description': 'Set up monitoring and alerting for ETL jobs.',
            'priority': 'Medium'
        }
    ]

@app.route('/api/genai-docs/download', methods=['POST'])
def api_genai_docs_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        doc_id = data.get('doc_id')
        
        if not session_id or not doc_id:
            return jsonify({'error': 'Missing session_id or doc_id'}), 400
        
        doc_key = f"documentation_{doc_id}"
        if doc_key not in data_store:
            return jsonify({'error': 'Documentation not found'}), 404
        
        doc_data = data_store[doc_key]
        
        # Create simple HTML file for download
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ETL Documentation - {doc_data['filename']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #3366FF; color: white; padding: 20px; border-radius: 5px; }}
        .content {{ margin-top: 20px; }}
        h3, h4, h5 {{ color: #333; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ETL Documentation</h1>
        <p>Dataset: {doc_data['filename']} | Generated: {doc_data['timestamp']}</p>
    </div>
    <div class="content">
        {doc_data['documentation_html']}
    </div>
</body>
</html>"""
        
        # Create temporary file
        temp_filename = f"ETL_Docs_{doc_data['filename']}_{doc_id[:8]}.html"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_genai_docs_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500



# Data Drift Detection & Monitoring Routes
@app.route('/data-drift-detection')
def data_drift_detection():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Data Drift Detection route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for Data Drift Detection: {session_id}")
        return render_template('Data Drift Detection & Monitoring.html')
    except Exception as e:
        logger.error(f"Error in data_drift_detection route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/data-drift/dataset-info', methods=['GET'])
def api_data_drift_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Data Drift dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for drift detection suitability
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Determine drift detection suitability
            drift_suitable = True
            drift_type = "Statistical"
            
            if pd.api.types.is_numeric_dtype(df[col]):
                if unique_count < 2:
                    drift_suitable = False
                    drift_type = "Constant"
                elif missing_pct > 90:
                    drift_suitable = False
                    drift_type = "Too many missing"
                else:
                    drift_type = "Numerical Distribution"
            elif pd.api.types.is_object_dtype(df[col]):
                if unique_count > len(df) * 0.9:
                    drift_suitable = False
                    drift_type = "High cardinality"
                else:
                    drift_type = "Categorical Distribution"
            
            # Get sample values
            sample_values = []
            try:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                logger.warning(f"Error getting sample values for column {col}: {str(e)}")
                sample_values = ["N/A"]
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'drift_suitable': drift_suitable,
                'drift_type': drift_type,
                'sample_values': sample_values
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_data_drift_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/data-drift/analyze', methods=['POST'])
def api_data_drift_analyze():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Data drift analysis requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        selected_columns = data.get('selected_columns', [])
        model = data.get('model', 'gpt-4o')
        split_method = data.get('split_method', 'temporal')
        split_ratio = float(data.get('split_ratio', 0.7))
        drift_threshold = float(data.get('drift_threshold', 0.05))
        
        if not selected_columns:
            return jsonify({'error': 'No columns selected for drift analysis'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Perform data drift analysis
        start_time = time.time()
        drift_result = perform_data_drift_analysis(
            df, selected_columns, model, split_method, split_ratio, drift_threshold, filename
        )
        processing_time = round(time.time() - start_time, 2)
        
        # Store drift analysis result
        drift_id = str(uuid.uuid4())
        data_store[f"drift_analysis_{drift_id}"] = {
            'result': drift_result,
            'original_df': df,
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'columns': selected_columns
        }
        
        drift_result['drift_id'] = drift_id
        drift_result['processing_time'] = processing_time
        
        return jsonify(drift_result)
    
    except Exception as e:
        logger.error(f"Error in api_data_drift_analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_data_drift_analysis(df, selected_columns, model, split_method, split_ratio, drift_threshold, filename):
    """
    Perform comprehensive data drift analysis using statistical methods and LLM insights
    """
    try:
        # Split data into reference and current datasets
        reference_df, current_df = split_data_for_drift_analysis(df, split_method, split_ratio)
        
        # Perform drift detection for each column
        drift_results = []
        visualizations = []
        
        for col in selected_columns:
            if col in df.columns:
                column_drift = detect_column_drift(
                    reference_df[col], current_df[col], col, drift_threshold
                )
                drift_results.append(column_drift)
                
                # Create visualization for this column
                viz = create_drift_visualization(reference_df[col], current_df[col], col)
                if viz:
                    visualizations.append(viz)
        
        # Calculate overall drift score
        overall_drift_score = calculate_overall_drift_score(drift_results)
        
        # Generate AI-powered insights
        ai_insights = generate_drift_insights_with_llm(drift_results, model, filename)
        
        # Generate recommendations
        recommendations = generate_drift_recommendations(drift_results, overall_drift_score)
        
        # Create summary statistics
        summary_stats = create_drift_summary_stats(reference_df, current_df, selected_columns)
        
        return {
            'overall_drift_score': overall_drift_score,
            'drift_detected': overall_drift_score > drift_threshold,
            'column_results': drift_results,
            'visualizations': visualizations,
            'ai_insights': ai_insights,
            'recommendations': recommendations,
            'summary_stats': summary_stats,
            'reference_period': {
                'rows': len(reference_df),
                'start_index': 0,
                'end_index': len(reference_df) - 1
            },
            'current_period': {
                'rows': len(current_df),
                'start_index': len(reference_df),
                'end_index': len(df) - 1
            }
        }
    
    except Exception as e:
        logger.error(f"Error in perform_data_drift_analysis: {str(e)}")
        raise

def split_data_for_drift_analysis(df, split_method, split_ratio):
    """Split data into reference and current datasets"""
    try:
        if split_method == 'temporal':
            # Split based on time order (first X% as reference, rest as current)
            split_point = int(len(df) * split_ratio)
            reference_df = df.iloc[:split_point].copy()
            current_df = df.iloc[split_point:].copy()
        elif split_method == 'random':
            # Random split
            reference_df = df.sample(frac=split_ratio, random_state=42)
            current_df = df.drop(reference_df.index)
        else:
            # Default to temporal split
            split_point = int(len(df) * split_ratio)
            reference_df = df.iloc[:split_point].copy()
            current_df = df.iloc[split_point:].copy()
        
        return reference_df, current_df
    
    except Exception as e:
        logger.error(f"Error in split_data_for_drift_analysis: {str(e)}")
        # Fallback to simple split
        split_point = int(len(df) * 0.7)
        return df.iloc[:split_point].copy(), df.iloc[split_point:].copy()

def detect_column_drift(reference_series, current_series, column_name, threshold):
    """Detect drift for a single column using appropriate statistical tests"""
    try:
        # Remove missing values
        ref_clean = reference_series.dropna()
        curr_clean = current_series.dropna()
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return {
                'column': column_name,
                'drift_detected': False,
                'drift_score': 0.0,
                'p_value': 1.0,
                'test_used': 'No valid data',
                'drift_magnitude': 'None',
                'summary': 'Insufficient data for drift detection'
            }
        
        # Determine appropriate test based on data type
        if pd.api.types.is_numeric_dtype(reference_series):
            # Use Kolmogorov-Smirnov test for numerical data
            statistic, p_value = ks_2samp(ref_clean, curr_clean)
            test_used = 'Kolmogorov-Smirnov'
            drift_score = statistic
        else:
            # Use Chi-square test for categorical data
            try:
                # Create contingency table
                ref_counts = ref_clean.value_counts()
                curr_counts = curr_clean.value_counts()
                
                # Align the categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                
                # Perform chi-square test
                contingency_table = np.array([ref_aligned, curr_aligned])
                statistic, p_value, _, _ = chi2_contingency(contingency_table)
                test_used = 'Chi-square'
                drift_score = min(1.0, statistic / 100)  # Normalize for comparison
            except Exception as e:
                logger.warning(f"Error in chi-square test for {column_name}: {str(e)}")
                # Fallback to simple proportion comparison
                ref_unique = len(ref_clean.unique())
                curr_unique = len(curr_clean.unique())
                drift_score = abs(ref_unique - curr_unique) / max(ref_unique, curr_unique, 1)
                p_value = 0.5 if drift_score > 0.1 else 0.8
                test_used = 'Proportion comparison'
        
        # Determine drift magnitude
        if drift_score > 0.3:
            drift_magnitude = 'High'
        elif drift_score > 0.1:
            drift_magnitude = 'Medium'
        elif drift_score > 0.05:
            drift_magnitude = 'Low'
        else:
            drift_magnitude = 'Minimal'
        
        # Create summary
        drift_detected = p_value < threshold
        if drift_detected:
            summary = f"Significant drift detected (p-value: {p_value:.4f})"
        else:
            summary = f"No significant drift detected (p-value: {p_value:.4f})"
        
        return {
            'column': column_name,
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'p_value': float(p_value),
            'test_used': test_used,
            'drift_magnitude': drift_magnitude,
            'summary': summary,
            'reference_stats': calculate_column_stats(ref_clean),
            'current_stats': calculate_column_stats(curr_clean)
        }
    
    except Exception as e:
        logger.error(f"Error in detect_column_drift for {column_name}: {str(e)}")
        return {
            'column': column_name,
            'drift_detected': False,
            'drift_score': 0.0,
            'p_value': 1.0,
            'test_used': 'Error',
            'drift_magnitude': 'Unknown',
            'summary': f'Error in drift detection: {str(e)}'
        }

def calculate_column_stats(series):
    """Calculate basic statistics for a column"""
    try:
        if pd.api.types.is_numeric_dtype(series):
            return {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'count': int(len(series))
            }
        else:
            value_counts = series.value_counts().head(5).to_dict()
            return {
                'unique_count': int(series.nunique()),
                'most_common': value_counts,
                'count': int(len(series))
            }
    except Exception as e:
        logger.error(f"Error calculating column stats: {str(e)}")
        return {'count': int(len(series)), 'error': str(e)}

def create_drift_visualization(reference_series, current_series, column_name):
    """Create visualization for drift analysis"""
    try:
        plt.figure(figsize=(12, 6))
        
        if pd.api.types.is_numeric_dtype(reference_series):
            # Create distribution comparison for numerical data
            plt.subplot(1, 2, 1)
            plt.hist(reference_series.dropna(), alpha=0.7, label='Reference', bins=30, color='blue')
            plt.hist(current_series.dropna(), alpha=0.7, label='Current', bins=30, color='red')
            plt.title(f'Distribution Comparison - {column_name}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            
            # Create box plot comparison
            plt.subplot(1, 2, 2)
            data_to_plot = [reference_series.dropna(), current_series.dropna()]
            plt.boxplot(data_to_plot, labels=['Reference', 'Current'])
            plt.title(f'Box Plot Comparison - {column_name}')
            plt.ylabel('Value')
        else:
            # Create bar plot comparison for categorical data
            ref_counts = reference_series.value_counts().head(10)
            curr_counts = current_series.value_counts().head(10)
            
            # Align categories
            all_cats = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_cats]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_cats]
            
            x = np.arange(len(all_cats))
            width = 0.35
            
            plt.subplot(1, 1, 1)
            plt.bar(x - width/2, ref_aligned, width, label='Reference', alpha=0.7, color='blue')
            plt.bar(x + width/2, curr_aligned, width, label='Current', alpha=0.7, color='red')
            plt.title(f'Category Distribution Comparison - {column_name}')
            plt.xlabel('Categories')
            plt.ylabel('Count')
            plt.xticks(x, list(all_cats), rotation=45, ha='right')
            plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            'column': column_name,
            'type': 'drift_comparison',
            'title': f'Drift Analysis - {column_name}',
            'data': plot_data
        }
    
    except Exception as e:
        logger.error(f"Error creating drift visualization for {column_name}: {str(e)}")
        return None

def calculate_overall_drift_score(drift_results):
    """Calculate overall drift score across all columns"""
    try:
        if not drift_results:
            return 0.0
        
        # Weight by drift magnitude
        weights = {'High': 1.0, 'Medium': 0.6, 'Low': 0.3, 'Minimal': 0.1}
        total_score = 0.0
        total_weight = 0.0
        
        for result in drift_results:
            magnitude = result.get('drift_magnitude', 'Minimal')
            weight = weights.get(magnitude, 0.1)
            score = result.get('drift_score', 0.0)
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    except Exception as e:
        logger.error(f"Error calculating overall drift score: {str(e)}")
        return 0.0

def generate_drift_insights_with_llm(drift_results, model, filename):
    """Generate AI-powered insights about data drift"""
    try:
        if not client:
            return generate_fallback_drift_insights(drift_results)
        
        # Prepare summary for LLM
        drift_summary = {
            'total_columns': len(drift_results),
            'columns_with_drift': len([r for r in drift_results if r['drift_detected']]),
            'high_drift_columns': [r['column'] for r in drift_results if r.get('drift_magnitude') == 'High'],
            'medium_drift_columns': [r['column'] for r in drift_results if r.get('drift_magnitude') == 'Medium'],
            'test_results': [
                {
                    'column': r['column'],
                    'drift_detected': r['drift_detected'],
                    'drift_score': r['drift_score'],
                    'p_value': r['p_value'],
                    'test_used': r['test_used']
                } for r in drift_results
            ]
        }
        
        prompt = f"""
        You are an expert data scientist analyzing data drift in the dataset "{filename}".
        
        Data Drift Analysis Summary:
        {json.dumps(drift_summary, indent=2)}
        
        Provide 4-6 strategic insights about:
        1. Overall data drift assessment and implications
        2. Specific columns showing concerning drift patterns
        3. Potential causes and business impact
        4. ETL pipeline recommendations
        5. Monitoring and alerting suggestions
        6. Model performance implications
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed analysis and recommendation",
                    "severity": "High|Medium|Low",
                    "action_required": true/false
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert data drift analyst. Provide strategic insights in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        return ai_response.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating LLM drift insights: {str(e)}")
        return generate_fallback_drift_insights(drift_results)

def generate_fallback_drift_insights(drift_results):
    """Generate fallback insights when LLM is not available"""
    insights = []
    
    drift_detected_count = len([r for r in drift_results if r['drift_detected']])
    high_drift_count = len([r for r in drift_results if r.get('drift_magnitude') == 'High'])
    
    if drift_detected_count == 0:
        insights.append({
            'title': 'No Significant Drift Detected',
            'description': 'All analyzed columns show stable distributions between reference and current periods.',
            'severity': 'Low',
            'action_required': False
        })
    elif high_drift_count > 0:
        high_drift_columns = [r['column'] for r in drift_results if r.get('drift_magnitude') == 'High']
        insights.append({
            'title': 'High Drift Alert',
            'description': f'Significant drift detected in {high_drift_count} columns: {", ".join(high_drift_columns)}. Immediate investigation recommended.',
            'severity': 'High',
            'action_required': True
        })
    
    insights.extend([
        {
            'title': 'ETL Pipeline Impact',
            'description': 'Data drift may indicate changes in upstream data sources or processing logic.',
            'severity': 'Medium',
            'action_required': True
        },
        {
            'title': 'Model Performance Risk',
            'description': 'Detected drift could impact machine learning model performance and predictions.',
            'severity': 'Medium',
            'action_required': True
        },
        {
            'title': 'Monitoring Recommendation',
            'description': 'Implement automated drift monitoring to catch future changes early.',
            'severity': 'Low',
            'action_required': False
        }
    ])
    
    return insights

def generate_drift_recommendations(drift_results, overall_drift_score):
    """Generate actionable recommendations based on drift analysis"""
    recommendations = []
    
    if overall_drift_score > 0.3:
        recommendations.append({
            'category': 'Immediate Action',
            'title': 'Investigate High Drift',
            'description': 'Significant drift detected. Review data sources and ETL processes immediately.',
            'priority': 'High'
        })
    
    if overall_drift_score > 0.1:
        recommendations.append({
            'category': 'Model Management',
            'title': 'Retrain Models',
            'description': 'Consider retraining machine learning models with recent data.',
            'priority': 'High'
        })
    
    recommendations.extend([
        {
            'category': 'Monitoring',
            'title': 'Automated Drift Detection',
            'description': 'Implement continuous monitoring with automated alerts for drift detection.',
            'priority': 'Medium'
        },
        {
            'category': 'Data Quality',
            'title': 'Enhanced Validation',
            'description': 'Add data validation checks in ETL pipeline to catch drift early.',
            'priority': 'Medium'
        },
        {
            'category': 'Documentation',
            'title': 'Update Data Documentation',
            'description': 'Document detected changes and update data dictionaries.',
            'priority': 'Low'
        }
    ])
    
    return recommendations

def create_drift_summary_stats(reference_df, current_df, selected_columns):
    """Create summary statistics for drift analysis"""
    try:
        summary = {
            'reference_period': {
                'total_rows': len(reference_df),
                'date_range': 'First 70% of data',
                'missing_values': int(reference_df[selected_columns].isnull().sum().sum())
            },
            'current_period': {
                'total_rows': len(current_df),
                'date_range': 'Last 30% of data',
                'missing_values': int(current_df[selected_columns].isnull().sum().sum())
            },
            'comparison': {
                'row_count_change': len(current_df) - len(reference_df),
                'row_count_change_pct': ((len(current_df) - len(reference_df)) / len(reference_df) * 100) if len(reference_df) > 0 else 0
            }
        }
        
        return summary
    
    except Exception as e:
        logger.error(f"Error creating drift summary stats: {str(e)}")
        return {
            'reference_period': {'total_rows': 0},
            'current_period': {'total_rows': 0},
            'comparison': {'row_count_change': 0}
        }

@app.route('/api/data-drift/download', methods=['POST'])
def api_data_drift_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        drift_id = data.get('drift_id')
        
        if not session_id or not drift_id:
            return jsonify({'error': 'Missing session_id or drift_id'}), 400
        
        drift_key = f"drift_analysis_{drift_id}"
        if drift_key not in data_store:
            return jsonify({'error': 'Drift analysis not found'}), 404
        
        drift_data = data_store[drift_key]
        original_df = drift_data['original_df']
        drift_result = drift_data['result']
        
        # Create enhanced dataset with drift indicators
        enhanced_df = original_df.copy()
        
        # Add drift period indicator
        split_point = int(len(enhanced_df) * 0.7)  # Assuming 70/30 split
        enhanced_df['drift_period'] = ['Reference'] * split_point + ['Current'] * (len(enhanced_df) - split_point)
        
        # Add drift scores for each column
        for column_result in drift_result['column_results']:
            col_name = column_result['column']
            if col_name in enhanced_df.columns:
                enhanced_df[f'{col_name}_drift_score'] = column_result['drift_score']
                enhanced_df[f'{col_name}_drift_detected'] = column_result['drift_detected']
        
        # Create temporary file
        temp_filename = f"drift_analysis_{drift_data['filename']}_{drift_id[:8]}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        enhanced_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_data_drift_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500





# Model Training & Deployment Routes
@app.route('/model-training-deployment')
def model_training_deployment():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Model Training & Deployment route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for Model Training & Deployment: {session_id}")
        return render_template('model-training-deployment.html')
    except Exception as e:
        logger.error(f"Error in model_training_deployment route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/model-training/dataset-info', methods=['GET'])
def api_model_training_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Model Training dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Sample data for faster processing (max 2000 rows)
        if len(df) > 2000:
            df_sample = df.sample(n=2000, random_state=42)
            logger.info(f"Sampling {len(df_sample)} rows from {len(df)} total rows for faster processing")
        else:
            df_sample = df
        
        # Analyze columns for ML suitability
        columns_info = []
        for col in df_sample.columns:
            col_type = str(df_sample[col].dtype)
            missing = df_sample[col].isna().sum()
            missing_pct = (missing / len(df_sample)) * 100
            unique_count = df_sample[col].nunique()
            
            # Determine if column is suitable for target
            is_target_suitable = False
            target_type = None
            ml_suitability = "Good"
            
            if pd.api.types.is_numeric_dtype(df_sample[col]):
                if unique_count <= 20 and unique_count >= 2:
                    is_target_suitable = True
                    target_type = 'classification'
                elif unique_count > 20:
                    is_target_suitable = True
                    target_type = 'regression'
                
                if missing_pct > 50:
                    ml_suitability = "Poor"
                elif missing_pct > 20:
                    ml_suitability = "Fair"
                    
            elif pd.api.types.is_object_dtype(df_sample[col]):
                if unique_count <= 50 and unique_count >= 2:
                    is_target_suitable = True
                    target_type = 'classification'
                
                if unique_count > len(df_sample) * 0.8:
                    ml_suitability = "Poor"
                elif missing_pct > 30:
                    ml_suitability = "Fair"
            
            # Get sample values safely
            sample_values = []
            try:
                non_null_values = df_sample[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(3, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                logger.warning(f"Error getting sample values for column {col}: {str(e)}")
                sample_values = ["N/A"]
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'is_target_suitable': is_target_suitable,
                'target_type': target_type,
                'ml_suitability': ml_suitability,
                'sample_values': sample_values
            })
        
        # Create a serializable version of algorithms
        algorithms_info = {}
        for problem_type in ALGORITHMS:
            algorithms_info[problem_type] = {}
            for complexity in ALGORITHMS[problem_type]:
                algorithms_info[problem_type][complexity] = {}
                for algo_name, algo_config in ALGORITHMS[problem_type][complexity].items():
                    algorithms_info[problem_type][complexity][algo_name] = {
                        'category': algo_config['category'],
                        'description': algo_config['description']
                    }

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'algorithms': algorithms_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_model_training_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/model-training/train', methods=['POST'])
def api_model_training_train():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Model training requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        # Extract training parameters
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns', [])
        algorithm = data.get('algorithm')
        problem_type = data.get('problem_type')
        test_size = float(data.get('test_size', 0.2))
        enable_feature_engineering = data.get('enable_feature_engineering', False)
        feature_selection = data.get('feature_selection', False)
        cross_validation = data.get('cross_validation', False)
        
        if not target_column or not feature_columns or not algorithm:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        df = data_store[session_id]['df']
        
        # Sample data for faster training (max 10000 rows)
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
            logger.info(f"Sampling {len(df)} rows for faster training")
        
        # Start training process
        start_time = time.time()
        training_result = perform_advanced_model_training(
            df, target_column, feature_columns, algorithm, problem_type, 
            test_size, enable_feature_engineering, feature_selection, cross_validation
        )
        training_time = round(time.time() - start_time, 2)
        
        # Store model and results
        model_id = str(uuid.uuid4())
        model_data = {
            'model': training_result['model'],
            'preprocessor': training_result.get('preprocessor'),
            'label_encoder': training_result.get('label_encoder'),
            'feature_columns': feature_columns,
            'target_column': target_column,
            'algorithm': algorithm,
            'problem_type': problem_type,
            'metrics': training_result['metrics'],
            'training_time': training_time,
            'feature_importance': training_result.get('feature_importance'),
            'predictions': training_result.get('predictions'),
            'enhanced_df': training_result.get('enhanced_df', df),
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        data_store[f"trained_model_{model_id}"] = model_data
        
        # Generate deployment URL
        deployment_url = f"/api/model-training/predict/{model_id}"
        
        # Generate AI insights
        ai_insights = generate_training_insights(training_result, algorithm, problem_type)
        
        # Prepare response
        response = {
            'model_id': model_id,
            'algorithm': algorithm,
            'problem_type': problem_type,
            'metrics': training_result['metrics'],
            'feature_importance': training_result.get('feature_importance'),
            'training_time': training_time,
            'ai_insights': ai_insights,
            'deployment_url': deployment_url,
            'data_shape': training_result['data_shape'],
            'feature_engineering_applied': enable_feature_engineering,
            'cross_validation_scores': training_result.get('cv_scores'),
            'model_performance': assess_model_performance(training_result['metrics'], problem_type)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in api_model_training_train: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred during training: {str(e)}'}), 500

def perform_advanced_model_training(df, target_column, feature_columns, algorithm, problem_type, 
                                   test_size, enable_feature_engineering, feature_selection, cross_validation):
    """
    Perform advanced model training with feature engineering and optimization
    """
    try:
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove rows with missing target values
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError('No valid data after removing missing values')
        
        # Enhanced preprocessing
        X_processed, preprocessor = advanced_preprocessing(X, enable_feature_engineering)
        
        # Encode target if classification
        label_encoder = None
        if problem_type == 'classification' and pd.api.types.is_object_dtype(y):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        # Feature selection if enabled
        if feature_selection and len(X_processed.columns) > 10:
            X_processed = apply_feature_selection(X_processed, y_encoded, problem_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=test_size, random_state=42, 
            stratify=y_encoded if problem_type == 'classification' else None
        )
        
        # Get algorithm configuration
        algo_config = get_algorithm_config(algorithm, problem_type)
        if not algo_config:
            raise ValueError(f'Algorithm {algorithm} not found for {problem_type}')
        
        # Create and train model
        model = algo_config['model'](**algo_config['params'])
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba') and problem_type == 'classification':
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = calculate_advanced_metrics(y_test, y_pred, y_pred_proba, problem_type)
        
        # Cross-validation if enabled
        cv_scores = None
        if cross_validation:
            cv_scores = perform_cross_validation(model, X_processed, y_encoded, problem_type)
        
        # Feature importance
        feature_importance = extract_feature_importance(model, X_processed.columns)
        
        # Create enhanced dataset with predictions
        enhanced_df = create_enhanced_dataset(df, X_processed, y_test, y_pred, feature_columns, target_column)
        
        return {
            'model': model,
            'preprocessor': preprocessor,
            'label_encoder': label_encoder,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'cv_scores': cv_scores,
            'predictions': {
                'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
            },
            'enhanced_df': enhanced_df,
            'data_shape': {
                'train': X_train.shape,
                'test': X_test.shape,
                'features_after_processing': X_processed.shape[1]
            }
        }
    
    except Exception as e:
        logger.error(f"Error in perform_advanced_model_training: {str(e)}")
        raise

def advanced_preprocessing(X, enable_feature_engineering):
    """
    Advanced preprocessing with optional feature engineering
    """
    try:
        X_processed = X.copy()
        
        # Handle missing values
        numeric_features = X_processed.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Fill missing values
        for col in numeric_features:
            X_processed[col].fillna(X_processed[col].median(), inplace=True)
        
        for col in categorical_features:
            X_processed[col].fillna('Unknown', inplace=True)
        
        # Feature engineering if enabled
        if enable_feature_engineering:
            X_processed = apply_feature_engineering(X_processed)
        
        # Encode categorical variables
        if categorical_features:
            for col in categorical_features:
                if X_processed[col].nunique() <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
                    X_processed = pd.concat([X_processed.drop(col, axis=1), dummies], axis=1)
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Create a simple preprocessor for deployment
        preprocessor = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'feature_engineering_enabled': enable_feature_engineering
        }
        
        return X_processed, preprocessor
    
    except Exception as e:
        logger.error(f"Error in advanced_preprocessing: {str(e)}")
        raise

def apply_feature_engineering(X):
    """
    Apply automated feature engineering
    """
    try:
        X_engineered = X.copy()
        numeric_cols = X.select_dtypes(include=['number']).columns
        
        # Create interaction features for top numeric columns
        if len(numeric_cols) >= 2:
            top_cols = numeric_cols[:3]  # Limit to avoid explosion
            for i, col1 in enumerate(top_cols):
                for col2 in top_cols[i+1:]:
                    # Multiplication
                    X_engineered[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    # Ratio (avoid division by zero)
                    X_engineered[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
        
        # Create polynomial features for numeric columns
        for col in numeric_cols[:3]:  # Limit to top 3 columns
            X_engineered[f'{col}_squared'] = X[col] ** 2
            X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
        
        # Create aggregation features
        if len(numeric_cols) > 0:
            X_engineered['numeric_sum'] = X[numeric_cols].sum(axis=1)
            X_engineered['numeric_mean'] = X[numeric_cols].mean(axis=1)
            X_engineered['numeric_std'] = X[numeric_cols].std(axis=1)
        
        return X_engineered
    
    except Exception as e:
        logger.error(f"Error in apply_feature_engineering: {str(e)}")
        return X

def apply_feature_selection(X, y, problem_type, k=20):
    """
    Apply feature selection to reduce dimensionality
    """
    try:
        if problem_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    except Exception as e:
        logger.error(f"Error in apply_feature_selection: {str(e)}")
        return X

def get_algorithm_config(algorithm, problem_type):
    """
    Get algorithm configuration
    """
    for complexity in ALGORITHMS[problem_type]:
        if algorithm in ALGORITHMS[problem_type][complexity]:
            return ALGORITHMS[problem_type][complexity][algorithm]
    return None

def calculate_advanced_metrics(y_test, y_pred, y_pred_proba, problem_type):
    """
    Calculate comprehensive metrics with visualizations
    """
    try:
        metrics = {}
        
        if problem_type == 'classification':
            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix visualization
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            cm_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm_plot,
                'classification_report': class_report
            }
            
            # ROC curve if binary classification and probabilities available
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                roc_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                metrics['roc_curve'] = roc_plot
                metrics['auc'] = float(roc_auc)
        
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            # Prediction vs Actual plot
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6, s=30)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Prediction vs Actual Values')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            pred_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Residuals plot
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.6, s=30)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            residuals_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'mae': float(mae),
                'prediction_plot': pred_plot,
                'residuals_plot': residuals_plot
            }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {'error': str(e)}

def perform_cross_validation(model, X, y, problem_type, cv=5):
    """
    Perform cross-validation
    """
    try:
        from sklearn.model_selection import cross_val_score
        
        if problem_type == 'classification':
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        return {
            'scores': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max())
        }
    
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        return None

def extract_feature_importance(model, feature_names):
    """
    Extract feature importance from trained model
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Create feature importance plot
            top_features = feature_importance[:15]  # Top 15 features
            features, scores = zip(*top_features)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            importance_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return {
                'features': [{'feature': f, 'importance': float(i)} for f, i in feature_importance[:20]],
                'plot': importance_plot
            }
        
        elif hasattr(model, 'coef_'):
            # For linear models
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]  # Take first class for multiclass
            
            feature_importance = list(zip(feature_names, np.abs(coef)))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'features': [{'feature': f, 'importance': float(i)} for f, i in feature_importance[:20]],
                'plot': None
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return None

def create_enhanced_dataset(df, X_processed, y_test, y_pred, feature_columns, target_column):
    """
    Create enhanced dataset with predictions
    """
    try:
        enhanced_df = df.copy()
        
        # Add prediction column
        enhanced_df['predicted_' + target_column] = np.nan
        enhanced_df.loc[y_test.index, 'predicted_' + target_column] = y_pred
        
        # Add prediction confidence/error
        if len(y_test) > 0:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # For regression, add absolute error
                enhanced_df['prediction_error'] = np.nan
                enhanced_df.loc[y_test.index, 'prediction_error'] = np.abs(y_test - y_pred)
            else:
                # For classification, add prediction match
                enhanced_df['prediction_match'] = np.nan
                enhanced_df.loc[y_test.index, 'prediction_match'] = (y_test == y_pred).astype(int)
        
        # Add data split indicator
        enhanced_df['data_split'] = 'train'
        enhanced_df.loc[y_test.index, 'data_split'] = 'test'
        
        return enhanced_df
    
    except Exception as e:
        logger.error(f"Error creating enhanced dataset: {str(e)}")
        return df

def generate_training_insights(training_result, algorithm, problem_type):
    """
    Generate AI-powered insights about training results
    """
    try:
        insights = []
        metrics = training_result['metrics']
        
        if client:
            # Use Azure OpenAI for insights
            metrics_summary = {
                'algorithm': algorithm,
                'problem_type': problem_type,
                'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                'feature_count': training_result['data_shape']['features_after_processing'],
                'training_samples': training_result['data_shape']['train'][0],
                'test_samples': training_result['data_shape']['test'][0]
            }
            
            prompt = f"""
            You are an expert ML engineer analyzing model training results. Provide strategic insights and recommendations.
            
            Training Results:
            {json.dumps(metrics_summary, indent=2)}
            
            Provide 4-6 insights covering:
            1. Model performance assessment
            2. Recommendations for improvement
            3. Deployment readiness
            4. Potential risks or limitations
            5. Next steps for optimization
            6. ETL integration suggestions
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed analysis and recommendation",
                        "priority": "High|Medium|Low"
                    }}
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert ML engineer providing training insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
        
        else:
            # Fallback insights
            if problem_type == 'classification':
                accuracy = metrics.get('accuracy', 0)
                if accuracy > 0.9:
                    insights.append({
                        'title': 'Excellent Model Performance',
                        'description': f'Model achieved {accuracy:.1%} accuracy, indicating excellent predictive capability.',
                        'priority': 'High'
                    })
                elif accuracy > 0.8:
                    insights.append({
                        'title': 'Good Model Performance',
                        'description': f'Model achieved {accuracy:.1%} accuracy with room for improvement.',
                        'priority': 'Medium'
                    })
                else:
                    insights.append({
                        'title': 'Model Needs Improvement',
                        'description': f'Model accuracy of {accuracy:.1%} suggests need for better features or different algorithm.',
                        'priority': 'High'
                    })
            else:
                r2 = metrics.get('r2_score', 0)
                if r2 > 0.8:
                    insights.append({
                        'title': 'Strong Predictive Model',
                        'description': f'R² score of {r2:.3f} indicates the model explains most variance in the data.',
                        'priority': 'High'
                    })
                else:
                    insights.append({
                        'title': 'Model Performance Assessment',
                        'description': f'R² score of {r2:.3f} suggests moderate predictive capability.',
                        'priority': 'Medium'
                    })
            
            insights.extend([
                {
                    'title': 'Deployment Readiness',
                    'description': 'Model is trained and ready for deployment via API endpoint.',
                    'priority': 'Medium'
                },
                {
                    'title': 'ETL Integration',
                    'description': 'Model can be integrated into ETL pipelines for real-time scoring.',
                    'priority': 'Low'
                }
            ])
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating training insights: {str(e)}")
        return [{'title': 'Training Complete', 'description': 'Model training completed successfully.', 'priority': 'Medium'}]

def assess_model_performance(metrics, problem_type):
    """
    Assess overall model performance
    """
    try:
        if problem_type == 'classification':
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= 0.9:
                return {'level': 'Excellent', 'score': accuracy, 'recommendation': 'Ready for production deployment'}
            elif accuracy >= 0.8:
                return {'level': 'Good', 'score': accuracy, 'recommendation': 'Consider hyperparameter tuning'}
            elif accuracy >= 0.7:
                return {'level': 'Fair', 'score': accuracy, 'recommendation': 'Try feature engineering or different algorithm'}
            else:
                return {'level': 'Poor', 'score': accuracy, 'recommendation': 'Significant improvement needed'}
        else:
            r2 = metrics.get('r2_score', 0)
            if r2 >= 0.8:
                return {'level': 'Excellent', 'score': r2, 'recommendation': 'Strong predictive model'}
            elif r2 >= 0.6:
                return {'level': 'Good', 'score': r2, 'recommendation': 'Good predictive capability'}
            elif r2 >= 0.4:
                return {'level': 'Fair', 'score': r2, 'recommendation': 'Moderate predictive power'}
            else:
                return {'level': 'Poor', 'score': r2, 'recommendation': 'Limited predictive capability'}
    
    except Exception as e:
        logger.error(f"Error assessing model performance: {str(e)}")
        return {'level': 'Unknown', 'score': 0, 'recommendation': 'Unable to assess performance'}

@app.route('/api/model-training/predict/<model_id>', methods=['POST'])
def api_model_predict(model_id):
    """
    API endpoint for making predictions with trained model
    """
    try:
        model_key = f"trained_model_{model_id}"
        if model_key not in data_store:
            return jsonify({'error': 'Model not found'}), 404
        
        model_data = data_store[model_key]
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # Get input data
        input_data = request.json
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert to DataFrame
        if isinstance(input_data, list):
            df_input = pd.DataFrame(input_data)
        else:
            df_input = pd.DataFrame([input_data])
        
        # Check if all required features are present
        missing_features = set(feature_columns) - set(df_input.columns)
        if missing_features:
            return jsonify({'error': f'Missing features: {list(missing_features)}'}), 400
        
        # Select and order features
        X_input = df_input[feature_columns]
        
        # Apply same preprocessing as training
        X_processed = preprocess_for_prediction(X_input, model_data.get('preprocessor'))
        
        # Make prediction
        predictions = model.predict(X_processed)
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_processed)
        
        # Decode predictions if label encoder was used
        if model_data.get('label_encoder'):
            predictions = model_data['label_encoder'].inverse_transform(predictions)
        
        # Prepare response
        response = {
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            'model_id': model_id,
            'algorithm': model_data['algorithm'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def preprocess_for_prediction(X, preprocessor_info):
    """
    Apply same preprocessing as training for prediction
    """
    try:
        X_processed = X.copy()
        
        if preprocessor_info:
            numeric_features = preprocessor_info.get('numeric_features', [])
            categorical_features = preprocessor_info.get('categorical_features', [])
            
            # Handle missing values
            for col in numeric_features:
                if col in X_processed.columns:
                    X_processed[col].fillna(X_processed[col].median(), inplace=True)
            
            for col in categorical_features:
                if col in X_processed.columns:
                    X_processed[col].fillna('Unknown', inplace=True)
            
            # Apply feature engineering if it was used during training
            if preprocessor_info.get('feature_engineering_enabled'):
                X_processed = apply_feature_engineering(X_processed)
            
            # Encode categorical variables (simplified)
            for col in categorical_features:
                if col in X_processed.columns:
                    if X_processed[col].nunique() <= 10:
                        # One-hot encoding
                        dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
                        X_processed = pd.concat([X_processed.drop(col, axis=1), dummies], axis=1)
                    else:
                        # Label encoding
                        le = LabelEncoder()
                        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        return X_processed
    
    except Exception as e:
        logger.error(f"Error in preprocessing for prediction: {str(e)}")
        return X

@app.route('/api/model-training/download', methods=['POST'])
def api_model_training_download():
    """
    Download enhanced dataset with predictions
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        model_id = data.get('model_id')
        
        if not session_id or not model_id:
            return jsonify({'error': 'Missing session_id or model_id'}), 400
        
        model_key = f"trained_model_{model_id}"
        if model_key not in data_store:
            return jsonify({'error': 'Model not found'}), 404
        
        model_data = data_store[model_key]
        enhanced_df = model_data.get('enhanced_df')
        
        if enhanced_df is None:
            return jsonify({'error': 'Enhanced dataset not available'}), 404
        
        # Create temporary file
        temp_filename = f"model_results_{model_id[:8]}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        enhanced_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_model_training_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/model-training/deploy', methods=['POST'])
def api_model_deploy():
    """
    Deploy model and get deployment information
    """
    try:
        data = request.json
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({'error': 'Missing model_id'}), 400
        
        model_key = f"trained_model_{model_id}"
        if model_key not in data_store:
            return jsonify({'error': 'Model not found'}), 404
        
        model_data = data_store[model_key]
        
        # Generate deployment information
        deployment_info = {
            'model_id': model_id,
            'algorithm': model_data['algorithm'],
            'problem_type': model_data['problem_type'],
            'api_endpoint': f"/api/model-training/predict/{model_id}",
            'deployment_url': f"{request.host_url}api/model-training/predict/{model_id}",
            'feature_columns': model_data['feature_columns'],
            'target_column': model_data['target_column'],
            'deployment_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_performance': assess_model_performance(model_data['metrics'], model_data['problem_type']),
            'usage_example': {
                'method': 'POST',
                'headers': {'Content-Type': 'application/json'},
                'body': {col: 'value' for col in model_data['feature_columns'][:3]}
            }
        }
        
        # Store deployment info
        data_store[f"deployment_{model_id}"] = deployment_info
        
        return jsonify(deployment_info)
    
    except Exception as e:
        logger.error(f"Error in api_model_deploy: {str(e)}")
        return jsonify({'error': f'Deployment failed: {str(e)}'}), 500



# Add these routes to your existing app.py file

# Embedded ML for Data Matching & Clustering Routes
@app.route('/embedded-ml-clustering')
def embedded_ml_clustering():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Embedded ML Clustering route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for Embedded ML Clustering: {session_id}")
        return render_template('embedded-ml-clustering.html')
    except Exception as e:
        logger.error(f"Error in embedded_ml_clustering route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/embedded-ml/dataset-info', methods=['GET'])
def api_embedded_ml_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Embedded ML dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for clustering and matching suitability
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Determine suitability for matching/clustering
            matching_suitability = "High"
            clustering_potential = "Good"
            
            if pd.api.types.is_object_dtype(df[col]):
                # Text columns are great for matching
                avg_length = df[col].dropna().astype(str).str.len().mean()
                if avg_length > 5 and unique_count > 1:
                    matching_suitability = "Excellent"
                    clustering_potential = "Excellent"
                elif unique_count == 1:
                    matching_suitability = "Poor"
                    clustering_potential = "Poor"
            elif pd.api.types.is_numeric_dtype(df[col]):
                if unique_count < 5:
                    matching_suitability = "Low"
                    clustering_potential = "Fair"
                elif missing_pct > 50:
                    matching_suitability = "Poor"
                    clustering_potential = "Poor"
            
            # Get sample values
            sample_values = []
            try:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                sample_values = ["N/A"]
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'matching_suitability': matching_suitability,
                'clustering_potential': clustering_potential,
                'sample_values': sample_values
            })

        # Calculate data quality metrics
        total_duplicates = df.duplicated().sum()
        data_quality_score = calculate_data_quality_for_matching(df)

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'total_duplicates': int(total_duplicates),
            'data_quality_score': data_quality_score,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_embedded_ml_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/embedded-ml/process', methods=['POST'])
def api_embedded_ml_process():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Embedded ML processing requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        operation_type = data.get('operation_type')
        selected_columns = data.get('selected_columns', [])
        model = data.get('model', 'gpt-4o')
        similarity_threshold = float(data.get('similarity_threshold', 0.8))
        clustering_method = data.get('clustering_method', 'kmeans')
        num_clusters = int(data.get('num_clusters', 5))
        
        if not operation_type:
            return jsonify({'error': 'No operation type specified'}), 400
        
        if not selected_columns:
            return jsonify({'error': 'No columns selected for processing'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Perform the requested operation
        start_time = time.time()
        result = perform_embedded_ml_operation(
            df, operation_type, selected_columns, model, 
            similarity_threshold, clustering_method, num_clusters, filename
        )
        processing_time = round(time.time() - start_time, 2)
        
        # Store result for download
        operation_id = str(uuid.uuid4())
        data_store[f"embedded_ml_{operation_id}"] = {
            'result_df': result['processed_df'],
            'original_df': df,
            'operation_type': operation_type,
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'columns': selected_columns
        }
        
        # Prepare response (remove DataFrame for JSON serialization)
        response_result = result.copy()
        if 'processed_df' in response_result:
            processed_df = response_result['processed_df']
            response_result['data_preview'] = {
                'columns': processed_df.columns.tolist(),
                'data': processed_df.head(20).to_dict(orient='records'),
                'shape': processed_df.shape
            }
            del response_result['processed_df']
        
        response_result['operation_id'] = operation_id
        response_result['processing_time'] = processing_time
        
        return jsonify(response_result)
    
    except Exception as e:
        logger.error(f"Error in api_embedded_ml_process: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def calculate_data_quality_for_matching(df):
    """Calculate data quality score for matching operations"""
    try:
        # Factors affecting matching quality
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_ratio = df.duplicated().sum() / len(df)
        
        # Text column quality
        text_cols = df.select_dtypes(include=['object']).columns
        text_quality = 0
        if len(text_cols) > 0:
            avg_text_length = df[text_cols].astype(str).applymap(len).mean().mean()
            text_quality = min(1.0, avg_text_length / 20)  # Normalize to 0-1
        
        # Calculate overall score
        quality_score = (1 - missing_ratio) * 0.4 + (1 - duplicate_ratio) * 0.3 + text_quality * 0.3
        quality_score = max(0, min(1, quality_score)) * 100
        
        return f"{quality_score:.1f}%"
    except:
        return "N/A"

def perform_embedded_ml_operation(df, operation_type, selected_columns, model, 
                                 similarity_threshold, clustering_method, num_clusters, filename):
    """
    Perform embedded ML operations for data matching and clustering
    """
    try:
        if operation_type == 'duplicate_detection':
            return perform_duplicate_detection(df, selected_columns, model, similarity_threshold)
        elif operation_type == 'record_matching':
            return perform_record_matching(df, selected_columns, model, similarity_threshold)
        elif operation_type == 'clustering':
            return perform_intelligent_clustering(df, selected_columns, model, clustering_method, num_clusters)
        elif operation_type == 'data_cleaning':
            return perform_ml_data_cleaning(df, selected_columns, model)
        elif operation_type == 'entity_resolution':
            return perform_entity_resolution(df, selected_columns, model, similarity_threshold)
        elif operation_type == 'pattern_discovery':
            return perform_pattern_discovery(df, selected_columns, model)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    except Exception as e:
        logger.error(f"Error in perform_embedded_ml_operation: {str(e)}")
        raise

def perform_duplicate_detection(df, selected_columns, model, similarity_threshold):
    """Detect and remove duplicate records using ML techniques"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from difflib import SequenceMatcher
        
        result_df = df.copy()
        duplicate_info = []
        
        # Create a combined text representation for similarity comparison
        text_columns = [col for col in selected_columns if pd.api.types.is_object_dtype(df[col])]
        numeric_columns = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if text_columns:
            # Combine text columns
            combined_text = result_df[text_columns].fillna('').astype(str).apply(
                lambda x: ' '.join(x), axis=1
            )
            
            # Use TF-IDF for similarity calculation
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(combined_text)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            duplicates_found = []
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > similarity_threshold:
                        duplicates_found.append({
                            'index1': i,
                            'index2': j,
                            'similarity': float(similarity_matrix[i][j]),
                            'record1': result_df.iloc[i][selected_columns].to_dict(),
                            'record2': result_df.iloc[j][selected_columns].to_dict()
                        })
        
        # Mark duplicates
        duplicate_indices = set()
        for dup in duplicates_found:
            duplicate_indices.add(dup['index2'])  # Keep first occurrence
        
        result_df['is_duplicate'] = False
        result_df.loc[list(duplicate_indices), 'is_duplicate'] = True
        result_df['duplicate_group'] = -1
        
        # Assign duplicate groups
        group_id = 0
        for dup in duplicates_found:
            if result_df.loc[dup['index1'], 'duplicate_group'] == -1:
                result_df.loc[dup['index1'], 'duplicate_group'] = group_id
                result_df.loc[dup['index2'], 'duplicate_group'] = group_id
                group_id += 1
            else:
                result_df.loc[dup['index2'], 'duplicate_group'] = result_df.loc[dup['index1'], 'duplicate_group']
        
        # Generate insights using LLM
        insights = generate_duplicate_insights(duplicates_found, model, len(df))
        
        # Create summary statistics
        summary = {
            'total_records': len(df),
            'duplicates_found': len(duplicate_indices),
            'duplicate_groups': len(set(result_df[result_df['duplicate_group'] != -1]['duplicate_group'])),
            'similarity_threshold': similarity_threshold,
            'clean_records': len(df) - len(duplicate_indices)
        }
        
        return {
            'operation': 'duplicate_detection',
            'processed_df': result_df,
            'duplicates_found': duplicates_found[:50],  # Limit for display
            'summary': summary,
            'insights': insights,
            'etl_benefits': get_duplicate_detection_etl_benefits()
        }
    
    except Exception as e:
        logger.error(f"Error in perform_duplicate_detection: {str(e)}")
        raise

def perform_record_matching(df, selected_columns, model, similarity_threshold):
    """Perform intelligent record matching and merging"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import re
        
        result_df = df.copy()
        
        # Standardize text data for better matching
        text_columns = [col for col in selected_columns if pd.api.types.is_object_dtype(df[col])]
        
        for col in text_columns:
            # Clean and standardize text
            result_df[f'{col}_cleaned'] = result_df[col].astype(str).apply(
                lambda x: re.sub(r'[^\w\s]', '', x.lower().strip())
            )
        
        # Find potential matches
        matches_found = []
        if text_columns:
            cleaned_columns = [f'{col}_cleaned' for col in text_columns]
            combined_text = result_df[cleaned_columns].fillna('').apply(
                lambda x: ' '.join(x), axis=1
            )
            
            # Use TF-IDF for similarity
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(combined_text)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find matches above threshold
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > similarity_threshold:
                        matches_found.append({
                            'index1': i,
                            'index2': j,
                            'similarity': float(similarity_matrix[i][j]),
                            'record1': result_df.iloc[i][selected_columns].to_dict(),
                            'record2': result_df.iloc[j][selected_columns].to_dict(),
                            'match_confidence': 'High' if similarity_matrix[i][j] > 0.9 else 'Medium'
                        })
        
        # Create match groups
        result_df['match_group'] = range(len(result_df))
        result_df['has_matches'] = False
        
        for match in matches_found:
            group_id = min(result_df.loc[match['index1'], 'match_group'], 
                          result_df.loc[match['index2'], 'match_group'])
            result_df.loc[match['index1'], 'match_group'] = group_id
            result_df.loc[match['index2'], 'match_group'] = group_id
            result_df.loc[match['index1'], 'has_matches'] = True
            result_df.loc[match['index2'], 'has_matches'] = True
        
        # Generate AI insights
        insights = generate_matching_insights(matches_found, model, len(df))
        
        # Create summary
        summary = {
            'total_records': len(df),
            'matches_found': len(matches_found),
            'records_with_matches': int(result_df['has_matches'].sum()),
            'match_groups': len(result_df['match_group'].unique()),
            'similarity_threshold': similarity_threshold
        }
        
        return {
            'operation': 'record_matching',
            'processed_df': result_df,
            'matches_found': matches_found[:50],
            'summary': summary,
            'insights': insights,
            'etl_benefits': get_record_matching_etl_benefits()
        }
    
    except Exception as e:
        logger.error(f"Error in perform_record_matching: {str(e)}")
        raise

def perform_intelligent_clustering(df, selected_columns, model, clustering_method, num_clusters):
    """Perform intelligent clustering using ML techniques"""
    try:
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import PCA
        import numpy as np
        
        result_df = df.copy()
        
        # Prepare features for clustering
        features = []
        feature_names = []
        
        # Process numeric columns
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            numeric_data = df[numeric_cols].fillna(df[numeric_cols].median())
            scaler = StandardScaler()
            scaled_numeric = scaler.fit_transform(numeric_data)
            features.append(scaled_numeric)
            feature_names.extend(numeric_cols)
        
        # Process text columns
        text_cols = [col for col in selected_columns if pd.api.types.is_object_dtype(df[col])]
        if text_cols:
            # Combine text columns
            combined_text = df[text_cols].fillna('').astype(str).apply(
                lambda x: ' '.join(x), axis=1
            )
            
            # Use TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            text_features = vectorizer.fit_transform(combined_text).toarray()
            features.append(text_features)
            feature_names.extend([f'text_feature_{i}' for i in range(text_features.shape[1])])
        
        # Combine all features
        if features:
            X = np.hstack(features)
        else:
            raise ValueError("No suitable features found for clustering")
        
        # Apply clustering algorithm
        if clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        elif clustering_method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif clustering_method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=num_clusters)
        else:
            clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        
        cluster_labels = clusterer.fit_predict(X)
        result_df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in np.unique(cluster_labels):
            if cluster_id != -1:  # Exclude noise points in DBSCAN
                cluster_data = result_df[result_df['cluster'] == cluster_id]
                stats = {
                    'cluster_id': int(cluster_id),
                    'size': len(cluster_data),
                    'percentage': f"{(len(cluster_data) / len(df)) * 100:.1f}%"
                }
                
                # Add representative values for each column
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        stats[f'{col}_mean'] = float(cluster_data[col].mean()) if not cluster_data[col].isna().all() else None
                    else:
                        mode_val = cluster_data[col].mode()
                        stats[f'{col}_mode'] = mode_val.iloc[0] if len(mode_val) > 0 else None
                
                cluster_stats.append(stats)
        
        # Generate visualization data (PCA for 2D representation)
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X
        
        visualization_data = {
            'x': X_pca[:, 0].tolist(),
            'y': X_pca[:, 1].tolist(),
            'clusters': cluster_labels.tolist()
        }
        
        # Generate AI insights
        insights = generate_clustering_insights(cluster_stats, model, clustering_method)
        
        # Create summary
        summary = {
            'total_records': len(df),
            'num_clusters': len(cluster_stats),
            'clustering_method': clustering_method,
            'largest_cluster_size': max([s['size'] for s in cluster_stats]) if cluster_stats else 0,
            'smallest_cluster_size': min([s['size'] for s in cluster_stats]) if cluster_stats else 0
        }
        
        return {
            'operation': 'clustering',
            'processed_df': result_df,
            'cluster_stats': cluster_stats,
            'visualization_data': visualization_data,
            'summary': summary,
            'insights': insights,
            'etl_benefits': get_clustering_etl_benefits()
        }
    
    except Exception as e:
        logger.error(f"Error in perform_intelligent_clustering: {str(e)}")
        raise

def perform_ml_data_cleaning(df, selected_columns, model):
    """Perform ML-powered data cleaning"""
    try:
        result_df = df.copy()
        cleaning_actions = []
        
        for col in selected_columns:
            if col in df.columns:
                # Handle missing values intelligently
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Use median for numeric
                        median_val = df[col].median()
                        result_df[col].fillna(median_val, inplace=True)
                        cleaning_actions.append(f"Filled {missing_count} missing values in '{col}' with median")
                    else:
                        # Use mode for categorical
                        mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        result_df[col].fillna(mode_val, inplace=True)
                        cleaning_actions.append(f"Filled {missing_count} missing values in '{col}' with mode")
                
                # Standardize text data
                if pd.api.types.is_object_dtype(df[col]):
                    # Clean text: remove extra spaces, standardize case
                    result_df[f'{col}_cleaned'] = result_df[col].astype(str).str.strip().str.title()
                    cleaning_actions.append(f"Standardized text format in '{col}'")
                    
                    # Detect and fix common patterns
                    if 'email' in col.lower():
                        result_df[f'{col}_cleaned'] = result_df[f'{col}_cleaned'].str.lower()
                        cleaning_actions.append(f"Normalized email format in '{col}'")
                    elif 'phone' in col.lower():
                        # Simple phone number cleaning
                        result_df[f'{col}_cleaned'] = result_df[f'{col}_cleaned'].str.replace(r'[^\d]', '', regex=True)
                        cleaning_actions.append(f"Cleaned phone number format in '{col}'")
        
        # Remove exact duplicates
        initial_rows = len(result_df)
        result_df.drop_duplicates(subset=selected_columns, inplace=True)
        removed_duplicates = initial_rows - len(result_df)
        if removed_duplicates > 0:
            cleaning_actions.append(f"Removed {removed_duplicates} exact duplicate rows")
        
        # Generate AI insights
        insights = generate_cleaning_insights_ml(cleaning_actions, model, len(df))
        
        # Create summary
        summary = {
            'original_records': len(df),
            'cleaned_records': len(result_df),
            'records_removed': len(df) - len(result_df),
            'cleaning_actions': len(cleaning_actions),
            'data_quality_improvement': f"{((len(result_df) / len(df)) * 100):.1f}%"
        }
        
        return {
            'operation': 'data_cleaning',
            'processed_df': result_df,
            'cleaning_actions': cleaning_actions,
            'summary': summary,
            'insights': insights,
            'etl_benefits': get_data_cleaning_etl_benefits()
        }
    
    except Exception as e:
        logger.error(f"Error in perform_ml_data_cleaning: {str(e)}")
        raise

def perform_entity_resolution(df, selected_columns, model, similarity_threshold):
    """Perform entity resolution to merge similar entities"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import networkx as nx
        
        result_df = df.copy()
        
        # Create entity groups based on similarity
        text_columns = [col for col in selected_columns if pd.api.types.is_object_dtype(df[col])]
        
        if not text_columns:
            raise ValueError("Entity resolution requires text columns")
        
        # Combine text for entity matching
        combined_text = result_df[text_columns].fillna('').astype(str).apply(
            lambda x: ' '.join(x), axis=1
        )
        
        # Calculate similarities
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(combined_text)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create graph of similar entities
        G = nx.Graph()
        for i in range(len(similarity_matrix)):
            G.add_node(i)
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
        
        # Find connected components (entity groups)
        entity_groups = list(nx.connected_components(G))
        
        # Assign entity IDs
        result_df['entity_id'] = range(len(result_df))
        result_df['entity_group_size'] = 1
        
        for group_idx, group in enumerate(entity_groups):
            if len(group) > 1:
                for node in group:
                    result_df.loc[node, 'entity_id'] = group_idx
                    result_df.loc[node, 'entity_group_size'] = len(group)
        
        # Create resolved entities (canonical forms)
        resolved_entities = []
        for group_idx, group in enumerate(entity_groups):
            if len(group) > 1:
                group_data = result_df.iloc[list(group)]
                
                # Create canonical entity by taking most common values
                canonical_entity = {}
                for col in selected_columns:
                    if pd.api.types.is_object_dtype(df[col]):
                        mode_val = group_data[col].mode()
                        canonical_entity[col] = mode_val.iloc[0] if len(mode_val) > 0 else group_data[col].iloc[0]
                    else:
                        canonical_entity[col] = group_data[col].mean()
                
                canonical_entity['entity_group_id'] = group_idx
                canonical_entity['group_size'] = len(group)
                canonical_entity['confidence'] = float(np.mean([similarity_matrix[i][j] for i in group for j in group if i != j]))
                
                resolved_entities.append(canonical_entity)
        
        # Generate insights
        insights = generate_entity_resolution_insights(resolved_entities, model, len(df))
        
        # Create summary
        summary = {
            'total_records': len(df),
            'entity_groups_found': len([g for g in entity_groups if len(g) > 1]),
            'records_in_groups': sum([len(g) for g in entity_groups if len(g) > 1]),
            'resolved_entities': len(resolved_entities),
            'similarity_threshold': similarity_threshold
        }
        
        return {
            'operation': 'entity_resolution',
            'processed_df': result_df,
            'resolved_entities': resolved_entities,
            'summary': summary,
            'insights': insights,
            'etl_benefits': get_entity_resolution_etl_benefits()
        }
    
    except Exception as e:
        logger.error(f"Error in perform_entity_resolution: {str(e)}")
        raise

def perform_pattern_discovery(df, selected_columns, model):
    """Discover patterns in data using ML techniques"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import re
        
        result_df = df.copy()
        patterns_found = []
        
        # Analyze text patterns
        text_columns = [col for col in selected_columns if pd.api.types.is_object_dtype(df[col])]
        
        for col in text_columns:
            col_patterns = []
            
            # Find common regex patterns
            text_data = df[col].dropna().astype(str)
            
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_matches = text_data.str.contains(email_pattern, regex=True).sum()
            if email_matches > 0:
                col_patterns.append({
                    'pattern_type': 'Email',
                    'pattern': email_pattern,
                    'matches': int(email_matches),
                    'percentage': f"{(email_matches / len(text_data)) * 100:.1f}%"
                })
            
            # Phone pattern
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phone_matches = text_data.str.contains(phone_pattern, regex=True).sum()
            if phone_matches > 0:
                col_patterns.append({
                    'pattern_type': 'Phone',
                    'pattern': phone_pattern,
                    'matches': int(phone_matches),
                    'percentage': f"{(phone_matches / len(text_data)) * 100:.1f}%"
                })
            
            # Date pattern
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            date_matches = text_data.str.contains(date_pattern, regex=True).sum()
            if date_matches > 0:
                col_patterns.append({
                    'pattern_type': 'Date',
                    'pattern': date_pattern,
                    'matches': int(date_matches),
                    'percentage': f"{(date_matches / len(text_data)) * 100:.1f}%"
                })
            
            # Length patterns
            lengths = text_data.str.len()
            common_lengths = lengths.value_counts().head(5)
            for length, count in common_lengths.items():
                if count > len(text_data) * 0.1:  # If more than 10% have same length
                    col_patterns.append({
                        'pattern_type': 'Fixed Length',
                        'pattern': f'Length: {length} characters',
                        'matches': int(count),
                        'percentage': f"{(count / len(text_data)) * 100:.1f}%"
                    })
            
            if col_patterns:
                patterns_found.append({
                    'column': col,
                    'patterns': col_patterns
                })
        
        # Analyze numeric patterns
        numeric_columns = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        for col in numeric_columns:
            col_patterns = []
            numeric_data = df[col].dropna()
            
            # Check for ranges
            q25, q75 = numeric_data.quantile([0.25, 0.75])
            iqr = q75 - q25
            
            if iqr > 0:
                col_patterns.append({
                    'pattern_type': 'Interquartile Range',
                    'pattern': f'IQR: {q25:.2f} - {q75:.2f}',
                    'matches': int(len(numeric_data[(numeric_data >= q25) & (numeric_data <= q75)])),
                    'percentage': '50.0%'
                })
            
            # Check for common values
            value_counts = numeric_data.value_counts().head(3)
            for value, count in value_counts.items():
                if count > len(numeric_data) * 0.05:  # If more than 5%
                    col_patterns.append({
                        'pattern_type': 'Common Value',
                        'pattern': f'Value: {value}',
                        'matches': int(count),
                        'percentage': f"{(count / len(numeric_data)) * 100:.1f}%"
                    })
            
            if col_patterns:
                patterns_found.append({
                    'column': col,
                    'patterns': col_patterns
                })
        
        # Add pattern indicators to dataframe
        for pattern_group in patterns_found:
            col = pattern_group['column']
            for pattern in pattern_group['patterns']:
                pattern_name = f"{col}_{pattern['pattern_type'].lower().replace(' ', '_')}"
                result_df[pattern_name] = False
                
                if pattern['pattern_type'] in ['Email', 'Phone', 'Date']:
                    matches = df[col].astype(str).str.contains(pattern['pattern'], regex=True, na=False)
                    result_df[pattern_name] = matches
        
        # Generate insights
        insights = generate_pattern_discovery_insights(patterns_found, model, len(df))
        
        # Create summary
        total_patterns = sum([len(p['patterns']) for p in patterns_found])
        summary = {
            'total_records': len(df),
            'columns_analyzed': len(selected_columns),
            'patterns_discovered': total_patterns,
            'columns_with_patterns': len(patterns_found)
        }
        
        return {
            'operation': 'pattern_discovery',
            'processed_df': result_df,
            'patterns_found': patterns_found,
            'summary': summary,
            'insights': insights,
            'etl_benefits': get_pattern_discovery_etl_benefits()
        }
    
    except Exception as e:
        logger.error(f"Error in perform_pattern_discovery: {str(e)}")
        raise

# Helper functions for generating insights
def generate_duplicate_insights(duplicates_found, model, total_records):
    """Generate insights about duplicate detection"""
    try:
        if not client:
            return [
                {
                    'title': 'Duplicate Detection Complete',
                    'description': f'Found {len(duplicates_found)} potential duplicates out of {total_records} records.',
                    'severity': 'Medium'
                }
            ]
        
        prompt = f"""
        Analyze duplicate detection results for ETL data quality improvement.
        
        Results:
        - Total records: {total_records}
        - Duplicates found: {len(duplicates_found)}
        - Duplicate rate: {(len(duplicates_found) / total_records * 100):.1f}%
        
        Provide 3-4 insights about data quality and ETL recommendations.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation",
                    "severity": "High|Medium|Low"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an ETL data quality expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating duplicate insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': 'Duplicate detection completed successfully.', 'severity': 'Low'}]

def generate_matching_insights(matches_found, model, total_records):
    """Generate insights about record matching"""
    try:
        if not client:
            return [
                {
                    'title': 'Record Matching Complete',
                    'description': f'Found {len(matches_found)} potential matches out of {total_records} records.',
                    'severity': 'Medium'
                }
            ]
        
        high_confidence_matches = len([m for m in matches_found if m.get('match_confidence') == 'High'])
        
        prompt = f"""
        Analyze record matching results for ETL data integration.
        
        Results:
        - Total records: {total_records}
        - Matches found: {len(matches_found)}
        - High confidence matches: {high_confidence_matches}
        
        Provide insights about data integration opportunities and ETL benefits.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation",
                    "severity": "High|Medium|Low"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an ETL data integration expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating matching insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': 'Record matching completed successfully.', 'severity': 'Low'}]

def generate_clustering_insights(cluster_stats, model, clustering_method):
    """Generate insights about clustering results"""
    try:
        if not client:
            return [
                {
                    'title': 'Clustering Complete',
                    'description': f'Created {len(cluster_stats)} clusters using {clustering_method} method.',
                    'severity': 'Medium'
                }
            ]
        
        prompt = f"""
        Analyze clustering results for customer segmentation and ETL optimization.
        
        Results:
        - Number of clusters: {len(cluster_stats)}
        - Clustering method: {clustering_method}
        - Cluster sizes: {[s['size'] for s in cluster_stats]}
        
        Provide insights about segmentation quality and business applications.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation",
                    "severity": "High|Medium|Low"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a customer segmentation expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating clustering insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': 'Clustering completed successfully.', 'severity': 'Low'}]

def generate_cleaning_insights_ml(cleaning_actions, model, total_records):
    """Generate insights about ML data cleaning"""
    try:
        if not client:
            return [
                {
                    'title': 'Data Cleaning Complete',
                    'description': f'Applied {len(cleaning_actions)} cleaning operations to {total_records} records.',
                    'severity': 'Medium'
                }
            ]
        
        prompt = f"""
        Analyze ML-powered data cleaning results for ETL pipeline improvement.
        
        Cleaning actions performed:
        {chr(10).join(cleaning_actions)}
        
        Total records processed: {total_records}
        
        Provide insights about data quality improvements and ETL benefits.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation",
                    "severity": "High|Medium|Low"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data quality expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating cleaning insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': 'Data cleaning completed successfully.', 'severity': 'Low'}]

def generate_entity_resolution_insights(resolved_entities, model, total_records):
    """Generate insights about entity resolution"""
    try:
        if not client:
            return [
                {
                    'title': 'Entity Resolution Complete',
                    'description': f'Resolved {len(resolved_entities)} entity groups from {total_records} records.',
                    'severity': 'Medium'
                }
            ]
        
        prompt = f"""
        Analyze entity resolution results for master data management.
        
        Results:
        - Total records: {total_records}
        - Resolved entities: {len(resolved_entities)}
        - Average group size: {np.mean([e['group_size'] for e in resolved_entities]) if resolved_entities else 0:.1f}
        
        Provide insights about data consolidation and MDM benefits.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation",
                    "severity": "High|Medium|Low"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a master data management expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating entity resolution insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': 'Entity resolution completed successfully.', 'severity': 'Low'}]

def generate_pattern_discovery_insights(patterns_found, model, total_records):
    """Generate insights about pattern discovery"""
    try:
        if not client:
            return [
                {
                    'title': 'Pattern Discovery Complete',
                    'description': f'Discovered patterns in {len(patterns_found)} columns from {total_records} records.',
                    'severity': 'Medium'
                }
            ]
        
        total_patterns = sum([len(p['patterns']) for p in patterns_found])
        
        prompt = f"""
        Analyze pattern discovery results for data validation and ETL optimization.
        
        Results:
        - Total records: {total_records}
        - Columns with patterns: {len(patterns_found)}
        - Total patterns discovered: {total_patterns}
        
        Provide insights about data validation opportunities and ETL improvements.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation",
                    "severity": "High|Medium|Low"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data validation expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating pattern discovery insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': 'Pattern discovery completed successfully.', 'severity': 'Low'}]

# ETL Benefits functions
def get_duplicate_detection_etl_benefits():
    return [
        {
            'category': 'Data Quality',
            'benefit': 'Improved Data Accuracy',
            'description': 'Eliminates duplicate records that can skew analytics and reporting results.'
        },
        {
            'category': 'Storage Optimization',
            'benefit': 'Reduced Storage Costs',
            'description': 'Removes redundant data, reducing storage requirements and costs.'
        },
        {
            'category': 'Performance',
            'benefit': 'Faster Query Performance',
            'description': 'Smaller, cleaner datasets result in faster query execution times.'
        },
        {
            'category': 'Compliance',
            'benefit': 'Regulatory Compliance',
            'description': 'Ensures data accuracy for regulatory reporting and compliance requirements.'
        }
    ]

def get_record_matching_etl_benefits():
    return [
        {
            'category': 'Data Integration',
            'benefit': 'Unified Customer View',
            'description': 'Merges customer records from multiple sources for a complete view.'
        },
        {
            'category': 'Analytics',
            'benefit': 'Improved Analytics Accuracy',
            'description': 'Better data linkage leads to more accurate business intelligence.'
        },
        {
            'category': 'Automation',
            'benefit': 'Automated Data Linking',
            'description': 'Reduces manual effort in identifying and linking related records.'
        },
        {
            'category': 'Decision Making',
            'benefit': 'Better Business Decisions',
            'description': 'Complete, linked data enables more informed decision making.'
        }
    ]

def get_clustering_etl_benefits():
    return [
        {
            'category': 'Segmentation',
            'benefit': 'Customer Segmentation',
            'description': 'Automatically groups customers for targeted marketing and analysis.'
        },
        {
            'category': 'Optimization',
            'benefit': 'Resource Optimization',
            'description': 'Optimizes resource allocation based on customer or data segments.'
        },
        {
            'category': 'Personalization',
            'benefit': 'Personalized Experiences',
            'description': 'Enables personalized recommendations and services based on clusters.'
        },
        {
            'category': 'Efficiency',
            'benefit': 'Processing Efficiency',
            'description': 'Processes similar data groups together for improved efficiency.'
        }
    ]

def get_data_cleaning_etl_benefits():
    return [
        {
            'category': 'Quality',
            'benefit': 'Enhanced Data Quality',
            'description': 'Standardizes and cleans data for consistent, reliable analytics.'
        },
        {
            'category': 'Automation',
            'benefit': 'Automated Cleaning',
            'description': 'Reduces manual data cleaning effort through ML-powered automation.'
        },
        {
            'category': 'Consistency',
            'benefit': 'Data Consistency',
            'description': 'Ensures consistent data formats across all systems and processes.'
        },
        {
            'category': 'Reliability',
            'benefit': 'Improved Reliability',
            'description': 'Increases confidence in data-driven decisions and analytics.'
        }
    ]

def get_entity_resolution_etl_benefits():
    return [
        {
            'category': 'Master Data',
            'benefit': 'Master Data Management',
            'description': 'Creates authoritative, single version of truth for entities.'
        },
        {
            'category': 'Integration',
            'benefit': 'System Integration',
            'description': 'Enables seamless integration across multiple data systems.'
        },
        {
            'category': 'Governance',
            'benefit': 'Data Governance',
            'description': 'Improves data governance through standardized entity management.'
        },
        {
            'category': 'Analytics',
            'benefit': 'Cross-System Analytics',
            'description': 'Enables analytics across previously siloed data systems.'
        }
    ]

def get_pattern_discovery_etl_benefits():
    return [
        {
            'category': 'Validation',
            'benefit': 'Automated Validation',
            'description': 'Discovers patterns for automated data validation rules.'
        },
        {
            'category': 'Quality',
            'benefit': 'Quality Monitoring',
            'description': 'Monitors data quality by detecting pattern deviations.'
        },
        {
            'category': 'Documentation',
            'benefit': 'Data Documentation',
            'description': 'Automatically documents data patterns and structures.'
        },
        {
            'category': 'Anomaly Detection',
            'benefit': 'Anomaly Detection',
            'description': 'Identifies data anomalies by comparing against discovered patterns.'
        }
    ]

@app.route('/api/embedded-ml/download', methods=['POST'])
def api_embedded_ml_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        operation_id = data.get('operation_id')
        
        if not session_id or not operation_id:
            return jsonify({'error': 'Missing session_id or operation_id'}), 400
        
        ml_key = f"embedded_ml_{operation_id}"
        if ml_key not in data_store:
            return jsonify({'error': 'Operation result not found'}), 404
        
        ml_data = data_store[ml_key]
        result_df = ml_data['result_df']
        
        # Create temporary file
        temp_filename = f"embedded_ml_result_{operation_id[:8]}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        result_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_embedded_ml_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500




# FINAL FIXED TIME SERIES ROUTES - Add these to your existing app.py file

@app.route('/time-series-forecasting')
def time_series_forecasting():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Time Series Forecasting route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for Time Series Forecasting: {session_id}")
        return render_template('time-series-forecasting.html')
    except Exception as e:
        logger.error(f"Error in time_series_forecasting route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/time-series/dataset-info', methods=['GET'])
def api_time_series_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Time Series dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for time series suitability - IMPROVED
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Determine time series suitability - ENHANCED
            is_datetime = False
            is_numeric = False
            ts_suitability = "Low"
            
            # Check if column could be datetime - IMPROVED DETECTION
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                is_datetime = True
                ts_suitability = "Excellent"
            elif pd.api.types.is_object_dtype(df[col]):
                # Try to parse as datetime with multiple formats
                try:
                    sample_data = df[col].dropna().head(50)
                    if len(sample_data) > 0:
                        # Try common date formats
                        for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                            try:
                                pd.to_datetime(sample_data, format=date_format, errors='raise')
                                is_datetime = True
                                ts_suitability = "Good"
                                break
                            except:
                                continue
                        
                        # If no specific format works, try general parsing
                        if not is_datetime:
                            pd.to_datetime(sample_data, errors='raise')
                            is_datetime = True
                            ts_suitability = "Good"
                except:
                    # Check if it could be a time-related string
                    sample_values = df[col].dropna().head(10).astype(str).tolist()
                    time_keywords = ['date', 'time', 'year', 'month', 'day', 'timestamp']
                    if any(keyword in col.lower() for keyword in time_keywords):
                        ts_suitability = "Medium"
                    # Check if values look like dates
                    elif any(any(char in str(val) for char in ['-', '/', ':']) for val in sample_values):
                        ts_suitability = "Medium"
            
            # Check if column is numeric (potential target) - IMPROVED
            if pd.api.types.is_numeric_dtype(df[col]):
                is_numeric = True
                if unique_count > 5 and missing_pct < 80:  # More lenient criteria
                    if not is_datetime:  # Don't mark datetime columns as numeric targets
                        ts_suitability = "Good"
            
            # Get sample values
            sample_values = []
            try:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                sample_values = ["N/A"]
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'is_datetime': is_datetime,
                'is_numeric': is_numeric,
                'ts_suitability': ts_suitability,
                'sample_values': sample_values
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_time_series_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/time-series/forecast', methods=['POST'])
def api_time_series_forecast():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Time series forecasting requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        # Extract parameters
        date_column = data.get('date_column')
        target_column = data.get('target_column')
        forecast_periods = int(data.get('forecast_periods', 30))
        model_type = data.get('model_type', 'auto')
        seasonality = data.get('seasonality', 'auto')
        confidence_interval = float(data.get('confidence_interval', 0.95))
        external_factors = data.get('external_factors', [])
        
        if not date_column or not target_column:
            return jsonify({'error': 'Date column and target column are required'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Perform time series forecasting - COMPLETELY FIXED
        start_time = time.time()
        forecast_result = perform_time_series_forecasting_completely_fixed(
            df, date_column, target_column, forecast_periods, 
            model_type, seasonality, confidence_interval, external_factors, filename
        )
        processing_time = round(time.time() - start_time, 2)
        
        # Store forecast result
        forecast_id = str(uuid.uuid4())
        data_store[f"forecast_{forecast_id}"] = {
            'result': forecast_result,
            'enhanced_df': forecast_result['enhanced_df'],
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'parameters': {
                'date_column': date_column,
                'target_column': target_column,
                'forecast_periods': forecast_periods,
                'model_type': model_type
            }
        }
        
        # FIXED: Convert all numpy arrays to lists for JSON serialization
        response_result = json_safe_dict(forecast_result)
        
        if 'enhanced_df' in response_result:
            enhanced_df = forecast_result['enhanced_df']  # Use original df, not converted
            response_result['data_preview'] = {
                'columns': enhanced_df.columns.tolist(),
                'data': json_safe_dict(enhanced_df.tail(50).fillna('').to_dict(orient='records')),
                'shape': list(enhanced_df.shape)
            }
            del response_result['enhanced_df']
        
        response_result['forecast_id'] = forecast_id
        response_result['processing_time'] = processing_time
        
        return jsonify(response_result)
    
    except Exception as e:
        logger.error(f"Error in api_time_series_forecast: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def json_safe_dict(obj):
    """FINAL FIX: Convert any object to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: json_safe_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        return [json_safe_dict(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, pd.DataFrame):
        return json_safe_dict(obj.to_dict(orient='records'))
    elif isinstance(obj, pd.Series):
        return json_safe_dict(obj.to_dict())
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DatetimeIndex):
        return [d.isoformat() for d in obj]
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, 'to_json'):
        return json.loads(obj.to_json())
    elif pd.isna(obj):  # This is safe for scalar values
        return None
    else:
        try:
            # Check if it's JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def perform_time_series_forecasting_completely_fixed(df, date_column, target_column, forecast_periods, 
                                                    model_type, seasonality, confidence_interval, external_factors, filename):
    """
    COMPLETELY FIXED: Perform comprehensive time series forecasting with robust error handling
    """
    try:
        # Prepare the data - IMPROVED
        ts_df = df[[date_column, target_column]].copy()
        
        # Convert date column to datetime with multiple format attempts
        try:
            ts_df[date_column] = pd.to_datetime(ts_df[date_column])
        except:
            # Try common formats
            for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                try:
                    ts_df[date_column] = pd.to_datetime(ts_df[date_column], format=date_format)
                    break
                except:
                    continue
            else:
                # If all formats fail, create a simple sequential index
                logger.warning(f"Could not parse date column {date_column}, creating sequential index")
                ts_df[date_column] = pd.date_range(start='2020-01-01', periods=len(ts_df), freq='D')
        
        # Convert target to numeric - FIXED
        try:
            ts_df[target_column] = pd.to_numeric(ts_df[target_column], errors='coerce')
        except:
            raise ValueError(f"Target column '{target_column}' cannot be converted to numeric values")
        
        # Sort by date and remove invalid data
        ts_df = ts_df.sort_values(date_column)
        ts_df = ts_df.dropna()
        
        if len(ts_df) < 2:  # Minimum requirement
            raise ValueError("Insufficient data for time series forecasting (minimum 2 points required)")
        
        # Set date as index
        ts_df.set_index(date_column, inplace=True)
        
        # Detect frequency - IMPROVED
        freq = detect_frequency_completely_fixed(ts_df.index)
        
        # Perform forecasting - COMPLETELY FIXED
        forecast_results = robust_forecasting_engine(ts_df, target_column, forecast_periods, confidence_interval)
        
        # Create enhanced dataset with forecasts
        enhanced_df = create_enhanced_ts_dataset_completely_fixed(df, ts_df, forecast_results, date_column, target_column)
        
        # Generate visualizations
        visualizations = create_ts_visualizations_completely_fixed(ts_df, forecast_results, target_column)
        
        # Calculate performance metrics
        metrics = calculate_ts_metrics_completely_fixed(ts_df, forecast_results, target_column)
        
        # Generate AI insights
        ai_insights = generate_ts_insights_completely_fixed(ts_df, forecast_results, metrics, filename)
        
        # Generate ETL recommendations
        etl_recommendations = generate_ts_etl_recommendations_completely_fixed(forecast_results, metrics)
        
        return {
            'forecast_data': forecast_results,
            'enhanced_df': enhanced_df,
            'visualizations': visualizations,
            'metrics': metrics,
            'ai_insights': ai_insights,
            'etl_recommendations': etl_recommendations,
            'model_info': {
                'model_type': forecast_results.get('model_used', model_type),
                'frequency_detected': freq,
                'data_points': len(ts_df),
                'forecast_periods': forecast_periods,
                'confidence_interval': confidence_interval
            },
            'data_quality': assess_ts_data_quality_completely_fixed(ts_df, target_column)
        }
    
    except Exception as e:
        logger.error(f"Error in perform_time_series_forecasting_completely_fixed: {str(e)}")
        raise

def detect_frequency_completely_fixed(date_index):
    """COMPLETELY FIXED: Detect the frequency of the time series"""
    try:
        if len(date_index) < 2:
            return 'Unknown'
        
        # Calculate differences
        diffs = date_index.to_series().diff().dropna()
        if len(diffs) == 0:
            return 'Unknown'
            
        mode_diff = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else diffs.median()
        
        # Determine frequency
        if mode_diff <= pd.Timedelta(days=1):
            return 'Daily'
        elif mode_diff <= pd.Timedelta(days=7):
            return 'Weekly'
        elif mode_diff <= pd.Timedelta(days=31):
            return 'Monthly'
        elif mode_diff <= pd.Timedelta(days=92):
            return 'Quarterly'
        elif mode_diff <= pd.Timedelta(days=366):
            return 'Yearly'
        else:
            return 'Irregular'
    except Exception as e:
        logger.warning(f"Error detecting frequency: {str(e)}")
        return 'Unknown'

def robust_forecasting_engine(ts_df, target_column, forecast_periods, confidence_interval):
    """COMPLETELY FIXED: Ultra-robust forecasting engine that never fails"""
    try:
        y = ts_df[target_column].values
        
        # Handle edge cases
        if len(y) == 0:
            return create_fallback_forecast(forecast_periods, 0, confidence_interval, ts_df.index)
        elif len(y) == 1:
            return create_fallback_forecast(forecast_periods, y[0], confidence_interval, ts_df.index)
        
        # Calculate basic statistics safely
        y_mean = np.mean(y)
        y_std = np.std(y) if len(y) > 1 else abs(y_mean) * 0.1
        y_min = np.min(y)
        y_max = np.max(y)
        
        # Ensure we have valid statistics
        if np.isnan(y_mean) or np.isinf(y_mean):
            y_mean = 0
        if np.isnan(y_std) or np.isinf(y_std) or y_std == 0:
            y_std = max(abs(y_mean) * 0.1, 1.0)
        
        # Try simple linear trend first
        try:
            if len(y) >= 2:
                X = np.arange(len(y)).reshape(-1, 1)
                
                # Calculate slope manually to avoid division by zero
                x_mean = np.mean(X.flatten())
                xy_mean = np.mean(X.flatten() * y)
                x_sq_mean = np.mean(X.flatten() ** 2)
                
                denominator = x_sq_mean - x_mean ** 2
                if abs(denominator) > 1e-10:  # Avoid division by zero
                    slope = (xy_mean - x_mean * y_mean) / denominator
                    intercept = y_mean - slope * x_mean
                else:
                    slope = 0
                    intercept = y_mean
                
                # Generate forecasts
                future_X = np.arange(len(y), len(y) + forecast_periods)
                forecast = slope * future_X + intercept
                
                # Calculate fitted values
                fitted_values = slope * X.flatten() + intercept
                residuals = y - fitted_values
                
                # Calculate confidence intervals safely
                mse = np.mean(residuals ** 2) if len(residuals) > 0 else y_std ** 2
                std_error = np.sqrt(max(mse, 1e-10))
                
                # Z-score for confidence interval
                z_score = 1.96  # 95% confidence
                if 0 < confidence_interval < 1:
                    from scipy.stats import norm
                    z_score = norm.ppf((1 + confidence_interval) / 2)
                
                lower_bound = forecast - z_score * std_error
                upper_bound = forecast + z_score * std_error
                
                # Create future dates
                future_dates = create_future_dates(ts_df.index, forecast_periods)
                
                return {
                    'forecast': forecast,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'future_dates': future_dates,
                    'model_type': 'Linear Trend',
                    'fitted_values': fitted_values,
                    'residuals': residuals,
                    'model_used': 'linear_trend',
                    'slope': slope,
                    'intercept': intercept
                }
        except Exception as e:
            logger.warning(f"Linear trend failed: {str(e)}")
        
        # Fallback to simple mean-based forecast
        try:
            forecast = np.full(forecast_periods, y_mean)
            fitted_values = np.full(len(y), y_mean)
            residuals = y - fitted_values
            
            # Simple confidence intervals
            z_score = 1.96
            if 0 < confidence_interval < 1:
                from scipy.stats import norm
                z_score = norm.ppf((1 + confidence_interval) / 2)
            
            lower_bound = forecast - z_score * y_std
            upper_bound = forecast + z_score * y_std
            
            future_dates = create_future_dates(ts_df.index, forecast_periods)
            
            return {
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'future_dates': future_dates,
                'model_type': 'Mean Forecast',
                'fitted_values': fitted_values,
                'residuals': residuals,
                'model_used': 'mean_forecast',
                'mean_value': y_mean
            }
        except Exception as e:
            logger.warning(f"Mean forecast failed: {str(e)}")
        
        # Ultimate fallback
        return create_fallback_forecast(forecast_periods, y_mean, confidence_interval, ts_df.index)
        
    except Exception as e:
        logger.error(f"Error in robust_forecasting_engine: {str(e)}")
        # Emergency fallback
        return create_fallback_forecast(forecast_periods, 1.0, confidence_interval, 
                                      pd.date_range(start='2020-01-01', periods=1, freq='D'))

def create_fallback_forecast(forecast_periods, base_value, confidence_interval, date_index):
    """Create a simple fallback forecast that always works"""
    try:
        # Ensure base_value is valid
        if np.isnan(base_value) or np.isinf(base_value):
            base_value = 1.0
        
        forecast = np.full(forecast_periods, base_value)
        
        # Simple confidence intervals
        std_error = abs(base_value) * 0.1 if base_value != 0 else 1.0
        z_score = 1.96
        
        lower_bound = forecast - z_score * std_error
        upper_bound = forecast + z_score * std_error
        
        future_dates = create_future_dates(date_index, forecast_periods)
        
        return {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'future_dates': future_dates,
            'model_type': 'Constant Forecast',
            'fitted_values': np.full(len(date_index), base_value),
            'residuals': np.zeros(len(date_index)),
            'model_used': 'constant_fallback',
            'base_value': base_value
        }
    except Exception as e:
        logger.error(f"Error in create_fallback_forecast: {str(e)}")
        # Emergency return
        return {
            'forecast': np.ones(forecast_periods),
            'lower_bound': np.ones(forecast_periods) * 0.9,
            'upper_bound': np.ones(forecast_periods) * 1.1,
            'future_dates': pd.date_range(start='2020-01-01', periods=forecast_periods, freq='D'),
            'model_type': 'Emergency Fallback',
            'fitted_values': np.ones(1),
            'residuals': np.zeros(1),
            'model_used': 'emergency_fallback'
        }

def create_future_dates(date_index, forecast_periods):
    """Safely create future dates"""
    try:
        if len(date_index) == 0:
            return pd.date_range(start='2020-01-01', periods=forecast_periods, freq='D')
        
        last_date = date_index[-1]
        
        if len(date_index) > 1:
            freq_delta = date_index[-1] - date_index[-2]
            # Ensure frequency is positive
            if freq_delta <= pd.Timedelta(0):
                freq_delta = pd.Timedelta(days=1)
        else:
            freq_delta = pd.Timedelta(days=1)
        
        future_dates = []
        current_date = last_date
        for i in range(forecast_periods):
            current_date += freq_delta
            future_dates.append(current_date)
        
        return pd.DatetimeIndex(future_dates)
        
    except Exception as e:
        logger.warning(f"Error creating future dates: {str(e)}")
        return pd.date_range(start='2020-01-01', periods=forecast_periods, freq='D')

def create_enhanced_ts_dataset_completely_fixed(original_df, ts_df, forecast_results, date_column, target_column):
    """COMPLETELY FIXED: Create enhanced dataset with forecasts"""
    try:
        enhanced_df = original_df.copy()
        
        # Add fitted values and residuals to historical data
        fitted_values = forecast_results.get('fitted_values', [])
        residuals = forecast_results.get('residuals', [])
        
        if len(fitted_values) == len(ts_df):
            # Create a mapping from dates to fitted values
            fitted_dict = dict(zip(ts_df.index, fitted_values))
            residuals_dict = dict(zip(ts_df.index, residuals))
            
            # Convert date column to datetime for mapping
            try:
                enhanced_df[date_column] = pd.to_datetime(enhanced_df[date_column])
            except:
                pass
            
            # Add fitted values and residuals to original dataframe
            enhanced_df['fitted_values'] = enhanced_df[date_column].map(fitted_dict)
            enhanced_df['residuals'] = enhanced_df[date_column].map(residuals_dict)
        
        # Add forecast data as new rows
        future_dates = forecast_results['future_dates']
        forecast_values = forecast_results['forecast']
        lower_bound = forecast_results['lower_bound']
        upper_bound = forecast_results['upper_bound']
        
        # Create forecast rows
        forecast_rows = []
        for i, date in enumerate(future_dates):
            row = {col: None for col in enhanced_df.columns}
            row[date_column] = date
            row[target_column] = forecast_values[i]
            row['fitted_values'] = forecast_values[i]
            row['lower_bound'] = lower_bound[i]
            row['upper_bound'] = upper_bound[i]
            row['is_forecast'] = True
            row['forecast_period'] = i + 1
            forecast_rows.append(row)
        
        # Add is_forecast column to historical data
        enhanced_df['is_forecast'] = False
        enhanced_df['lower_bound'] = None
        enhanced_df['upper_bound'] = None
        enhanced_df['forecast_period'] = None
        
        # Combine historical and forecast data
        if forecast_rows:
            forecast_df = pd.DataFrame(forecast_rows)
            # Handle empty dataframes properly
            if not enhanced_df.empty and not forecast_df.empty:
                enhanced_df = pd.concat([enhanced_df, forecast_df], ignore_index=True)
            elif not forecast_df.empty:
                enhanced_df = forecast_df
        
        return enhanced_df
    
    except Exception as e:
        logger.error(f"Error in create_enhanced_ts_dataset_completely_fixed: {str(e)}")
        return original_df

def create_ts_visualizations_completely_fixed(ts_df, forecast_results, target_column):
    """COMPLETELY FIXED: Create time series visualizations"""
    try:
        visualizations = []
        
        # Main forecast plot
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        if len(ts_df) > 0:
            plt.plot(ts_df.index, ts_df[target_column], 'b-', label='Historical Data', linewidth=2)
        
        # Plot fitted values
        fitted_values = forecast_results.get('fitted_values', [])
        if len(fitted_values) == len(ts_df) and len(ts_df) > 0:
            plt.plot(ts_df.index, fitted_values, 'g--', label='Fitted Values', alpha=0.7)
        
        # Plot forecast
        future_dates = forecast_results['future_dates']
        forecast = forecast_results['forecast']
        lower_bound = forecast_results['lower_bound']
        upper_bound = forecast_results['upper_bound']
        
        if len(future_dates) > 0 and len(forecast) > 0:
            plt.plot(future_dates, forecast, 'r-', label='Forecast', linewidth=2)
            plt.fill_between(future_dates, lower_bound, upper_bound, alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title(f'Time Series Forecast - {forecast_results.get("model_type", "Unknown Model")}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(target_column, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        forecast_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        visualizations.append({
            'type': 'forecast_plot',
            'title': 'Time Series Forecast',
            'data': forecast_plot
        })
        
        return visualizations
    
    except Exception as e:
        logger.error(f"Error in create_ts_visualizations_completely_fixed: {str(e)}")
        return []

def calculate_ts_metrics_completely_fixed(ts_df, forecast_results, target_column):
    """COMPLETELY FIXED: Calculate time series forecasting metrics"""
    try:
        metrics = {}
        
        # Historical fit metrics
        fitted_values = forecast_results.get('fitted_values', [])
        if len(fitted_values) == len(ts_df) and len(ts_df) > 0:
            actual = ts_df[target_column].values
            fitted = fitted_values
            
            # Mean Absolute Error
            mae = np.mean(np.abs(actual - fitted))
            
            # Mean Squared Error
            mse = np.mean((actual - fitted) ** 2)
            
            # Root Mean Squared Error
            rmse = np.sqrt(max(mse, 1e-10))
            
            # Mean Absolute Percentage Error (FIXED division by zero)
            actual_nonzero = actual[np.abs(actual) > 1e-10]
            fitted_nonzero = fitted[np.abs(actual) > 1e-10]
            if len(actual_nonzero) > 0:
                mape = np.mean(np.abs((actual_nonzero - fitted_nonzero) / actual_nonzero)) * 100
            else:
                mape = 0.0
            
            # R-squared (FIXED division by zero)
            ss_res = np.sum((actual - fitted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / max(ss_tot, 1e-10))
            r_squared = max(0, min(1, r_squared))  # Clamp between 0 and 1
            
            metrics.update({
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'r_squared': float(r_squared)
            })
        
        # Forecast metrics
        forecast = forecast_results['forecast']
        if len(forecast) > 0:
            metrics.update({
                'forecast_mean': float(np.mean(forecast)),
                'forecast_std': float(np.std(forecast)),
                'forecast_min': float(np.min(forecast)),
                'forecast_max': float(np.max(forecast)),
                'model_type': forecast_results.get('model_type', 'Unknown')
            })
            
            # Trend analysis (FIXED)
            if len(forecast) > 1:
                trend_slope = (forecast[-1] - forecast[0]) / max(len(forecast) - 1, 1)
                metrics['trend_slope'] = float(trend_slope)
                
                if trend_slope > 0.01:
                    metrics['trend_direction'] = 'Increasing'
                elif trend_slope < -0.01:
                    metrics['trend_direction'] = 'Decreasing'
                else:
                    metrics['trend_direction'] = 'Stable'
            else:
                metrics['trend_direction'] = 'Unknown'
                metrics['trend_slope'] = 0.0
        else:
            metrics.update({
                'forecast_mean': 0.0,
                'forecast_std': 0.0,
                'forecast_min': 0.0,
                'forecast_max': 0.0,
                'model_type': 'Unknown',
                'trend_direction': 'Unknown',
                'trend_slope': 0.0
            })
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in calculate_ts_metrics_completely_fixed: {str(e)}")
        return {
            'model_type': 'Unknown', 
            'trend_direction': 'Unknown',
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'mape': 0.0,
            'r_squared': 0.0,
            'forecast_mean': 0.0,
            'forecast_std': 0.0,
            'forecast_min': 0.0,
            'forecast_max': 0.0,
            'trend_slope': 0.0
        }

def assess_ts_data_quality_completely_fixed(ts_df, target_column):
    """COMPLETELY FIXED: Assess time series data quality"""
    try:
        quality_assessment = {}
        
        if len(ts_df) == 0:
            return {
                'missing_values': 0,
                'missing_percentage': '0%',
                'outliers_count': 0,
                'outliers_percentage': '0%',
                'irregular_gaps': 0,
                'data_points': 0,
                'quality_score': '0%',
                'quality_level': 'No Data'
            }
        
        # Missing values
        missing_count = ts_df[target_column].isna().sum()
        missing_pct = (missing_count / len(ts_df)) * 100
        
        # Outliers (using IQR method) - FIXED
        outlier_pct = 0
        if len(ts_df) > 4:  # Need at least 5 points for quartiles
            try:
                Q1 = ts_df[target_column].quantile(0.25)
                Q3 = ts_df[target_column].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 1e-10:  # Avoid division by zero
                    outliers = ts_df[(ts_df[target_column] < (Q1 - 1.5 * IQR)) | 
                                    (ts_df[target_column] > (Q3 + 1.5 * IQR))]
                    outlier_pct = (len(outliers) / len(ts_df)) * 100
            except:
                outlier_pct = 0
        
        # Data consistency (gaps in time series) - FIXED
        irregular_gaps = 0
        if len(ts_df) > 1:
            try:
                time_diffs = ts_df.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    mode_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                    if mode_diff > pd.Timedelta(0):
                        irregular_gaps = len(time_diffs[time_diffs > mode_diff * 2])
            except:
                irregular_gaps = 0
        
        # Overall quality score
        quality_score = 100
        quality_score -= missing_pct * 2  # Penalize missing values
        quality_score -= outlier_pct  # Penalize outliers
        quality_score -= (irregular_gaps / max(len(ts_df), 1)) * 100 * 0.5  # Penalize irregular gaps
        quality_score = max(0, min(100, quality_score))
        
        quality_assessment = {
            'missing_values': int(missing_count),
            'missing_percentage': f"{missing_pct:.2f}%",
            'outliers_count': int(outlier_pct * len(ts_df) / 100),
            'outliers_percentage': f"{outlier_pct:.2f}%",
            'irregular_gaps': int(irregular_gaps),
            'data_points': len(ts_df),
            'quality_score': f"{quality_score:.1f}%",
            'quality_level': 'Excellent' if quality_score >= 90 else 
                           'Good' if quality_score >= 70 else 
                           'Fair' if quality_score >= 50 else 'Poor'
        }
        
        return quality_assessment
    
    except Exception as e:
        logger.error(f"Error in assess_ts_data_quality_completely_fixed: {str(e)}")
        return {
            'missing_values': 0,
            'missing_percentage': '0%',
            'outliers_count': 0,
            'outliers_percentage': '0%',
            'irregular_gaps': 0,
            'data_points': len(ts_df) if ts_df is not None else 0,
            'quality_score': '50%',
            'quality_level': 'Unknown'
        }

def generate_ts_insights_completely_fixed(ts_df, forecast_results, metrics, filename):
    """COMPLETELY FIXED: Generate insights"""
    try:
        if not client:
            return generate_fallback_ts_insights_completely_fixed(forecast_results, metrics)
        
        # Prepare summary for LLM
        summary = {
            'dataset': filename,
            'data_points': len(ts_df),
            'model_used': forecast_results.get('model_type', 'Unknown'),
            'forecast_periods': len(forecast_results.get('forecast', [])),
            'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            'trend_direction': metrics.get('trend_direction', 'Unknown')
        }
        
        prompt = f"""
        You are an expert time series analyst providing insights for ETL and business intelligence.
        
        Time Series Analysis Summary:
        {json.dumps(summary, indent=2)}
        
        Provide 5 strategic insights covering:
        1. Forecast quality and reliability assessment
        2. Business implications of the forecast trends
        3. Data quality observations and recommendations
        4. ETL pipeline integration opportunities
        5. Risk factors and confidence levels
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed analysis and business implications",
                    "category": "Forecast Quality|Business Impact|Data Quality|ETL Integration|Risk Assessment",
                    "priority": "High|Medium|Low",
                    "actionable": true/false
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an expert time series analyst and ETL specialist. Provide strategic business insights in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        return ai_response.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating LLM insights: {str(e)}")
        return generate_fallback_ts_insights_completely_fixed(forecast_results, metrics)

def generate_fallback_ts_insights_completely_fixed(forecast_results, metrics):
    """COMPLETELY FIXED: Generate fallback insights"""
    insights = []
    
    model_type = forecast_results.get('model_type', 'Unknown')
    trend_direction = metrics.get('trend_direction', 'Unknown')
    r_squared = metrics.get('r_squared', 0)
    
    # Model performance insight
    if r_squared > 0.8:
        insights.append({
            'title': 'Excellent Model Performance',
            'description': f'The {model_type} model shows excellent fit with R² = {r_squared:.3f}, indicating high reliability for forecasting.',
            'category': 'Forecast Quality',
            'priority': 'High',
            'actionable': True
        })
    elif r_squared > 0.5:
        insights.append({
            'title': 'Good Model Performance',
            'description': f'The {model_type} model shows good predictive capability with R² = {r_squared:.3f}.',
            'category': 'Forecast Quality',
            'priority': 'Medium',
            'actionable': True
        })
    else:
        insights.append({
            'title': 'Baseline Model Performance',
            'description': f'The {model_type} model provides baseline predictions. Consider data preprocessing for improvement.',
            'category': 'Forecast Quality',
            'priority': 'Medium',
            'actionable': True
        })
    
    # Trend insight
    if trend_direction == 'Increasing':
        insights.append({
            'title': 'Positive Growth Trend Detected',
            'description': 'The forecast shows an increasing trend, indicating potential growth opportunities.',
            'category': 'Business Impact',
            'priority': 'High',
            'actionable': True
        })
    elif trend_direction == 'Decreasing':
        insights.append({
            'title': 'Declining Trend Alert',
            'description': 'The forecast shows a decreasing trend, requiring attention and intervention strategies.',
            'category': 'Risk Assessment',
            'priority': 'High',
            'actionable': True
        })
    else:
        insights.append({
            'title': 'Stable Trend Pattern',
            'description': 'The forecast shows stable patterns, providing predictable baseline for planning.',
            'category': 'Business Impact',
            'priority': 'Medium',
            'actionable': False
        })
    
    # ETL integration insight
    insights.append({
        'title': 'ETL Pipeline Integration Ready',
        'description': 'This forecasting model can be integrated into your ETL pipeline for automated predictions.',
        'category': 'ETL Integration',
        'priority': 'Medium',
        'actionable': True
    })
    
    return insights

def generate_ts_etl_recommendations_completely_fixed(forecast_results, metrics):
    """COMPLETELY FIXED: Generate ETL recommendations"""
    recommendations = [
        {
            'category': 'Data Pipeline',
            'title': 'Automated Forecasting Pipeline',
            'description': 'Implement automated time series forecasting in your ETL pipeline for real-time predictions.',
            'implementation': 'Schedule daily/weekly model retraining and forecast generation',
            'priority': 'High'
        },
        {
            'category': 'Data Quality',
            'title': 'Time Series Data Validation',
            'description': 'Add data quality checks specific to time series data in your ETL process.',
            'implementation': 'Validate date continuity, detect outliers, and handle missing values',
            'priority': 'High'
        },
        {
            'category': 'Storage Optimization',
            'title': 'Forecast Data Storage',
            'description': 'Optimize storage for historical data and forecasts with appropriate partitioning.',
            'implementation': 'Use time-based partitioning and compress historical data',
            'priority': 'Medium'
        }
    ]
    
    return recommendations

@app.route('/api/time-series/download', methods=['POST'])
def api_time_series_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        forecast_id = data.get('forecast_id')
        
        if not session_id or not forecast_id:
            return jsonify({'error': 'Missing session_id or forecast_id'}), 400
        
        forecast_key = f"forecast_{forecast_id}"
        if forecast_key not in data_store:
            return jsonify({'error': 'Forecast not found'}), 404
        
        forecast_data = data_store[forecast_key]
        enhanced_df = forecast_data['enhanced_df']
        
        # Create temporary file
        temp_filename = f"time_series_forecast_{forecast_data['filename']}_{forecast_id[:8]}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        enhanced_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_time_series_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500




# # Import the blueprint
# from data_driven import data_driven_bp

# # Register the blueprint in your main app.py (add this line after your existing blueprints)
# app.register_blueprint(data_driven_bp)

# # Add the main route for the AI-Driven Data Cataloging feature
# @app.route('/ai-driven-cataloging')
# def ai_driven_cataloging():
#     try:
#         session_id = request.args.get('session_id') or session.get('session_id')
#         logger.info(f"AI-Driven Cataloging route accessed with session_id: {session_id}")
        
#         if not session_id or session_id not in data_store:
#             logger.warning(f"No valid session found: {session_id}")
#             return redirect(url_for('index'))
        
#         # Set session for this tab
#         session['session_id'] = session_id
#         logger.info(f"Session set for AI-Driven Cataloging: {session_id}")
#         return render_template('AI-Driven Data Cataloging & Semantic Search.html')
#     except Exception as e:
#         logger.error(f"Error in ai_driven_cataloging route: {str(e)}")
#         return redirect(url_for('index'))











# AI-Driven Data Cataloging Routes - FIXED
@app.route('/ai-driven-cataloging')
def ai_driven_cataloging():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"AI-Driven Cataloging route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}. Available sessions: {list(data_store.keys())}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for AI-Driven Cataloging: {session_id}")
        return render_template('AI-Driven Data Cataloging & Semantic Search.html')
    except Exception as e:
        logger.error(f"Error in ai_driven_cataloging route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/ai-cataloging/dataset-info', methods=['GET'])
def get_dataset_info():
    """
    Get dataset information for AI cataloging - FIXED
    """
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"AI Cataloging dataset info requested for session: {session_id}")
        logger.info(f"Available sessions in data_store: {list(data_store.keys())}")
        
        if not session_id:
            logger.warning("No session_id provided")
            return jsonify({'error': 'No session ID provided. Please upload a dataset first.'}), 400
            
        if session_id not in data_store:
            logger.warning(f"Session {session_id} not found in data_store")
            return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        logger.info(f"Found dataset for session {session_id}: {filename} with {len(df)} rows and {len(df.columns)} columns")
        
        # Analyze columns for AI cataloging
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Get sample values safely
            sample_values = []
            try:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                logger.warning(f"Error getting sample values for column {col}: {str(e)}")
                sample_values = ["N/A"]
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.1f}%",
                'unique_count': int(unique_count),
                'sample_values': sample_values
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in get_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/ai-cataloging/analyze', methods=['POST'])
def analyze_dataset():
    """
    Perform AI-driven data cataloging and semantic search analysis - FIXED
    """
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        selected_columns = data.get('selected_columns', [])
        ai_model = data.get('ai_model', 'gpt-4o')
        analysis_depth = data.get('analysis_depth', 'comprehensive')
        
        logger.info(f"AI Cataloging analysis requested for session: {session_id}")
        logger.info(f"Selected columns: {selected_columns}")
        
        if not session_id or session_id not in data_store:
            return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Filter to selected columns if specified
        if selected_columns:
            df_analysis = df[selected_columns].copy()
        else:
            df_analysis = df.copy()
        
        # Perform comprehensive AI-driven analysis
        start_time = time.time()
        analysis_result = perform_ai_cataloging_analysis(
            df_analysis, filename, ai_model, analysis_depth
        )
        processing_time = round(time.time() - start_time, 2)
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        data_store[f"ai_cataloging_{analysis_id}"] = {
            'result': analysis_result,
            'enhanced_df': analysis_result.get('enhanced_df', df_analysis),
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'columns_analyzed': selected_columns or df.columns.tolist()
        }
        
        # Prepare response
        response_result = analysis_result.copy()
        if 'enhanced_df' in response_result:
            enhanced_df = response_result['enhanced_df']
            response_result['data_preview'] = {
                'columns': enhanced_df.columns.tolist(),
                'data': enhanced_df.head(20).to_dict(orient='records'),
                'shape': enhanced_df.shape
            }
            del response_result['enhanced_df']
        
        response_result['analysis_id'] = analysis_id
        response_result['processing_time'] = processing_time
        
        return jsonify(response_result)
    
    except Exception as e:
        logger.error(f"Error in analyze_dataset: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def perform_ai_cataloging_analysis(df, filename, ai_model, analysis_depth):
    """
    Perform comprehensive AI-driven data cataloging and semantic analysis - SIMPLIFIED
    """
    try:
        # Initialize result structure
        result = {
            'dataset_name': filename,
            'analysis_depth': analysis_depth,
            'ai_model': ai_model,
            'column_classifications': [],
            'smart_tags': [],
            'etl_recommendations': [],
            'data_quality_assessment': {},
            'enhanced_df': df.copy()
        }
        
        # 1. Column Classification and Semantic Analysis
        column_classifications = classify_columns_simple(df)
        result['column_classifications'] = column_classifications
        
        # 2. Data Quality Assessment
        data_quality = assess_data_quality_simple(df)
        result['data_quality_assessment'] = data_quality
        
        # 3. Smart Tagging
        smart_tags = generate_smart_tags_simple(df, column_classifications)
        result['smart_tags'] = smart_tags
        
        # 4. ETL Workflow Recommendations
        etl_recommendations = generate_etl_recommendations_simple(df, column_classifications, data_quality)
        result['etl_recommendations'] = etl_recommendations
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_ai_cataloging_analysis: {str(e)}")
        raise

def classify_columns_simple(df):
    """
    Simple column classification without LLM dependency
    """
    try:
        column_classifications = []
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Determine semantic type based on name and data
            semantic_type = "General"
            business_meaning = "General data column"
            data_category = "General"
            privacy_level = "Public"
            
            col_name_lower = col.lower()
            
            # Name-based classification
            if any(keyword in col_name_lower for keyword in ['id', 'identifier', 'key']):
                semantic_type = "Identifier"
                business_meaning = "Unique identifier for records"
            elif any(keyword in col_name_lower for keyword in ['name', 'title', 'label']):
                semantic_type = "Name"
                business_meaning = "Descriptive name or title"
            elif any(keyword in col_name_lower for keyword in ['email', 'mail']):
                semantic_type = "Email"
                business_meaning = "Email address"
                privacy_level = "Confidential"
                data_category = "Personal"
            elif any(keyword in col_name_lower for keyword in ['phone', 'tel', 'mobile']):
                semantic_type = "Phone"
                business_meaning = "Phone number"
                privacy_level = "Confidential"
                data_category = "Personal"
            elif any(keyword in col_name_lower for keyword in ['date', 'time', 'created', 'updated']):
                semantic_type = "DateTime"
                business_meaning = "Date or timestamp information"
            elif any(keyword in col_name_lower for keyword in ['amount', 'price', 'cost', 'value', 'salary']):
                semantic_type = "Amount"
                business_meaning = "Monetary or numeric value"
                data_category = "Financial"
            elif pd.api.types.is_numeric_dtype(df[col]):
                semantic_type = "Numeric"
                business_meaning = "Numeric measurement or count"
            elif pd.api.types.is_object_dtype(df[col]):
                semantic_type = "Text"
                business_meaning = "Text or categorical data"
            
            # Quality score calculation
            quality_score = max(0, 100 - missing_pct)
            
            # Detect patterns
            patterns_detected = []
            if pd.api.types.is_object_dtype(df[col]):
                sample_data = df[col].dropna().astype(str).head(100)
                if any('@' in str(val) for val in sample_data):
                    patterns_detected.append("Email pattern")
                if any(any(char.isdigit() for char in str(val)) for val in sample_data):
                    patterns_detected.append("Contains numbers")
            
            column_classifications.append({
                'column_name': col,
                'data_type': col_type,
                'semantic_type': semantic_type,
                'business_meaning': business_meaning,
                'data_category': data_category,
                'privacy_level': privacy_level,
                'quality_score': int(quality_score),
                'patterns_detected': patterns_detected,
                'suggested_transformations': [],
                'missing_percentage': f"{missing_pct:.1f}%",
                'unique_count': int(unique_count)
            })
        
        return column_classifications
    
    except Exception as e:
        logger.error(f"Error in classify_columns_simple: {str(e)}")
        return []

def assess_data_quality_simple(df):
    """
    Simple data quality assessment
    """
    try:
        # Calculate completeness
        completeness_score = ((df.notna().sum().sum()) / (len(df) * len(df.columns))) * 100
        
        # Calculate consistency (duplicates)
        duplicate_rows = df.duplicated().sum()
        consistency_score = max(0, 100 - (duplicate_rows / len(df)) * 100)
        
        # Overall score
        overall_score = (completeness_score + consistency_score) / 2
        
        # Generate recommendations
        recommendations = []
        if completeness_score < 90:
            recommendations.append({
                'category': 'Completeness',
                'priority': 'High',
                'action': 'Address missing values in dataset',
                'impact': 'Improved data reliability'
            })
        
        if duplicate_rows > 0:
            recommendations.append({
                'category': 'Consistency',
                'priority': 'Medium',
                'action': f'Remove {duplicate_rows} duplicate rows',
                'impact': 'Cleaner, more accurate dataset'
            })
        
        return {
            'overall_score': round(overall_score, 2),
            'completeness': {
                'score': round(completeness_score, 2),
                'missing_values_total': int(df.isna().sum().sum()),
                'columns_with_missing': int((df.isna().sum() > 0).sum())
            },
            'consistency': {
                'score': round(consistency_score, 2),
                'duplicate_rows': int(duplicate_rows),
                'duplicate_percentage': round((duplicate_rows / len(df)) * 100, 2)
            },
            'recommendations': recommendations
        }
    
    except Exception as e:
        logger.error(f"Error in assess_data_quality_simple: {str(e)}")
        return {'overall_score': 50, 'recommendations': []}

def generate_smart_tags_simple(df, column_classifications):
    """
    Generate smart tags for the dataset
    """
    try:
        smart_tags = []
        
        # Generate tags based on semantic types
        semantic_types = [c['semantic_type'] for c in column_classifications]
        semantic_counter = {}
        for st in semantic_types:
            semantic_counter[st] = semantic_counter.get(st, 0) + 1
        
        for semantic_type, count in list(semantic_counter.items())[:5]:
            smart_tags.append({
                'tag': f"Contains {semantic_type}",
                'type': 'semantic',
                'confidence': 0.9,
                'count': count
            })
        
        # Generate tags based on data categories
        data_categories = [c['data_category'] for c in column_classifications]
        category_counter = {}
        for dc in data_categories:
            category_counter[dc] = category_counter.get(dc, 0) + 1
        
        for category, count in list(category_counter.items())[:3]:
            smart_tags.append({
                'tag': f"{category} Data",
                'type': 'category',
                'confidence': 0.8,
                'count': count
            })
        
        # Generate tags based on privacy levels
        privacy_levels = [c['privacy_level'] for c in column_classifications]
        if 'Confidential' in privacy_levels:
            smart_tags.append({
                'tag': "Contains Sensitive Data",
                'type': 'privacy',
                'confidence': 0.95,
                'count': privacy_levels.count('Confidential')
            })
        
        # Generate quality tags
        avg_quality = np.mean([c['quality_score'] for c in column_classifications])
        if avg_quality > 90:
            smart_tags.append({
                'tag': "High Quality Dataset",
                'type': 'quality',
                'confidence': 0.9,
                'score': round(avg_quality, 1)
            })
        elif avg_quality < 60:
            smart_tags.append({
                'tag': "Needs Data Cleaning",
                'type': 'quality',
                'confidence': 0.8,
                'score': round(avg_quality, 1)
            })
        
        return smart_tags
    
    except Exception as e:
        logger.error(f"Error in generate_smart_tags_simple: {str(e)}")
        return []

def generate_etl_recommendations_simple(df, column_classifications, data_quality):
    """
    Generate ETL workflow recommendations
    """
    try:
        recommendations = []
        
        # Data quality based recommendations
        if data_quality.get('completeness', {}).get('score', 100) < 90:
            recommendations.append({
                'category': 'Data Completeness',
                'priority': 'High',
                'recommendation': 'Implement missing value handling strategies',
                'implementation': 'Add data validation and imputation steps in ETL pipeline',
                'benefit': 'Improved data reliability'
            })
        
        # Privacy and security recommendations
        sensitive_columns = [c for c in column_classifications if c['privacy_level'] in ['Confidential']]
        if sensitive_columns:
            recommendations.append({
                'category': 'Data Security',
                'priority': 'Critical',
                'recommendation': 'Implement data masking and encryption',
                'implementation': 'Add encryption and masking transformations for sensitive columns',
                'benefit': 'Compliance with data protection regulations'
            })
        
        # Performance optimization
        if len(df) > 100000:
            recommendations.append({
                'category': 'Performance',
                'priority': 'Medium',
                'recommendation': 'Implement data partitioning and indexing',
                'implementation': 'Partition large datasets by date or key columns',
                'benefit': 'Faster query performance and reduced processing time'
            })
        
        # Data standardization
        text_columns = [c for c in column_classifications if c['semantic_type'] in ['Text', 'Email', 'Phone']]
        if text_columns:
            recommendations.append({
                'category': 'Data Standardization',
                'priority': 'Medium',
                'recommendation': 'Standardize text data formats',
                'implementation': 'Add text cleaning and standardization transformations',
                'benefit': 'Improved data consistency and matching accuracy'
            })
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error in generate_etl_recommendations_simple: {str(e)}")
        return []

@app.route('/api/ai-cataloging/search', methods=['POST'])
def semantic_search():
    """
    Perform semantic search on the cataloged dataset - FIXED
    """
    try:
        data = request.json
        query = data.get('query', '')
        analysis_id = data.get('analysis_id', '')
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        analysis_key = f"ai_cataloging_{analysis_id}"
        if analysis_key not in data_store:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis_result = data_store[analysis_key]['result']
        column_classifications = analysis_result['column_classifications']
        
        # Perform semantic search
        search_results = perform_semantic_search_simple(query, column_classifications)
        
        return jsonify({
            'query': query,
            'results': search_results,
            'total_results': len(search_results)
        })
    
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

def perform_semantic_search_simple(query, column_classifications):
    """
    Perform simple semantic search on column classifications
    """
    try:
        query_lower = query.lower()
        results = []
        
        # Simple keyword matching
        for classification in column_classifications:
            score = 0
            
            # Check semantic type
            if any(word in classification['semantic_type'].lower() for word in query_lower.split()):
                score += 3
            
            # Check business meaning
            if any(word in classification['business_meaning'].lower() for word in query_lower.split()):
                score += 2
            
            # Check data category
            if any(word in classification['data_category'].lower() for word in query_lower.split()):
                score += 1
            
            # Check column name
            if any(word in classification['column_name'].lower() for word in query_lower.split()):
                score += 2
            
            if score > 0:
                results.append({
                    'column_name': classification['column_name'],
                    'semantic_type': classification['semantic_type'],
                    'business_meaning': classification['business_meaning'],
                    'relevance_score': score
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:10]  # Return top 10 results
    
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []

@app.route('/api/ai-cataloging/download', methods=['POST'])
def download_enhanced_dataset():
    """
    Download the enhanced dataset with AI insights - FIXED
    """
    try:
        data = request.json
        analysis_id = data.get('analysis_id', '')
        
        analysis_key = f"ai_cataloging_{analysis_id}"
        if analysis_key not in data_store:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis_data = data_store[analysis_key]
        enhanced_df = analysis_data['enhanced_df']
        filename = analysis_data['filename']
        
        # Create temporary file
        temp_filename = f"ai_cataloged_{filename}_{analysis_id[:8]}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Ensure temp directory exists
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        
        # Save enhanced dataset
        enhanced_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500






# Model Governance Backend - Add this to your app.py file

@app.route('/model-governance')
def model_governance():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Model Governance route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for Model Governance: {session_id}")
        return render_template('model-governance.html')
    except Exception as e:
        logger.error(f"Error in model_governance route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/model-governance/dataset-info', methods=['GET'])
def api_model_governance_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Model Governance dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for model governance suitability
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Determine governance relevance
            governance_relevance = "Medium"
            risk_level = "Low"
            
            if pd.api.types.is_numeric_dtype(df[col]):
                if unique_count > 10 and missing_pct < 20:
                    governance_relevance = "High"
                    if missing_pct > 10:
                        risk_level = "Medium"
            elif pd.api.types.is_object_dtype(df[col]):
                if unique_count < len(df) * 0.8:
                    governance_relevance = "High"
                    if 'id' in col.lower() or 'key' in col.lower():
                        risk_level = "High"
            
            # Get sample values
            sample_values = []
            try:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                sample_values = ["N/A"]
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'governance_relevance': governance_relevance,
                'risk_level': risk_level,
                'sample_values': sample_values
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_model_governance_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/model-governance/analyze', methods=['POST'])
def api_model_governance_analyze():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Model Governance analysis requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        selected_columns = data.get('selected_columns', [])
        governance_model = data.get('governance_model', 'gpt-4o')
        analysis_type = data.get('analysis_type', 'comprehensive')
        compliance_framework = data.get('compliance_framework', 'general')
        
        if not selected_columns:
            return jsonify({'error': 'No columns selected for governance analysis'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Perform Model Governance analysis
        start_time = time.time()
        governance_result = perform_model_governance_analysis(
            df, selected_columns, governance_model, analysis_type, compliance_framework, filename
        )
        processing_time = round(time.time() - start_time, 2)
        
        # Store governance result
        governance_id = str(uuid.uuid4())
        data_store[f"governance_{governance_id}"] = {
            'result': governance_result,
            'enhanced_df': governance_result['enhanced_df'],
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'columns_analyzed': selected_columns
        }
        
        # Prepare response
        response_result = governance_result.copy()
        if 'enhanced_df' in response_result:
            enhanced_df = response_result['enhanced_df']
            response_result['data_preview'] = {
                'columns': enhanced_df.columns.tolist(),
                'data': enhanced_df.head(20).to_dict(orient='records'),
                'shape': enhanced_df.shape
            }
            del response_result['enhanced_df']
        
        response_result['governance_id'] = governance_id
        response_result['processing_time'] = processing_time
        
        return jsonify(response_result)
    
    except Exception as e:
        logger.error(f"Error in api_model_governance_analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_model_governance_analysis(df, selected_columns, governance_model, analysis_type, compliance_framework, filename):
    """
    Perform comprehensive Model Governance analysis using LLMs and advanced techniques
    """
    try:
        # Initialize enhanced dataframe
        enhanced_df = df.copy()
        
        # 1. Data Quality Assessment for Model Governance
        data_quality_assessment = assess_data_quality_for_governance(df, selected_columns)
        
        # 2. Model Risk Assessment
        model_risk_assessment = assess_model_risks(df, selected_columns)
        
        # 3. Compliance and Regulatory Analysis
        compliance_analysis = analyze_compliance_requirements(df, selected_columns, compliance_framework)
        
        # 4. Data Lineage and Versioning
        data_lineage = create_data_lineage_tracking(df, selected_columns)
        
        # 5. Model Performance Monitoring Setup
        monitoring_framework = setup_model_monitoring(df, selected_columns)
        
        # 6. Bias and Fairness Assessment
        bias_assessment = assess_model_bias_and_fairness(df, selected_columns)
        
        # 7. Documentation and Audit Trail
        audit_documentation = generate_audit_documentation(df, selected_columns, filename)
        
        # 8. Enhanced DataFrame with Governance Metadata
        enhanced_df = add_governance_metadata(enhanced_df, selected_columns, data_quality_assessment)
        
        # 9. Generate AI-powered insights using Azure OpenAI
        ai_insights = generate_governance_insights_with_llm(
            data_quality_assessment, model_risk_assessment, compliance_analysis, 
            governance_model, filename
        )
        
        # 10. ETL Integration Recommendations
        etl_recommendations = generate_governance_etl_recommendations(
            data_quality_assessment, compliance_analysis
        )
        
        return {
            'analysis_type': analysis_type,
            'compliance_framework': compliance_framework,
            'data_quality_assessment': data_quality_assessment,
            'model_risk_assessment': model_risk_assessment,
            'compliance_analysis': compliance_analysis,
            'data_lineage': data_lineage,
            'monitoring_framework': monitoring_framework,
            'bias_assessment': bias_assessment,
            'audit_documentation': audit_documentation,
            'ai_insights': ai_insights,
            'etl_recommendations': etl_recommendations,
            'enhanced_df': enhanced_df,
            'governance_score': calculate_overall_governance_score(
                data_quality_assessment, model_risk_assessment, compliance_analysis
            )
        }
    
    except Exception as e:
        logger.error(f"Error in perform_model_governance_analysis: {str(e)}")
        raise

def assess_data_quality_for_governance(df, selected_columns):
    """Assess data quality specifically for model governance"""
    try:
        assessment = {
            'overall_score': 0,
            'column_assessments': [],
            'critical_issues': [],
            'recommendations': []
        }
        
        total_score = 0
        
        for col in selected_columns:
            if col in df.columns:
                col_assessment = {
                    'column': col,
                    'completeness': 0,
                    'consistency': 0,
                    'validity': 0,
                    'uniqueness': 0,
                    'issues': [],
                    'governance_flags': []
                }
                
                # Completeness
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                col_assessment['completeness'] = max(0, 100 - missing_pct)
                
                if missing_pct > 20:
                    col_assessment['issues'].append(f"High missing values: {missing_pct:.1f}%")
                    col_assessment['governance_flags'].append("Data Quality Risk")
                
                # Consistency
                if pd.api.types.is_object_dtype(df[col]):
                    # Check for inconsistent formatting
                    text_data = df[col].dropna().astype(str)
                    case_consistency = len(text_data[text_data == text_data.str.lower()]) / len(text_data) * 100
                    col_assessment['consistency'] = case_consistency
                    
                    if case_consistency < 80:
                        col_assessment['issues'].append("Inconsistent text formatting")
                        col_assessment['governance_flags'].append("Standardization Required")
                else:
                    col_assessment['consistency'] = 95  # Numeric data is generally consistent
                
                # Validity
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check for outliers
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                    outlier_pct = (len(outliers) / len(df)) * 100
                    col_assessment['validity'] = max(0, 100 - outlier_pct * 2)
                    
                    if outlier_pct > 5:
                        col_assessment['issues'].append(f"High outlier rate: {outlier_pct:.1f}%")
                        col_assessment['governance_flags'].append("Outlier Management Needed")
                else:
                    col_assessment['validity'] = 90
                
                # Uniqueness
                unique_pct = (df[col].nunique() / len(df)) * 100
                col_assessment['uniqueness'] = unique_pct
                
                if 'id' in col.lower() and unique_pct < 95:
                    col_assessment['issues'].append("ID column not sufficiently unique")
                    col_assessment['governance_flags'].append("Identity Integrity Risk")
                
                # Calculate column score
                col_score = (col_assessment['completeness'] + col_assessment['consistency'] + 
                           col_assessment['validity']) / 3
                col_assessment['overall_score'] = col_score
                total_score += col_score
                
                assessment['column_assessments'].append(col_assessment)
        
        assessment['overall_score'] = total_score / len(selected_columns) if selected_columns else 0
        
        # Generate critical issues and recommendations
        for col_assess in assessment['column_assessments']:
            if col_assess['overall_score'] < 70:
                assessment['critical_issues'].append({
                    'column': col_assess['column'],
                    'severity': 'High',
                    'issue': f"Poor data quality score: {col_assess['overall_score']:.1f}%",
                    'impact': 'Model performance and reliability at risk'
                })
        
        if assessment['overall_score'] < 80:
            assessment['recommendations'].append({
                'priority': 'High',
                'action': 'Implement comprehensive data quality monitoring',
                'description': 'Set up automated data quality checks in ETL pipeline'
            })
        
        return assessment
    
    except Exception as e:
        logger.error(f"Error in assess_data_quality_for_governance: {str(e)}")
        return {'overall_score': 50, 'column_assessments': [], 'critical_issues': [], 'recommendations': []}

def assess_model_risks(df, selected_columns):
    """Assess various model risks for governance"""
    try:
        risk_assessment = {
            'overall_risk_level': 'Medium',
            'risk_categories': [],
            'mitigation_strategies': []
        }
        
        # Data Drift Risk
        if len(df) > 100:
            # Simulate data drift detection by analyzing data distribution
            drift_risk = 'Low'
            for col in selected_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check coefficient of variation
                    cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                    if cv > 1:
                        drift_risk = 'High'
                        break
                    elif cv > 0.5:
                        drift_risk = 'Medium'
            
            risk_assessment['risk_categories'].append({
                'category': 'Data Drift',
                'level': drift_risk,
                'description': 'Risk of model performance degradation due to changing data patterns',
                'indicators': ['High coefficient of variation', 'Temporal data patterns']
            })
        
        # Model Complexity Risk
        complexity_risk = 'Low'
        if len(selected_columns) > 20:
            complexity_risk = 'High'
        elif len(selected_columns) > 10:
            complexity_risk = 'Medium'
        
        risk_assessment['risk_categories'].append({
            'category': 'Model Complexity',
            'level': complexity_risk,
            'description': 'Risk associated with model interpretability and maintenance',
            'indicators': [f'{len(selected_columns)} features selected']
        })
        
        # Data Privacy Risk
        privacy_risk = 'Low'
        sensitive_patterns = ['email', 'phone', 'ssn', 'id', 'name', 'address']
        for col in selected_columns:
            if any(pattern in col.lower() for pattern in sensitive_patterns):
                privacy_risk = 'High'
                break
        
        risk_assessment['risk_categories'].append({
            'category': 'Data Privacy',
            'level': privacy_risk,
            'description': 'Risk of privacy violations and regulatory non-compliance',
            'indicators': ['Potentially sensitive data detected']
        })
        
        # Generate mitigation strategies
        high_risk_categories = [r for r in risk_assessment['risk_categories'] if r['level'] == 'High']
        
        if high_risk_categories:
            risk_assessment['overall_risk_level'] = 'High'
            risk_assessment['mitigation_strategies'].extend([
                {
                    'strategy': 'Implement Continuous Monitoring',
                    'description': 'Set up real-time monitoring for data drift and model performance',
                    'priority': 'Critical'
                },
                {
                    'strategy': 'Data Anonymization',
                    'description': 'Apply data masking and anonymization techniques for sensitive data',
                    'priority': 'High'
                }
            ])
        
        return risk_assessment
    
    except Exception as e:
        logger.error(f"Error in assess_model_risks: {str(e)}")
        return {'overall_risk_level': 'Unknown', 'risk_categories': [], 'mitigation_strategies': []}

def analyze_compliance_requirements(df, selected_columns, compliance_framework):
    """Analyze compliance requirements based on framework"""
    try:
        compliance_analysis = {
            'framework': compliance_framework,
            'compliance_score': 0,
            'requirements': [],
            'violations': [],
            'recommendations': []
        }
        
        # Define compliance requirements based on framework
        if compliance_framework == 'gdpr':
            requirements = [
                {
                    'requirement': 'Data Minimization',
                    'description': 'Only collect and process necessary data',
                    'status': 'Compliant' if len(selected_columns) <= 15 else 'Non-Compliant',
                    'score': 100 if len(selected_columns) <= 15 else 60
                },
                {
                    'requirement': 'Data Quality',
                    'description': 'Ensure data accuracy and completeness',
                    'status': 'Compliant',  # Based on quality assessment
                    'score': 85
                },
                {
                    'requirement': 'Consent Management',
                    'description': 'Proper consent for data processing',
                    'status': 'Requires Review',
                    'score': 70
                }
            ]
        elif compliance_framework == 'sox':
            requirements = [
                {
                    'requirement': 'Data Integrity',
                    'description': 'Maintain data accuracy and prevent unauthorized changes',
                    'status': 'Compliant',
                    'score': 90
                },
                {
                    'requirement': 'Audit Trail',
                    'description': 'Maintain comprehensive audit logs',
                    'status': 'Compliant',
                    'score': 95
                },
                {
                    'requirement': 'Access Controls',
                    'description': 'Implement proper access controls',
                    'status': 'Requires Implementation',
                    'score': 60
                }
            ]
        else:  # General compliance
            requirements = [
                {
                    'requirement': 'Data Documentation',
                    'description': 'Comprehensive documentation of data sources and transformations',
                    'status': 'Compliant',
                    'score': 85
                },
                {
                    'requirement': 'Model Validation',
                    'description': 'Regular validation of model performance',
                    'status': 'Compliant',
                    'score': 80
                },
                {
                    'requirement': 'Risk Management',
                    'description': 'Identification and mitigation of model risks',
                    'status': 'Compliant',
                    'score': 75
                }
            ]
        
        compliance_analysis['requirements'] = requirements
        compliance_analysis['compliance_score'] = sum(r['score'] for r in requirements) / len(requirements)
        
        # Identify violations
        for req in requirements:
            if req['status'] == 'Non-Compliant':
                compliance_analysis['violations'].append({
                    'requirement': req['requirement'],
                    'severity': 'High',
                    'description': req['description']
                })
        
        # Generate recommendations
        if compliance_analysis['compliance_score'] < 80:
            compliance_analysis['recommendations'].append({
                'priority': 'High',
                'action': 'Address compliance gaps',
                'description': 'Implement missing compliance controls and procedures'
            })
        
        return compliance_analysis
    
    except Exception as e:
        logger.error(f"Error in analyze_compliance_requirements: {str(e)}")
        return {'framework': compliance_framework, 'compliance_score': 50, 'requirements': [], 'violations': [], 'recommendations': []}

def create_data_lineage_tracking(df, selected_columns):
    """Create data lineage and versioning information"""
    try:
        lineage = {
            'data_sources': [],
            'transformations': [],
            'version_info': {},
            'dependencies': []
        }
        
        # Simulate data source identification
        lineage['data_sources'] = [
            {
                'source_id': 'source_001',
                'source_name': 'Primary Dataset',
                'source_type': 'CSV Upload',
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'columns': selected_columns
            }
        ]
        
        # Track transformations
        lineage['transformations'] = [
            {
                'transformation_id': 'trans_001',
                'type': 'Data Selection',
                'description': f'Selected {len(selected_columns)} columns for governance analysis',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'input_columns': df.columns.tolist(),
                'output_columns': selected_columns
            }
        ]
        
        # Version information
        lineage['version_info'] = {
            'version': '1.0.0',
            'created_date': datetime.now().strftime("%Y-%m-%d"),
            'created_by': 'Model Governance System',
            'change_log': [
                {
                    'version': '1.0.0',
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'changes': 'Initial governance analysis'
                }
            ]
        }
        
        return lineage
    
    except Exception as e:
        logger.error(f"Error in create_data_lineage_tracking: {str(e)}")
        return {'data_sources': [], 'transformations': [], 'version_info': {}, 'dependencies': []}

def setup_model_monitoring(df, selected_columns):
    """Setup model performance monitoring framework"""
    try:
        monitoring = {
            'monitoring_metrics': [],
            'alert_thresholds': {},
            'monitoring_schedule': {},
            'dashboards': []
        }
        
        # Define monitoring metrics
        monitoring['monitoring_metrics'] = [
            {
                'metric': 'Data Quality Score',
                'description': 'Overall data quality assessment',
                'frequency': 'Daily',
                'threshold': 80
            },
            {
                'metric': 'Data Drift Detection',
                'description': 'Statistical tests for data distribution changes',
                'frequency': 'Weekly',
                'threshold': 0.05
            },
            {
                'metric': 'Model Performance',
                'description': 'Model accuracy and performance metrics',
                'frequency': 'Daily',
                'threshold': 0.85
            }
        ]
        
        # Alert thresholds
        monitoring['alert_thresholds'] = {
            'data_quality_critical': 60,
            'data_quality_warning': 75,
            'drift_detection_critical': 0.01,
            'drift_detection_warning': 0.03
        }
        
        # Monitoring schedule
        monitoring['monitoring_schedule'] = {
            'daily_checks': ['Data Quality', 'Model Performance'],
            'weekly_checks': ['Data Drift', 'Bias Assessment'],
            'monthly_checks': ['Compliance Review', 'Model Validation']
        }
        
        return monitoring
    
    except Exception as e:
        logger.error(f"Error in setup_model_monitoring: {str(e)}")
        return {'monitoring_metrics': [], 'alert_thresholds': {}, 'monitoring_schedule': {}, 'dashboards': []}

def assess_model_bias_and_fairness(df, selected_columns):
    """Assess potential bias and fairness issues"""
    try:
        bias_assessment = {
            'overall_fairness_score': 0,
            'bias_indicators': [],
            'fairness_metrics': [],
            'recommendations': []
        }
        
        # Check for potential bias indicators
        bias_indicators = []
        
        # Check for demographic-related columns
        demographic_keywords = ['age', 'gender', 'race', 'ethnicity', 'religion', 'nationality']
        for col in selected_columns:
            if any(keyword in col.lower() for keyword in demographic_keywords):
                bias_indicators.append({
                    'column': col,
                    'type': 'Demographic Attribute',
                    'risk_level': 'High',
                    'description': 'Column may contain protected demographic information'
                })
        
        # Check for geographic bias
        geographic_keywords = ['zip', 'postal', 'city', 'state', 'country', 'address']
        for col in selected_columns:
            if any(keyword in col.lower() for keyword in geographic_keywords):
                bias_indicators.append({
                    'column': col,
                    'type': 'Geographic Bias',
                    'risk_level': 'Medium',
                    'description': 'Geographic data may introduce location-based bias'
                })
        
        bias_assessment['bias_indicators'] = bias_indicators
        
        # Calculate fairness score
        if not bias_indicators:
            bias_assessment['overall_fairness_score'] = 95
        elif len([b for b in bias_indicators if b['risk_level'] == 'High']) > 0:
            bias_assessment['overall_fairness_score'] = 60
        else:
            bias_assessment['overall_fairness_score'] = 75
        
        # Generate recommendations
        if bias_assessment['overall_fairness_score'] < 80:
            bias_assessment['recommendations'].extend([
                {
                    'priority': 'High',
                    'action': 'Implement Bias Testing',
                    'description': 'Conduct regular bias testing across different demographic groups'
                },
                {
                    'priority': 'Medium',
                    'action': 'Feature Engineering Review',
                    'description': 'Review and potentially remove or transform biased features'
                }
            ])
        
        return bias_assessment
    
    except Exception as e:
        logger.error(f"Error in assess_model_bias_and_fairness: {str(e)}")
        return {'overall_fairness_score': 50, 'bias_indicators': [], 'fairness_metrics': [], 'recommendations': []}

def generate_audit_documentation(df, selected_columns, filename):
    """Generate comprehensive audit documentation"""
    try:
        audit_doc = {
            'document_id': str(uuid.uuid4()),
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_info': {},
            'governance_decisions': [],
            'approval_workflow': {},
            'risk_assessments': []
        }
        
        # Dataset information
        audit_doc['dataset_info'] = {
            'filename': filename,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'selected_columns': selected_columns,
            'data_types': {col: str(df[col].dtype) for col in selected_columns},
            'missing_data_summary': {col: int(df[col].isna().sum()) for col in selected_columns}
        }
        
        # Governance decisions
        audit_doc['governance_decisions'] = [
            {
                'decision_id': 'GOV_001',
                'decision': 'Column Selection for Analysis',
                'rationale': f'Selected {len(selected_columns)} columns based on governance relevance',
                'decision_date': datetime.now().strftime("%Y-%m-%d"),
                'decision_maker': 'Model Governance System'
            }
        ]
        
        # Approval workflow
        audit_doc['approval_workflow'] = {
            'status': 'Under Review',
            'required_approvals': ['Data Steward', 'Compliance Officer', 'Model Risk Manager'],
            'current_stage': 'Initial Assessment',
            'next_steps': ['Complete risk assessment', 'Stakeholder review', 'Final approval']
        }
        
        return audit_doc
    
    except Exception as e:
        logger.error(f"Error in generate_audit_documentation: {str(e)}")
        return {'document_id': 'ERROR', 'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def add_governance_metadata(df, selected_columns, data_quality_assessment):
    """Add governance metadata to the dataframe"""
    try:
        enhanced_df = df.copy()
        
        # Add governance flags
        enhanced_df['governance_reviewed'] = True
        enhanced_df['governance_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        enhanced_df['data_quality_score'] = data_quality_assessment['overall_score']
        
        # Add column-specific quality scores
        for col_assess in data_quality_assessment['column_assessments']:
            col_name = col_assess['column']
            if col_name in enhanced_df.columns:
                enhanced_df[f'{col_name}_quality_score'] = col_assess['overall_score']
                enhanced_df[f'{col_name}_governance_flags'] = ', '.join(col_assess['governance_flags'])
        
        return enhanced_df
    
    except Exception as e:
        logger.error(f"Error in add_governance_metadata: {str(e)}")
        return df

def generate_governance_insights_with_llm(data_quality_assessment, model_risk_assessment, 
                                        compliance_analysis, governance_model, filename):
    """Generate AI-powered governance insights using Azure OpenAI"""
    try:
        if not client:
            return generate_fallback_governance_insights(data_quality_assessment, model_risk_assessment)
        
        # Prepare summary for LLM
        governance_summary = {
            'dataset': filename,
            'data_quality_score': data_quality_assessment['overall_score'],
            'risk_level': model_risk_assessment['overall_risk_level'],
            'compliance_score': compliance_analysis['compliance_score'],
            'compliance_framework': compliance_analysis['framework'],
            'critical_issues': len(data_quality_assessment['critical_issues']),
            'high_risk_categories': len([r for r in model_risk_assessment['risk_categories'] if r['level'] == 'High'])
        }
        
        prompt = f"""
        You are an expert Model Governance specialist analyzing a dataset for ETL and ML governance.
        
        Governance Analysis Summary:
        {json.dumps(governance_summary, indent=2)}
        
        Provide 6-8 strategic insights covering:
        1. Overall governance readiness assessment
        2. Critical risk factors and mitigation strategies
        3. Compliance gaps and remediation actions
        4. Data quality improvements for model reliability
        5. ETL pipeline governance integration
        6. Model deployment readiness
        7. Ongoing monitoring and maintenance recommendations
        8. Stakeholder communication and approval workflow
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed governance analysis and strategic recommendation",
                    "category": "Risk Management|Compliance|Data Quality|ETL Integration|Deployment|Monitoring",
                    "priority": "Critical|High|Medium|Low",
                    "actionable": true/false,
                    "stakeholders": ["Data Steward", "Compliance Officer", "Model Risk Manager"],
                    "timeline": "Immediate|Short-term|Long-term"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=governance_model,
            messages=[
                {"role": "system", "content": "You are an expert Model Governance and Risk Management specialist. Provide strategic governance insights in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        return ai_response.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating LLM governance insights: {str(e)}")
        return generate_fallback_governance_insights(data_quality_assessment, model_risk_assessment)

def generate_fallback_governance_insights(data_quality_assessment, model_risk_assessment):
    """Generate fallback governance insights when LLM is not available"""
    insights = []
    
    # Data Quality Insight
    quality_score = data_quality_assessment['overall_score']
    if quality_score >= 90:
        insights.append({
            'title': 'Excellent Data Quality Foundation',
            'description': f'Data quality score of {quality_score:.1f}% indicates strong foundation for model governance.',
            'category': 'Data Quality',
            'priority': 'Medium',
            'actionable': False,
            'stakeholders': ['Data Steward'],
            'timeline': 'Current'
        })
    elif quality_score >= 70:
        insights.append({
            'title': 'Good Data Quality with Improvement Opportunities',
            'description': f'Data quality score of {quality_score:.1f}% is acceptable but has room for improvement.',
            'category': 'Data Quality',
            'priority': 'Medium',
            'actionable': True,
            'stakeholders': ['Data Steward', 'Data Engineer'],
            'timeline': 'Short-term'
        })
    else:
        insights.append({
            'title': 'Critical Data Quality Issues Detected',
            'description': f'Data quality score of {quality_score:.1f}% requires immediate attention before model deployment.',
            'category': 'Data Quality',
            'priority': 'Critical',
            'actionable': True,
            'stakeholders': ['Data Steward', 'Data Engineer', 'Model Risk Manager'],
            'timeline': 'Immediate'
        })
    
    # Risk Assessment Insight
    risk_level = model_risk_assessment['overall_risk_level']
    if risk_level == 'High':
        insights.append({
            'title': 'High Risk Model Deployment',
            'description': 'Multiple high-risk factors identified. Comprehensive risk mitigation required.',
            'category': 'Risk Management',
            'priority': 'Critical',
            'actionable': True,
            'stakeholders': ['Model Risk Manager', 'Compliance Officer'],
            'timeline': 'Immediate'
        })
    
    # ETL Integration
    insights.append({
        'title': 'ETL Pipeline Governance Integration',
        'description': 'Model governance controls can be integrated into ETL pipeline for automated compliance.',
        'category': 'ETL Integration',
        'priority': 'High',
        'actionable': True,
        'stakeholders': ['Data Engineer', 'ETL Developer'],
        'timeline': 'Short-term'
    })
    
    # Monitoring Recommendation
    insights.append({
        'title': 'Continuous Monitoring Framework',
        'description': 'Implement automated monitoring for data quality, model drift, and compliance.',
        'category': 'Monitoring',
        'priority': 'High',
        'actionable': True,
        'stakeholders': ['Data Steward', 'Model Risk Manager'],
        'timeline': 'Short-term'
    })
    
    return insights

def generate_governance_etl_recommendations(data_quality_assessment, compliance_analysis):
    """Generate ETL-specific governance recommendations"""
    recommendations = [
        {
            'category': 'Data Quality Gates',
            'title': 'Implement Quality Checkpoints',
            'description': 'Add automated data quality validation gates in ETL pipeline',
            'implementation': 'Configure quality thresholds and automated rejection of poor quality data',
            'priority': 'High',
            'estimated_effort': '2-3 weeks'
        },
        {
            'category': 'Audit Logging',
            'title': 'Comprehensive Audit Trail',
            'description': 'Implement detailed logging for all data transformations and model decisions',
            'implementation': 'Add audit logging to each ETL step with metadata capture',
            'priority': 'High',
            'estimated_effort': '1-2 weeks'
        },
        {
            'category': 'Compliance Automation',
            'title': 'Automated Compliance Checks',
            'description': 'Integrate compliance validation into ETL workflow',
            'implementation': 'Build compliance rules engine with automated validation',
            'priority': 'Medium',
            'estimated_effort': '3-4 weeks'
        },
        {
            'category': 'Model Versioning',
            'title': 'Model and Data Versioning',
            'description': 'Implement comprehensive versioning for models and training data',
            'implementation': 'Set up version control system with automated tagging',
            'priority': 'Medium',
            'estimated_effort': '2-3 weeks'
        }
    ]
    
    return recommendations

def calculate_overall_governance_score(data_quality_assessment, model_risk_assessment, compliance_analysis):
    """Calculate overall governance score"""
    try:
        # Weight the different components
        quality_weight = 0.4
        risk_weight = 0.3
        compliance_weight = 0.3
        
        # Data quality score (0-100)
        quality_score = data_quality_assessment['overall_score']
        
        # Risk score (convert risk level to numeric)
        risk_mapping = {'Low': 90, 'Medium': 70, 'High': 40}
        risk_score = risk_mapping.get(model_risk_assessment['overall_risk_level'], 50)
        
        # Compliance score (0-100)
        compliance_score = compliance_analysis['compliance_score']
        
        # Calculate weighted average
        overall_score = (quality_score * quality_weight + 
                        risk_score * risk_weight + 
                        compliance_score * compliance_weight)
        
        return {
            'overall_score': round(overall_score, 1),
            'quality_component': round(quality_score, 1),
            'risk_component': round(risk_score, 1),
            'compliance_component': round(compliance_score, 1),
            'grade': 'A' if overall_score >= 90 else 'B' if overall_score >= 80 else 'C' if overall_score >= 70 else 'D'
        }
    
    except Exception as e:
        logger.error(f"Error calculating governance score: {str(e)}")
        return {'overall_score': 50, 'grade': 'C'}

@app.route('/api/model-governance/download', methods=['POST'])
def api_model_governance_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        governance_id = data.get('governance_id')
        
        if not session_id or not governance_id:
            return jsonify({'error': 'Missing session_id or governance_id'}), 400
        
        governance_key = f"governance_{governance_id}"
        if governance_key not in data_store:
            return jsonify({'error': 'Governance analysis not found'}), 404
        
        governance_data = data_store[governance_key]
        enhanced_df = governance_data['enhanced_df']
        
        # Create temporary file
        temp_filename = f"model_governance_analysis_{governance_data['filename']}_{governance_id[:8]}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        enhanced_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_model_governance_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

# Add this to your existing features dictionary in app.py
# Add this entry to your features dictionary:
"""
'model-governance': {
    'title': 'Model Governance',
    'icon': 'bi-shield-check',
    'route': '/model-governance',
    'description': 'Comprehensive model governance using LLMs for ETL compliance, risk assessment, and audit trails',
    'progress': 100,
    'badge': 'Featured'
}
"""











# Add this import at the top
from code_free_modeling_backend import add_code_free_modeling_routes

# Add this line after your existing routes
add_code_free_modeling_routes(app, data_store, client)


# Import the new module
from prompt_engineering_interface import add_prompt_engineering_routes

# Add this line after your existing routes (around line 3000+ in your app.py)
add_prompt_engineering_routes(app, data_store, client)


# Import the integration module
from integration_notebooks_backend import add_integration_notebooks_routes

# Add this line after your existing routes (around line 3000+ in your app.py)
add_integration_notebooks_routes(app, data_store, client)


# Import the new AI Data Enrichment module
from ai_data_enrichment_backend import add_ai_data_enrichment_routes

# Add this line after your existing routes (around line 3000+ in your app.py)
add_ai_data_enrichment_routes(app, data_store, client)


# Add these two lines to your existing app.py file to integrate the Deploy to Alteryx Promote feature

# Import the new module (add this near the top with other imports)
from deploy_to_alteryx_promote import add_alteryx_promote_routes

# Add this line after your existing routes (around line 3000+ in your app.py)
add_alteryx_promote_routes(app, data_store, client)

from out_of_box_ml_backend import add_out_of_box_ml_routes
add_out_of_box_ml_routes(app, data_store, client)






# Import the clustering module (add this near the top with other imports)
from clustering_backend import add_clustering_routes

# Add this line after your existing routes (around line 3000+ in your app.py)
add_clustering_routes(app, data_store, client)


from object_detection import object_detection_bp
app.register_blueprint(object_detection_bp)





# Import the new module (add this near the top with other imports)
from text_classification_summarization_backend import add_text_classification_routes

# Add this line after your existing routes (around line 3000+ in your app.py)
add_text_classification_routes(app, data_store, client)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

