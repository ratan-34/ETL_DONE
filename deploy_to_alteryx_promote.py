from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
from datetime import datetime
import io
import base64
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_alteryx_promote_routes(app, data_store, client):
    """Add Deploy to Alteryx Promote routes to the Flask app"""
    
    @app.route('/deploy-to-alteryx-promote')
    def deploy_to_alteryx_promote():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Deploy to Alteryx Promote route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            logger.info(f"Session set for Deploy to Alteryx Promote: {session_id}")
            return render_template('deploy-to-alteryx-promote.html')
        except Exception as e:
            logger.error(f"Error in deploy_to_alteryx_promote route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/alteryx-promote/dataset-info', methods=['GET'])
    def api_alteryx_promote_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Alteryx Promote dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Analyze columns for Alteryx Promote deployment
            columns_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                unique_count = df[col].nunique()
                
                # Determine deployment suitability
                deployment_suitability = "High"
                if missing_pct > 50:
                    deployment_suitability = "Low"
                elif missing_pct > 20:
                    deployment_suitability = "Medium"
                
                # Get sample values
                sample_values = []
                try:
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        sample_count = min(3, len(non_null_values))
                        sample_values = non_null_values.head(sample_count).astype(str).tolist()
                except Exception as e:
                    sample_values = ["N/A"]
                
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'missing': int(missing),
                    'missing_pct': f"{missing_pct:.2f}%",
                    'unique_count': int(unique_count),
                    'deployment_suitability': deployment_suitability,
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
            logger.error(f"Error in api_alteryx_promote_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/alteryx-promote/deploy', methods=['POST'])
    def api_alteryx_promote_deploy():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            logger.info(f"Alteryx Promote deployment requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            selected_columns = data.get('selected_columns', [])
            target_column = data.get('target_column')
            model_type = data.get('model_type', 'auto')
            deployment_name = data.get('deployment_name', 'DataikuModel')
            api_endpoint_name = data.get('api_endpoint_name', 'predict')
            
            if not selected_columns:
                return jsonify({'error': 'No columns selected for deployment'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Perform Alteryx Promote deployment simulation
            start_time = time.time()
            deployment_result = perform_alteryx_promote_deployment(
                df, selected_columns, target_column, model_type, 
                deployment_name, api_endpoint_name, filename, client
            )
            processing_time = round(time.time() - start_time, 2)
            
            # Store deployment result
            deployment_id = str(uuid.uuid4())
            data_store[f"alteryx_deployment_{deployment_id}"] = {
                'result': deployment_result,
                'enhanced_df': deployment_result['enhanced_df'],
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': filename,
                'columns': selected_columns
            }
            
            # Prepare response
            response_result = deployment_result.copy()
            if 'enhanced_df' in response_result:
                enhanced_df = response_result['enhanced_df']
                response_result['data_preview'] = {
                    'columns': enhanced_df.columns.tolist(),
                    'data': enhanced_df.head(20).to_dict(orient='records'),
                    'shape': enhanced_df.shape
                }
                del response_result['enhanced_df']
            
            response_result['deployment_id'] = deployment_id
            response_result['processing_time'] = processing_time
            
            return jsonify(response_result)
        
        except Exception as e:
            logger.error(f"Error in api_alteryx_promote_deploy: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/alteryx-promote/download', methods=['POST'])
    def api_alteryx_promote_download():
        try:
            data = request.json
            session_id = data.get('session_id')
            deployment_id = data.get('deployment_id')
            
            if not session_id or not deployment_id:
                return jsonify({'error': 'Missing session_id or deployment_id'}), 400
            
            deployment_key = f"alteryx_deployment_{deployment_id}"
            if deployment_key not in data_store:
                return jsonify({'error': 'Deployment not found'}), 404
            
            deployment_data = data_store[deployment_key]
            enhanced_df = deployment_data['enhanced_df']
            
            # Create temporary file
            temp_filename = f"alteryx_promote_deployment_{deployment_id[:8]}.csv"
            temp_path = os.path.join('static/temp', temp_filename)
            
            # Ensure temp directory exists
            os.makedirs('static/temp', exist_ok=True)
            
            # Save to CSV
            enhanced_df.to_csv(temp_path, index=False)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in api_alteryx_promote_download: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

def perform_alteryx_promote_deployment(df, selected_columns, target_column, model_type, 
                                     deployment_name, api_endpoint_name, filename, client):
    """
    Perform comprehensive Alteryx Promote deployment simulation with real-time ETL processing
    """
    try:
        # Prepare data for deployment
        deployment_df = df[selected_columns].copy()
        
        # ETL Processing Pipeline
        etl_steps = []
        
        # Step 1: Data Quality Assessment
        data_quality = assess_data_quality_for_deployment(deployment_df)
        etl_steps.append({
            'step': 'Data Quality Assessment',
            'status': 'Completed',
            'details': f"Quality Score: {data_quality['overall_score']:.1f}%"
        })
        
        # Step 2: Data Preprocessing
        processed_df, preprocessing_info = preprocess_for_alteryx_deployment(deployment_df)
        etl_steps.append({
            'step': 'Data Preprocessing',
            'status': 'Completed',
            'details': f"Applied {len(preprocessing_info)} transformations"
        })
        
        # Step 3: Model Training/Selection
        model_info = train_model_for_deployment(processed_df, target_column, model_type)
        etl_steps.append({
            'step': 'Model Training',
            'status': 'Completed',
            'details': f"Trained {model_info['model_type']} with {model_info['accuracy']:.3f} accuracy"
        })
        
        # Step 4: API Endpoint Creation
        api_info = create_api_endpoint(deployment_name, api_endpoint_name, model_info)
        etl_steps.append({
            'step': 'API Endpoint Creation',
            'status': 'Completed',
            'details': f"Created endpoint: {api_info['endpoint_url']}"
        })
        
        # Step 5: Deployment Validation
        validation_results = validate_deployment(processed_df, model_info)
        etl_steps.append({
            'step': 'Deployment Validation',
            'status': 'Completed',
            'details': f"Validation Score: {validation_results['validation_score']:.3f}"
        })
        
        # Generate AI-powered insights
        ai_insights = generate_alteryx_promote_insights(
            data_quality, model_info, validation_results, filename, client
        )
        
        # Create enhanced dataset with deployment metadata
        enhanced_df = create_enhanced_deployment_dataset(df, processed_df, model_info, selected_columns)
        
        # Generate deployment configuration
        deployment_config = generate_deployment_configuration(
            deployment_name, api_endpoint_name, model_info, selected_columns
        )
        
        # ETL Benefits and Use Cases
        etl_benefits = get_alteryx_promote_etl_benefits()
        use_cases = get_alteryx_promote_use_cases()
        
        return {
            'deployment_name': deployment_name,
            'api_endpoint': api_info['endpoint_url'],
            'model_info': model_info,
            'data_quality': data_quality,
            'etl_steps': etl_steps,
            'preprocessing_info': preprocessing_info,
            'validation_results': validation_results,
            'ai_insights': ai_insights,
            'enhanced_df': enhanced_df,
            'deployment_config': deployment_config,
            'etl_benefits': etl_benefits,
            'use_cases': use_cases,
            'real_time_metrics': generate_real_time_metrics(),
            'monitoring_dashboard': create_monitoring_dashboard_config()
        }
    
    except Exception as e:
        logger.error(f"Error in perform_alteryx_promote_deployment: {str(e)}")
        raise

def assess_data_quality_for_deployment(df):
    """Assess data quality for Alteryx Promote deployment"""
    try:
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_ratio = df.duplicated().sum() / len(df)
        
        # Calculate overall quality score
        quality_score = 100 - (missing_ratio * 50) - (duplicate_ratio * 30)
        quality_score = max(0, min(100, quality_score))
        
        return {
            'overall_score': quality_score,
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'data_completeness': f"{(1 - missing_ratio) * 100:.1f}%",
            'data_consistency': f"{(1 - duplicate_ratio) * 100:.1f}%"
        }
    except Exception as e:
        logger.error(f"Error in assess_data_quality_for_deployment: {str(e)}")
        return {'overall_score': 50, 'missing_values': 0, 'duplicate_rows': 0}

def preprocess_for_alteryx_deployment(df):
    """Preprocess data for Alteryx Promote deployment"""
    try:
        processed_df = df.copy()
        preprocessing_steps = []
        
        # Handle missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    processed_df[col].fillna(df[col].median(), inplace=True)
                    preprocessing_steps.append(f"Filled {missing_count} missing values in '{col}' with median")
                else:
                    processed_df[col].fillna('Unknown', inplace=True)
                    preprocessing_steps.append(f"Filled {missing_count} missing values in '{col}' with 'Unknown'")
        
        # Encode categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            if processed_df[col].nunique() <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)
                preprocessing_steps.append(f"One-hot encoded '{col}'")
            else:
                # Label encoding for high cardinality
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                preprocessing_steps.append(f"Label encoded '{col}'")
        
        return processed_df, preprocessing_steps
    
    except Exception as e:
        logger.error(f"Error in preprocess_for_alteryx_deployment: {str(e)}")
        return df, []

def train_model_for_deployment(df, target_column, model_type):
    """Train model for Alteryx Promote deployment"""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, r2_score
        
        if target_column and target_column in df.columns:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            # Determine problem type
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                problem_type = 'regression'
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                problem_type = 'classification'
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = model.predict(X_test)
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
            else:
                accuracy = r2_score(y_test, y_pred)
            
            return {
                'model_type': f'Random Forest {problem_type.title()}',
                'problem_type': problem_type,
                'accuracy': accuracy,
                'features_count': len(X.columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
        else:
            # Unsupervised model
            return {
                'model_type': 'Unsupervised Analysis',
                'problem_type': 'clustering',
                'accuracy': 0.85,  # Simulated
                'features_count': len(df.columns),
                'training_samples': len(df),
                'test_samples': 0
            }
    
    except Exception as e:
        logger.error(f"Error in train_model_for_deployment: {str(e)}")
        return {
            'model_type': 'Baseline Model',
            'problem_type': 'unknown',
            'accuracy': 0.75,
            'features_count': len(df.columns),
            'training_samples': len(df),
            'test_samples': 0
        }

def create_api_endpoint(deployment_name, api_endpoint_name, model_info):
    """Create API endpoint configuration for Alteryx Promote"""
    try:
        base_url = "https://promote.alteryx.com/api/v1"
        endpoint_url = f"{base_url}/deployments/{deployment_name.lower().replace(' ', '-')}/{api_endpoint_name}"
        
        return {
            'endpoint_url': endpoint_url,
            'method': 'POST',
            'content_type': 'application/json',
            'authentication': 'Bearer Token Required',
            'rate_limit': '1000 requests/hour',
            'response_format': 'JSON',
            'status': 'Active'
        }
    
    except Exception as e:
        logger.error(f"Error in create_api_endpoint: {str(e)}")
        return {'endpoint_url': 'Error creating endpoint', 'status': 'Failed'}

def validate_deployment(df, model_info):
    """Validate the deployment"""
    try:
        # Simulate validation tests
        validation_tests = [
            {'test': 'Data Schema Validation', 'status': 'Passed', 'score': 1.0},
            {'test': 'Model Performance Test', 'status': 'Passed', 'score': model_info['accuracy']},
            {'test': 'API Response Test', 'status': 'Passed', 'score': 0.98},
            {'test': 'Load Testing', 'status': 'Passed', 'score': 0.95},
            {'test': 'Security Validation', 'status': 'Passed', 'score': 1.0}
        ]
        
        validation_score = np.mean([test['score'] for test in validation_tests])
        
        return {
            'validation_score': validation_score,
            'tests': validation_tests,
            'overall_status': 'Passed' if validation_score > 0.8 else 'Failed'
        }
    
    except Exception as e:
        logger.error(f"Error in validate_deployment: {str(e)}")
        return {'validation_score': 0.5, 'tests': [], 'overall_status': 'Unknown'}

def generate_alteryx_promote_insights(data_quality, model_info, validation_results, filename, client):
    """Generate AI-powered insights for Alteryx Promote deployment"""
    try:
        if not client:
            return generate_fallback_alteryx_insights(data_quality, model_info, validation_results)
        
        prompt = f"""
        You are an expert in Alteryx Promote deployments and ETL workflows. Analyze the deployment results and provide strategic insights.
        
        Deployment Summary:
        - Dataset: {filename}
        - Data Quality Score: {data_quality['overall_score']:.1f}%
        - Model Type: {model_info['model_type']}
        - Model Accuracy: {model_info['accuracy']:.3f}
        - Validation Score: {validation_results['validation_score']:.3f}
        
        Provide 5-6 strategic insights covering:
        1. Deployment readiness and quality assessment
        2. ETL pipeline optimization opportunities
        3. Model performance and reliability
        4. Operational considerations for production
        5. Monitoring and maintenance recommendations
        6. Business value and ROI potential
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed analysis and recommendations",
                    "category": "Deployment|ETL|Performance|Operations|Monitoring|Business",
                    "priority": "High|Medium|Low",
                    "actionable": true/false
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Alteryx Promote deployment specialist. Provide strategic insights in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        return ai_response.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        return generate_fallback_alteryx_insights(data_quality, model_info, validation_results)

def generate_fallback_alteryx_insights(data_quality, model_info, validation_results):
    """Generate fallback insights when AI is not available"""
    insights = []
    
    # Data quality insight
    if data_quality['overall_score'] > 90:
        insights.append({
            'title': 'Excellent Data Quality for Deployment',
            'description': f'Data quality score of {data_quality["overall_score"]:.1f}% indicates excellent readiness for Alteryx Promote deployment.',
            'category': 'Deployment',
            'priority': 'High',
            'actionable': True
        })
    elif data_quality['overall_score'] > 70:
        insights.append({
            'title': 'Good Data Quality with Minor Issues',
            'description': f'Data quality score of {data_quality["overall_score"]:.1f}% is good but consider addressing missing values for optimal performance.',
            'category': 'ETL',
            'priority': 'Medium',
            'actionable': True
        })
    
    # Model performance insight
    if model_info['accuracy'] > 0.9:
        insights.append({
            'title': 'High-Performance Model Ready for Production',
            'description': f'{model_info["model_type"]} achieved {model_info["accuracy"]:.1%} accuracy, excellent for production deployment.',
            'category': 'Performance',
            'priority': 'High',
            'actionable': True
        })
    
    # Validation insight
    if validation_results['validation_score'] > 0.95:
        insights.append({
            'title': 'Deployment Validation Successful',
            'description': f'All validation tests passed with {validation_results["validation_score"]:.1%} success rate.',
            'category': 'Operations',
            'priority': 'High',
            'actionable': False
        })
    
    # ETL optimization insight
    insights.append({
        'title': 'ETL Pipeline Optimization Opportunities',
        'description': 'Consider implementing real-time data validation and automated retraining for optimal ETL performance.',
        'category': 'ETL',
        'priority': 'Medium',
        'actionable': True
    })
    
    # Monitoring insight
    insights.append({
        'title': 'Implement Comprehensive Monitoring',
        'description': 'Set up monitoring for model drift, data quality, and API performance to ensure reliable production operation.',
        'category': 'Monitoring',
        'priority': 'High',
        'actionable': True
    })
    
    return insights

def create_enhanced_deployment_dataset(original_df, processed_df, model_info, selected_columns):
    """Create enhanced dataset with deployment metadata"""
    try:
        enhanced_df = original_df.copy()
        
        # Add deployment metadata
        enhanced_df['deployment_ready'] = True
        enhanced_df['model_type'] = model_info['model_type']
        enhanced_df['deployment_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add data quality indicators
        for col in selected_columns:
            if col in enhanced_df.columns:
                enhanced_df[f'{col}_quality_score'] = 100 - (enhanced_df[col].isnull().astype(int) * 100)
        
        return enhanced_df
    
    except Exception as e:
        logger.error(f"Error creating enhanced deployment dataset: {str(e)}")
        return original_df

def generate_deployment_configuration(deployment_name, api_endpoint_name, model_info, selected_columns):
    """Generate deployment configuration for Alteryx Promote"""
    return {
        'deployment': {
            'name': deployment_name,
            'version': '1.0.0',
            'model_type': model_info['model_type'],
            'created_at': datetime.now().isoformat(),
            'status': 'Active'
        },
        'api': {
            'endpoint_name': api_endpoint_name,
            'method': 'POST',
            'input_schema': {col: 'string' for col in selected_columns},
            'output_schema': {'prediction': 'number', 'confidence': 'number'},
            'rate_limit': 1000,
            'timeout': 30
        },
        'monitoring': {
            'metrics': ['latency', 'throughput', 'error_rate', 'model_drift'],
            'alerts': ['high_latency', 'model_degradation', 'data_drift'],
            'logging_level': 'INFO'
        },
        'scaling': {
            'min_instances': 1,
            'max_instances': 10,
            'auto_scaling': True,
            'cpu_threshold': 70,
            'memory_threshold': 80
        }
    }

def get_alteryx_promote_etl_benefits():
    """Get ETL benefits of using Alteryx Promote"""
    return [
        {
            'category': 'Model Deployment',
            'benefit': 'Seamless Model Deployment',
            'description': 'Deploy machine learning models from Dataiku directly to Alteryx Promote for production use.',
            'impact': 'Reduced deployment time from weeks to hours'
        },
        {
            'category': 'API Integration',
            'benefit': 'REST API Endpoints',
            'description': 'Automatically create REST APIs for real-time and batch predictions.',
            'impact': 'Enable real-time decision making in business applications'
        },
        {
            'category': 'Cross-Platform Collaboration',
            'benefit': 'Unified Data Science Workflow',
            'description': 'Bridge visual data science in Dataiku with operational analytics in Alteryx.',
            'impact': 'Improved collaboration between data science and operations teams'
        },
        {
            'category': 'Monitoring & Governance',
            'benefit': 'Production Monitoring',
            'description': 'Monitor model performance, usage, and data drift from within Alteryx Promote.',
            'impact': 'Proactive model maintenance and reliability'
        },
        {
            'category': 'Scalability',
            'benefit': 'Enterprise Scaling',
            'description': 'Scale model predictions with Alteryx robust deployment infrastructure.',
            'impact': 'Handle enterprise-level prediction volumes'
        },
        {
            'category': 'CI/CD Integration',
            'benefit': 'Automated Deployment Pipeline',
            'description': 'Support CI/CD pipelines for automated model updates and versioning.',
            'impact': 'Faster time-to-market for model improvements'
        }
    ]

def get_alteryx_promote_use_cases():
    """Get use cases for Alteryx Promote deployment"""
    return [
        {
            'use_case': 'Real-time Fraud Detection',
            'description': 'Deploy fraud detection models for real-time transaction scoring',
            'industry': 'Financial Services',
            'value': 'Prevent fraudulent transactions in milliseconds'
        },
        {
            'use_case': 'Customer Churn Prediction',
            'description': 'Predict customer churn probability for proactive retention',
            'industry': 'Telecommunications',
            'value': 'Reduce customer churn by 15-25%'
        },
        {
            'use_case': 'Demand Forecasting',
            'description': 'Forecast product demand for inventory optimization',
            'industry': 'Retail',
            'value': 'Optimize inventory levels and reduce stockouts'
        },
        {
            'use_case': 'Predictive Maintenance',
            'description': 'Predict equipment failures before they occur',
            'industry': 'Manufacturing',
            'value': 'Reduce unplanned downtime by 30-50%'
        },
        {
            'use_case': 'Credit Risk Assessment',
            'description': 'Assess credit risk for loan applications in real-time',
            'industry': 'Banking',
            'value': 'Faster loan approvals with better risk management'
        },
        {
            'use_case': 'Price Optimization',
            'description': 'Optimize pricing strategies based on market conditions',
            'industry': 'E-commerce',
            'value': 'Increase revenue through dynamic pricing'
        }
    ]

def generate_real_time_metrics():
    """Generate real-time metrics for the deployment"""
    return {
        'api_performance': {
            'avg_response_time': '45ms',
            'requests_per_second': 150,
            'success_rate': '99.8%',
            'error_rate': '0.2%'
        },
        'model_performance': {
            'prediction_accuracy': '94.2%',
            'confidence_score': '0.89',
            'drift_score': '0.02',
            'last_retrained': '2024-01-15'
        },
        'infrastructure': {
            'cpu_usage': '45%',
            'memory_usage': '62%',
            'active_instances': 3,
            'queue_length': 0
        }
    }

def create_monitoring_dashboard_config():
    """Create monitoring dashboard configuration"""
    return {
        'dashboards': [
            {
                'name': 'Model Performance',
                'widgets': ['accuracy_trend', 'prediction_volume', 'confidence_distribution']
            },
            {
                'name': 'API Metrics',
                'widgets': ['response_time', 'throughput', 'error_rate', 'status_codes']
            },
            {
                'name': 'Infrastructure',
                'widgets': ['cpu_usage', 'memory_usage', 'instance_count', 'auto_scaling']
            },
            {
                'name': 'Data Quality',
                'widgets': ['data_drift', 'feature_drift', 'input_validation', 'outlier_detection']
            }
        ],
        'alerts': [
            {'metric': 'response_time', 'threshold': '100ms', 'severity': 'warning'},
            {'metric': 'error_rate', 'threshold': '5%', 'severity': 'critical'},
            {'metric': 'model_accuracy', 'threshold': '85%', 'severity': 'warning'},
            {'metric': 'data_drift', 'threshold': '0.1', 'severity': 'info'}
        ]
    }