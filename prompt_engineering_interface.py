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
import os
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptEngineeringInterface:
    """
    Advanced Prompt Engineering Interface for ETL using LLMs
    Similar to Dataiku's AI/ML workflows with real-time processing
    """
    
    def __init__(self, azure_client=None):
        self.client = azure_client
        self.prompt_templates = {
            'data_analysis': {
                'name': 'Data Analysis & Insights',
                'description': 'Analyze data patterns and generate insights',
                'template': '''
                Analyze the following dataset and provide comprehensive insights:
                
                Dataset: {dataset_name}
                Columns: {columns}
                Sample Data: {sample_data}
                
                Please provide:
                1. Data quality assessment
                2. Key patterns and trends
                3. Potential data issues
                4. ETL recommendations
                5. Business insights
                '''
            },
            'etl_optimization': {
                'name': 'ETL Pipeline Optimization',
                'description': 'Optimize ETL processes and workflows',
                'template': '''
                Optimize the ETL pipeline for this dataset:
                
                Dataset: {dataset_name}
                Current Structure: {columns}
                Data Types: {dtypes}
                
                Provide recommendations for:
                1. Data transformation strategies
                2. Performance optimization
                3. Data validation rules
                4. Error handling approaches
                5. Monitoring and alerting
                '''
            },
            'data_transformation': {
                'name': 'Data Transformation Suggestions',
                'description': 'Generate data transformation recommendations',
                'template': '''
                Suggest data transformations for this dataset:
                
                Dataset: {dataset_name}
                Columns: {columns}
                Data Quality Issues: {quality_issues}
                
                Recommend:
                1. Cleaning operations
                2. Feature engineering
                3. Data standardization
                4. Aggregation strategies
                5. Derived columns
                '''
            },
            'custom_prompt': {
                'name': 'Custom Analysis',
                'description': 'Custom prompt for specific analysis needs',
                'template': '{custom_prompt}'
            }
        }
    
    def analyze_dataset_for_prompting(self, df, selected_columns=None):
        """Analyze dataset to prepare context for LLM prompting"""
        try:
            if selected_columns:
                analysis_df = df[selected_columns].copy()
            else:
                analysis_df = df.copy()
            
            # Basic dataset information
            dataset_info = {
                'shape': analysis_df.shape,
                'columns': analysis_df.columns.tolist(),
                'dtypes': {col: str(analysis_df[col].dtype) for col in analysis_df.columns},
                'missing_values': analysis_df.isnull().sum().to_dict(),
                'sample_data': analysis_df.head(5).to_dict(orient='records')
            }
            
            # Data quality assessment
            quality_issues = []
            for col in analysis_df.columns:
                missing_pct = (analysis_df[col].isnull().sum() / len(analysis_df)) * 100
                if missing_pct > 20:
                    quality_issues.append(f"{col}: {missing_pct:.1f}% missing values")
                
                if analysis_df[col].dtype == 'object':
                    unique_ratio = analysis_df[col].nunique() / len(analysis_df)
                    if unique_ratio > 0.8:
                        quality_issues.append(f"{col}: High cardinality ({unique_ratio:.1%})")
            
            dataset_info['quality_issues'] = quality_issues
            
            return dataset_info
        
        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            raise
    
    def execute_prompt_engineering(self, dataset_info, prompt_type, custom_prompt=None, model='gpt-4o'):
        """Execute prompt engineering with LLM"""
        try:
            if not self.client:
                return self._fallback_analysis(dataset_info, prompt_type)
            
            # Get prompt template
            if prompt_type == 'custom_prompt' and custom_prompt:
                prompt_template = self.prompt_templates[prompt_type]['template']
                formatted_prompt = prompt_template.format(custom_prompt=custom_prompt)
            else:
                prompt_template = self.prompt_templates[prompt_type]['template']
                formatted_prompt = prompt_template.format(
                    dataset_name=f"Dataset with {dataset_info['shape'][0]} rows and {dataset_info['shape'][1]} columns",
                    columns=', '.join(dataset_info['columns']),
                    sample_data=json.dumps(dataset_info['sample_data'][:3], indent=2),
                    dtypes=json.dumps(dataset_info['dtypes'], indent=2),
                    quality_issues='; '.join(dataset_info['quality_issues']) if dataset_info['quality_issues'] else 'No major issues detected'
                )
            
            # Execute LLM prompt
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert ETL engineer and data scientist. Provide detailed, actionable insights and recommendations in a structured format."
                    },
                    {
                        "role": "user", 
                        "content": formatted_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            llm_response = response.choices[0].message.content
            
            # Process and structure the response
            structured_response = self._structure_llm_response(llm_response, prompt_type)
            
            return structured_response
        
        except Exception as e:
            logger.error(f"Error in prompt engineering: {str(e)}")
            return self._fallback_analysis(dataset_info, prompt_type)
    
    def _structure_llm_response(self, llm_response, prompt_type):
        """Structure the LLM response into actionable insights"""
        try:
            # Parse the response and create structured output
            insights = {
                'prompt_type': prompt_type,
                'raw_response': llm_response,
                'structured_insights': [],
                'recommendations': [],
                'etl_actions': [],
                'code_suggestions': []
            }
            
            # Split response into sections
            sections = llm_response.split('\n\n')
            
            for section in sections:
                if any(keyword in section.lower() for keyword in ['insight', 'pattern', 'trend']):
                    insights['structured_insights'].append(section.strip())
                elif any(keyword in section.lower() for keyword in ['recommend', 'suggest', 'should']):
                    insights['recommendations'].append(section.strip())
                elif any(keyword in section.lower() for keyword in ['etl', 'pipeline', 'transform']):
                    insights['etl_actions'].append(section.strip())
                elif any(keyword in section.lower() for keyword in ['code', 'script', 'function']):
                    insights['code_suggestions'].append(section.strip())
            
            return insights
        
        except Exception as e:
            logger.error(f"Error structuring response: {str(e)}")
            return {
                'prompt_type': prompt_type,
                'raw_response': llm_response,
                'structured_insights': [llm_response],
                'recommendations': [],
                'etl_actions': [],
                'code_suggestions': []
            }
    
    def _fallback_analysis(self, dataset_info, prompt_type):
        """Fallback analysis when LLM is not available"""
        fallback_insights = {
            'prompt_type': prompt_type,
            'raw_response': 'LLM analysis not available. Providing basic analysis.',
            'structured_insights': [
                f"Dataset contains {dataset_info['shape'][0]} rows and {dataset_info['shape'][1]} columns",
                f"Columns: {', '.join(dataset_info['columns'])}",
                f"Data quality issues: {len(dataset_info['quality_issues'])} identified"
            ],
            'recommendations': [
                "Review data quality issues identified",
                "Consider data validation rules",
                "Implement proper error handling"
            ],
            'etl_actions': [
                "Set up data profiling pipeline",
                "Implement data quality monitoring",
                "Create transformation workflows"
            ],
            'code_suggestions': [
                "Use pandas for data manipulation",
                "Implement data validation functions",
                "Create automated testing for data pipelines"
            ]
        }
        
        return fallback_insights
    
    def generate_etl_recommendations(self, dataset_info, analysis_results):
        """Generate specific ETL recommendations based on analysis"""
        try:
            recommendations = {
                'data_quality': [],
                'performance': [],
                'transformation': [],
                'monitoring': []
            }
            
            # Data quality recommendations
            if dataset_info['quality_issues']:
                recommendations['data_quality'].extend([
                    "Implement data validation checks",
                    "Set up automated data quality monitoring",
                    "Create data cleansing workflows"
                ])
            
            # Performance recommendations
            if dataset_info['shape'][0] > 100000:
                recommendations['performance'].extend([
                    "Consider data partitioning strategies",
                    "Implement incremental processing",
                    "Optimize data storage formats"
                ])
            
            # Transformation recommendations
            recommendations['transformation'].extend([
                "Standardize data formats",
                "Implement feature engineering pipelines",
                "Create reusable transformation functions"
            ])
            
            # Monitoring recommendations
            recommendations['monitoring'].extend([
                "Set up data pipeline monitoring",
                "Implement alerting for data anomalies",
                "Create data lineage tracking"
            ])
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {}
    
    def create_enhanced_dataset(self, original_df, analysis_results, selected_columns=None):
        """Create enhanced dataset with analysis results"""
        try:
            enhanced_df = original_df.copy()
            
            # Add analysis metadata
            enhanced_df['prompt_analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            enhanced_df['prompt_type'] = analysis_results['prompt_type']
            
            # Add data quality flags
            for col in enhanced_df.columns:
                if col in original_df.columns:
                    missing_pct = (enhanced_df[col].isnull().sum() / len(enhanced_df)) * 100
                    enhanced_df[f'{col}_quality_score'] = 100 - missing_pct
            
            return enhanced_df
        
        except Exception as e:
            logger.error(f"Error creating enhanced dataset: {str(e)}")
            return original_df

def add_prompt_engineering_routes(app, data_store, client):
    """Add Prompt Engineering Interface routes to Flask app"""
    
    prompt_engine = PromptEngineeringInterface(client)
    
    @app.route('/prompt-engineering-interface')
    def prompt_engineering_interface():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            if not session_id or session_id not in data_store:
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            return render_template('prompt-engineering-interface.html')
        except Exception as e:
            logger.error(f"Error in prompt_engineering_interface route: {str(e)}")
            return redirect(url_for('index'))
    
    @app.route('/api/prompt-engineering/dataset-info', methods=['GET'])
    def get_prompt_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Analyze dataset for prompting
            dataset_info = prompt_engine.analyze_dataset_for_prompting(df)
            dataset_info['filename'] = filename
            dataset_info['session_id'] = session_id
            
            return jsonify(dataset_info)
        
        except Exception as e:
            logger.error(f"Error in get_prompt_dataset_info: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/api/prompt-engineering/analyze', methods=['POST'])
    def execute_prompt_analysis():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Extract parameters
            selected_columns = data.get('selected_columns', [])
            prompt_type = data.get('prompt_type', 'data_analysis')
            custom_prompt = data.get('custom_prompt', '')
            model = data.get('model', 'gpt-4o')
            
            # Analyze dataset
            dataset_info = prompt_engine.analyze_dataset_for_prompting(df, selected_columns)
            
            # Execute prompt engineering
            start_time = time.time()
            analysis_results = prompt_engine.execute_prompt_engineering(
                dataset_info, prompt_type, custom_prompt, model
            )
            processing_time = round(time.time() - start_time, 2)
            
            # Generate ETL recommendations
            etl_recommendations = prompt_engine.generate_etl_recommendations(dataset_info, analysis_results)
            
            # Create enhanced dataset
            enhanced_df = prompt_engine.create_enhanced_dataset(df, analysis_results, selected_columns)
            
            # Store results
            analysis_id = str(uuid.uuid4())
            data_store[f"prompt_analysis_{analysis_id}"] = {
                'results': analysis_results,
                'enhanced_df': enhanced_df,
                'etl_recommendations': etl_recommendations,
                'dataset_info': dataset_info,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': filename
            }
            
            # Prepare response
            response = {
                'analysis_id': analysis_id,
                'results': analysis_results,
                'etl_recommendations': etl_recommendations,
                'dataset_info': dataset_info,
                'processing_time': processing_time,
                'data_preview': {
                    'columns': enhanced_df.columns.tolist(),
                    'data': enhanced_df.head(10).to_dict(orient='records'),
                    'shape': enhanced_df.shape
                }
            }
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error in execute_prompt_analysis: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/api/prompt-engineering/download', methods=['POST'])
    def download_prompt_results():
        try:
            data = request.json
            analysis_id = data.get('analysis_id')
            
            if not analysis_id:
                return jsonify({'error': 'Missing analysis_id'}), 400
            
            analysis_key = f"prompt_analysis_{analysis_id}"
            if analysis_key not in data_store:
                return jsonify({'error': 'Analysis results not found'}), 404
            
            analysis_data = data_store[analysis_key]
            enhanced_df = analysis_data['enhanced_df']
            
            # Create temporary file
            temp_filename = f"prompt_engineering_results_{analysis_id[:8]}.csv"
            temp_path = os.path.join('static/temp', temp_filename)
            
            # Ensure temp directory exists
            os.makedirs('static/temp', exist_ok=True)
            
            # Save to CSV
            enhanced_df.to_csv(temp_path, index=False)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in download_prompt_results: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500
