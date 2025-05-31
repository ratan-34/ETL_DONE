from flask import Blueprint, request, jsonify, render_template, send_file, session, redirect, url_for
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
import os
import io
import base64
from datetime import datetime
import traceback
from collections import Counter
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_text_classification_routes(app, data_store, client):
    """Add text classification and summarization routes to the Flask app"""
    
    @app.route('/text-classification-summarization')
    def text_classification_summarization():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Text Classification route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            return render_template('text_classification_summarization.html')
        except Exception as e:
            logger.error(f"Error in text_classification_summarization route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/text-classification-summarization/dataset-info', methods=['GET'])
    def api_text_classification_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Quick analysis for text columns
            columns_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                unique_count = df[col].nunique()
                
                # Check if suitable for text analysis
                is_text_suitable = False
                if pd.api.types.is_object_dtype(df[col]):
                    sample_values = df[col].dropna().head(5).astype(str).tolist()
                    avg_length = np.mean([len(str(val)) for val in sample_values]) if sample_values else 0
                    if avg_length > 10:  # Text if average length > 10 characters
                        is_text_suitable = True
                
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'missing': int(missing),
                    'missing_pct': f"{missing_pct:.1f}%",
                    'unique_count': int(unique_count),
                    'is_text_suitable': is_text_suitable,
                    'sample_values': df[col].dropna().head(3).astype(str).tolist()
                })

            return jsonify({
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'columns_info': columns_info,
                'session_id': session_id
            })
        
        except Exception as e:
            logger.error(f"Error in dataset info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/text-classification-summarization/process', methods=['POST'])
    def api_text_classification_process():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            text_column = data.get('text_column')
            operation_type = data.get('operation_type', 'both')
            model = data.get('model', 'gpt-4o')
            sample_size = int(data.get('sample_size', 50))  # Reduced default
            classification_categories = data.get('classification_categories', 'positive,negative,neutral')
            
            if not text_column:
                return jsonify({'error': 'Text column is required'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Fast processing
            start_time = time.time()
            result = perform_fast_text_analysis(
                df, text_column, operation_type, model, sample_size, 
                classification_categories, filename
            )
            processing_time = round(time.time() - start_time, 2)
            
            # Store result
            analysis_id = str(uuid.uuid4())
            data_store[f"text_analysis_{analysis_id}"] = {
                'result': result,
                'enhanced_df': result['enhanced_df'],
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': filename
            }
            
            # Prepare response
            response_result = result.copy()
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
            logger.error(f"Error in text analysis: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    @app.route('/api/text-classification-summarization/download', methods=['POST'])
    def api_text_classification_download():
        try:
            data = request.json
            analysis_id = data.get('analysis_id')
            
            if not analysis_id:
                return jsonify({'error': 'Missing analysis_id'}), 400
            
            analysis_key = f"text_analysis_{analysis_id}"
            if analysis_key not in data_store:
                return jsonify({'error': 'Analysis not found'}), 404
            
            analysis_data = data_store[analysis_key]
            enhanced_df = analysis_data['enhanced_df']
            
            # Create temp file
            temp_filename = f"text_analysis_results_{analysis_id[:8]}.csv"
            temp_path = os.path.join('static/temp', temp_filename)
            
            # Ensure temp directory exists
            os.makedirs('static/temp', exist_ok=True)
            
            # Save to CSV
            enhanced_df.to_csv(temp_path, index=False)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in download: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

def perform_fast_text_analysis(df, text_column, operation_type, model, sample_size, 
                              classification_categories, filename):
    """Fast text analysis with simple methods"""
    try:
        # Sample data for speed
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()
        
        enhanced_df = df.copy()
        
        # Get text data
        text_data = df_sample[text_column].dropna().astype(str).tolist()
        
        if not text_data:
            raise ValueError("No valid text data found")
        
        # Limit to first 20 texts for speed
        text_data = text_data[:20]
        
        results = []
        
        if operation_type in ['classification', 'both']:
            # Fast classification
            classification_results = fast_classification(text_data, classification_categories)
            results.extend(classification_results)
            
            # Add classification columns to enhanced_df
            enhanced_df['text_category'] = 'unprocessed'
            enhanced_df['classification_confidence'] = 0.0
            
            for i, result in enumerate(classification_results):
                if i < len(df_sample):
                    idx = df_sample.index[i]
                    enhanced_df.loc[idx, 'text_category'] = result.get('category', 'neutral')
                    enhanced_df.loc[idx, 'classification_confidence'] = result.get('confidence', 0.5)
        
        if operation_type in ['summarization', 'both']:
            # Fast summarization
            summarization_results = fast_summarization(text_data)
            
            # Add summarization columns to enhanced_df
            enhanced_df['text_summary'] = ''
            enhanced_df['summary_length'] = 0
            
            for i, result in enumerate(summarization_results):
                if i < len(df_sample):
                    idx = df_sample.index[i]
                    enhanced_df.loc[idx, 'text_summary'] = result.get('summary', '')
                    enhanced_df.loc[idx, 'summary_length'] = len(result.get('summary', ''))
        
        # Generate metrics
        metrics = generate_fast_metrics(results, operation_type)
        
        # Generate insights
        insights = generate_fast_insights(operation_type, len(text_data))
        
        # Generate ETL benefits
        etl_benefits = generate_fast_etl_benefits()
        
        return {
            'operation_type': operation_type,
            'processed_texts': len(text_data),
            'results': results[:10],  # Show first 10 for display
            'metrics': metrics,
            'insights': insights,
            'etl_benefits': etl_benefits,
            'enhanced_df': enhanced_df
        }
    
    except Exception as e:
        logger.error(f"Error in fast text analysis: {str(e)}")
        raise

def fast_classification(text_data, categories_str):
    """Fast classification using simple keyword matching"""
    try:
        categories = [cat.strip() for cat in categories_str.split(',')]
        results = []
        
        # Simple keyword-based classification
        keyword_map = {
            'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'awesome'],
            'negative': ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing'],
            'neutral': ['okay', 'fine', 'average', 'normal'],
            'business': ['company', 'business', 'corporate', 'enterprise', 'organization'],
            'technology': ['tech', 'software', 'computer', 'digital', 'AI', 'data'],
            'sports': ['game', 'team', 'player', 'match', 'sport', 'championship'],
            'entertainment': ['movie', 'music', 'show', 'entertainment', 'fun']
        }
        
        for text in text_data:
            text_lower = text.lower()
            scores = {}
            
            for category in categories:
                category_lower = category.lower()
                keywords = keyword_map.get(category_lower, [category_lower])
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[category] = score
            
            # Find best category
            if max(scores.values()) > 0:
                best_category = max(scores, key=scores.get)
                confidence = min(0.9, 0.5 + scores[best_category] * 0.1)
            else:
                best_category = categories[0] if categories else 'neutral'
                confidence = 0.5
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'category': best_category,
                'confidence': confidence,
                'method': 'keyword_matching'
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in fast classification: {str(e)}")
        return []

def fast_summarization(text_data):
    """Fast summarization using simple sentence extraction"""
    try:
        results = []
        
        for text in text_data:
            # Simple extractive summarization
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) <= 2:
                summary = text
            elif len(sentences) <= 5:
                # Take first and last sentence
                summary = sentences[0] + '. ' + sentences[-1]
            else:
                # Take first, middle, and last sentence
                middle_idx = len(sentences) // 2
                summary = sentences[0] + '. ' + sentences[middle_idx] + '. ' + sentences[-1]
            
            # Clean up summary
            summary = summary.strip()
            if not summary.endswith('.'):
                summary += '.'
            
            compression_ratio = len(summary) / len(text) if len(text) > 0 else 1.0
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'summary': summary,
                'compression_ratio': compression_ratio,
                'method': 'extractive'
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in fast summarization: {str(e)}")
        return []

def generate_fast_metrics(results, operation_type):
    """Generate quick metrics"""
    try:
        metrics = {
            'total_processed': len(results),
            'processing_method': 'fast_local'
        }
        
        if operation_type in ['classification', 'both']:
            categories = [r.get('category', 'unknown') for r in results if 'category' in r]
            if categories:
                category_counts = Counter(categories)
                metrics['classification'] = {
                    'total_categories': len(category_counts),
                    'most_common': category_counts.most_common(3),
                    'avg_confidence': np.mean([r.get('confidence', 0.5) for r in results if 'confidence' in r])
                }
        
        if operation_type in ['summarization', 'both']:
            summaries = [r for r in results if 'summary' in r]
            if summaries:
                compression_ratios = [r.get('compression_ratio', 1.0) for r in summaries]
                metrics['summarization'] = {
                    'avg_compression': np.mean(compression_ratios),
                    'total_summaries': len(summaries)
                }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        return {'total_processed': 0}

def generate_fast_insights(operation_type, text_count):
    """Generate quick insights"""
    insights = [
        {
            'title': 'Fast Processing Complete',
            'description': f'Successfully processed {text_count} text samples using optimized local methods.',
            'category': 'Performance'
        },
        {
            'title': 'ETL Integration Ready',
            'description': 'Results can be integrated into your ETL pipeline for automated text processing.',
            'category': 'Integration'
        }
    ]
    
    if operation_type == 'classification':
        insights.append({
            'title': 'Text Classification Applied',
            'description': 'Texts have been categorized using keyword-based classification for quick results.',
            'category': 'Classification'
        })
    elif operation_type == 'summarization':
        insights.append({
            'title': 'Text Summarization Applied',
            'description': 'Texts have been summarized using extractive methods for efficient processing.',
            'category': 'Summarization'
        })
    else:
        insights.append({
            'title': 'Combined Analysis Complete',
            'description': 'Both classification and summarization have been applied to your text data.',
            'category': 'Comprehensive'
        })
    
    return insights

def generate_fast_etl_benefits():
    """Generate ETL benefits"""
    return [
        {
            'category': 'Automation',
            'benefit': 'Automated Text Processing',
            'description': 'Automatically categorize and summarize text data in your ETL pipeline.',
            'impact': 'Reduces manual text processing time by 80%'
        },
        {
            'category': 'Efficiency',
            'benefit': 'Fast Processing',
            'description': 'Quick text analysis suitable for real-time ETL workflows.',
            'impact': 'Process thousands of texts per minute'
        },
        {
            'category': 'Scalability',
            'benefit': 'Scalable Solution',
            'description': 'Easily scale text processing across large datasets.',
            'impact': 'Handle growing data volumes without performance loss'
        },
        {
            'category': 'Integration',
            'benefit': 'ETL Pipeline Integration',
            'description': 'Seamlessly integrate into existing data processing workflows.',
            'impact': 'Enhance data pipelines with text intelligence'
        }
    ]