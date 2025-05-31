from flask import Flask, request, jsonify, render_template, session, redirect, url_for
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringAutoMLEngine:
    """Advanced Clustering AutoML Engine with Visual Analytics"""
    
    def __init__(self, azure_openai_client=None):
        self.client = azure_openai_client
        self.algorithms = {
            'kmeans': {
                'name': 'K-Means Clustering',
                'class': KMeans,
                'params': {'n_clusters': [3, 4, 5, 6, 7, 8], 'random_state': 42},
                'description': 'Partitions data into k clusters by minimizing within-cluster sum of squares'
            },
            'dbscan': {
                'name': 'DBSCAN',
                'class': DBSCAN,
                'params': {'eps': [0.3, 0.5, 0.7, 1.0], 'min_samples': [3, 5, 7]},
                'description': 'Density-based clustering that finds clusters of varying shapes'
            },
            'hierarchical': {
                'name': 'Hierarchical Clustering',
                'class': AgglomerativeClustering,
                'params': {'n_clusters': [3, 4, 5, 6, 7, 8], 'linkage': ['ward', 'complete']},
                'description': 'Creates a hierarchy of clusters using linkage criteria'
            }
        }
    
    def preprocess_data(self, df, selected_columns):
        """Intelligent data preprocessing for clustering"""
        try:
            # Select and clean data
            data = df[selected_columns].copy()
            
            # Handle missing values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Fill missing values
            for col in numeric_cols:
                data[col].fillna(data[col].median(), inplace=True)
            
            for col in categorical_cols:
                data[col].fillna(data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'Unknown', inplace=True)
            
            # Encode categorical variables
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                label_encoders[col] = le
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            return scaled_data, data, scaler, label_encoders
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def auto_select_algorithm(self, X, max_clusters=8):
        """Automatically select the best clustering algorithm"""
        try:
            best_algorithm = None
            best_score = -1
            best_params = None
            best_labels = None
            
            results = {}
            
            for algo_name, algo_config in self.algorithms.items():
                logger.info(f"Testing {algo_name}...")
                
                if algo_name == 'kmeans':
                    # Test different k values
                    for k in algo_config['params']['n_clusters']:
                        if k <= len(X):
                            model = KMeans(n_clusters=k, random_state=42, n_init=10)
                            labels = model.fit_predict(X)
                            
                            if len(np.unique(labels)) > 1:
                                score = silhouette_score(X, labels)
                                results[f"{algo_name}_k{k}"] = {
                                    'score': score,
                                    'labels': labels,
                                    'model': model,
                                    'params': {'n_clusters': k}
                                }
                                
                                if score > best_score:
                                    best_score = score
                                    best_algorithm = algo_name
                                    best_params = {'n_clusters': k}
                                    best_labels = labels
                
                elif algo_name == 'dbscan':
                    # Test different eps and min_samples
                    for eps in algo_config['params']['eps']:
                        for min_samples in algo_config['params']['min_samples']:
                            model = DBSCAN(eps=eps, min_samples=min_samples)
                            labels = model.fit_predict(X)
                            
                            if len(np.unique(labels)) > 1 and -1 not in labels:
                                score = silhouette_score(X, labels)
                                results[f"{algo_name}_eps{eps}_min{min_samples}"] = {
                                    'score': score,
                                    'labels': labels,
                                    'model': model,
                                    'params': {'eps': eps, 'min_samples': min_samples}
                                }
                                
                                if score > best_score:
                                    best_score = score
                                    best_algorithm = algo_name
                                    best_params = {'eps': eps, 'min_samples': min_samples}
                                    best_labels = labels
                
                elif algo_name == 'hierarchical':
                    # Test different cluster numbers and linkages
                    for k in algo_config['params']['n_clusters']:
                        for linkage in algo_config['params']['linkage']:
                            if k <= len(X):
                                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                                labels = model.fit_predict(X)
                                
                                if len(np.unique(labels)) > 1:
                                    score = silhouette_score(X, labels)
                                    results[f"{algo_name}_k{k}_{linkage}"] = {
                                        'score': score,
                                        'labels': labels,
                                        'model': model,
                                        'params': {'n_clusters': k, 'linkage': linkage}
                                    }
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_algorithm = algo_name
                                        best_params = {'n_clusters': k, 'linkage': linkage}
                                        best_labels = labels
            
            return {
                'best_algorithm': best_algorithm,
                'best_score': best_score,
                'best_params': best_params,
                'best_labels': best_labels,
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"Error in algorithm selection: {str(e)}")
            # Fallback to simple K-means
            model = KMeans(n_clusters=3, random_state=42)
            labels = model.fit_predict(X)
            return {
                'best_algorithm': 'kmeans',
                'best_score': 0.5,
                'best_params': {'n_clusters': 3},
                'best_labels': labels,
                'all_results': {}
            }
    
    def create_visualizations(self, X, labels, original_data, selected_columns):
        """Create comprehensive clustering visualizations"""
        try:
            visualizations = []
            
            # 1. PCA 2D Scatter Plot
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
            else:
                X_pca = X
                pca = None
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter)
            plt.title('Clustering Results - PCA Visualization', fontsize=16, fontweight='bold')
            plt.xlabel('First Principal Component' if pca else selected_columns[0])
            plt.ylabel('Second Principal Component' if pca else selected_columns[1])
            plt.grid(True, alpha=0.3)
            
            # Add cluster centers if available
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label != -1:  # Skip noise points in DBSCAN
                    cluster_center = X_pca[labels == label].mean(axis=0)
                    plt.scatter(cluster_center[0], cluster_center[1], 
                              marker='x', s=200, c='red', linewidths=3)
            
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            pca_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            visualizations.append({
                'type': 'pca_scatter',
                'title': 'PCA Clustering Visualization',
                'data': pca_plot,
                'description': 'Principal Component Analysis view of clusters'
            })
            
            # 2. Cluster Size Distribution
            plt.figure(figsize=(10, 6))
            unique_labels, counts = np.unique(labels, return_counts=True)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            bars = plt.bar(range(len(unique_labels)), counts, color=colors, alpha=0.8)
            plt.title('Cluster Size Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Data Points')
            plt.xticks(range(len(unique_labels)), [f'Cluster {i}' for i in unique_labels])
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            dist_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            visualizations.append({
                'type': 'cluster_distribution',
                'title': 'Cluster Size Distribution',
                'data': dist_plot,
                'description': 'Distribution of data points across clusters'
            })
            
            # 3. Feature Importance Heatmap
            if len(selected_columns) > 1:
                plt.figure(figsize=(12, 8))
                
                # Calculate cluster centers for original features
                cluster_centers = []
                for label in unique_labels:
                    if label != -1:
                        cluster_data = original_data[labels == label]
                        centers = cluster_data.mean()
                        cluster_centers.append(centers)
                
                if cluster_centers:
                    centers_df = pd.DataFrame(cluster_centers, 
                                            columns=selected_columns,
                                            index=[f'Cluster {i}' for i in unique_labels if i != -1])
                    
                    sns.heatmap(centers_df.T, annot=True, cmap='RdYlBu_r', center=0, 
                               fmt='.2f', cbar_kws={'label': 'Feature Value'})
                    plt.title('Cluster Characteristics Heatmap', fontsize=16, fontweight='bold')
                    plt.xlabel('Clusters')
                    plt.ylabel('Features')
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                    buffer.seek(0)
                    heatmap_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    visualizations.append({
                        'type': 'feature_heatmap',
                        'title': 'Cluster Characteristics',
                        'data': heatmap_plot,
                        'description': 'Average feature values for each cluster'
                    })
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return []
    
    def analyze_clusters(self, original_data, labels, selected_columns):
        """Analyze cluster characteristics and generate insights"""
        try:
            cluster_analysis = []
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points in DBSCAN
                    continue
                
                cluster_data = original_data[labels == label]
                cluster_info = {
                    'cluster_id': int(label),
                    'size': len(cluster_data),
                    'percentage': f"{(len(cluster_data) / len(original_data)) * 100:.1f}%",
                    'characteristics': {}
                }
                
                # Analyze each feature
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(original_data[col]):
                        cluster_info['characteristics'][col] = {
                            'mean': float(cluster_data[col].mean()),
                            'std': float(cluster_data[col].std()),
                            'min': float(cluster_data[col].min()),
                            'max': float(cluster_data[col].max())
                        }
                    else:
                        # For categorical data, show most common values
                        value_counts = cluster_data[col].value_counts().head(3)
                        cluster_info['characteristics'][col] = {
                            'most_common': value_counts.to_dict()
                        }
                
                cluster_analysis.append(cluster_info)
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"Error in cluster analysis: {str(e)}")
            return []
    
    def generate_llm_insights(self, cluster_analysis, algorithm_info, selected_columns):
        """Generate AI-powered insights about clustering results"""
        try:
            if not self.client:
                return self.generate_fallback_insights(cluster_analysis, algorithm_info)
            
            # Prepare data for LLM
            summary = {
                'algorithm_used': algorithm_info['best_algorithm'],
                'silhouette_score': algorithm_info['best_score'],
                'num_clusters': len(cluster_analysis),
                'features_analyzed': selected_columns,
                'cluster_sizes': [c['size'] for c in cluster_analysis]
            }
            
            prompt = f"""
            You are an expert data scientist analyzing clustering results for ETL and business intelligence.
            
            Clustering Analysis Summary:
            {json.dumps(summary, indent=2)}
            
            Cluster Details:
            {json.dumps(cluster_analysis[:3], indent=2)}  # First 3 clusters for brevity
            
            Provide 5-7 strategic insights covering:
            1. Quality assessment of the clustering results
            2. Business interpretation of discovered segments
            3. Actionable recommendations for each cluster
            4. ETL pipeline integration opportunities
            5. Data-driven decision making suggestions
            6. Customer segmentation strategies (if applicable)
            7. Performance optimization recommendations
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed business-focused explanation",
                        "category": "Quality|Business|ETL|Strategy|Performance",
                        "priority": "High|Medium|Low",
                        "actionable": true/false
                    }}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist and business analyst specializing in clustering and customer segmentation. Provide strategic insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            return ai_response.get('insights', [])
            
        except Exception as e:
            logger.error(f"Error generating LLM insights: {str(e)}")
            return self.generate_fallback_insights(cluster_analysis, algorithm_info)
    
    def generate_fallback_insights(self, cluster_analysis, algorithm_info):
        """Generate fallback insights when LLM is not available"""
        insights = []
        
        # Quality assessment
        silhouette_score = algorithm_info['best_score']
        if silhouette_score > 0.7:
            quality = "Excellent"
        elif silhouette_score > 0.5:
            quality = "Good"
        elif silhouette_score > 0.3:
            quality = "Fair"
        else:
            quality = "Poor"
        
        insights.append({
            'title': f'{quality} Clustering Quality',
            'description': f'Silhouette score of {silhouette_score:.3f} indicates {quality.lower()} cluster separation and cohesion.',
            'category': 'Quality',
            'priority': 'High',
            'actionable': True
        })
        
        # Business insights
        if len(cluster_analysis) >= 3:
            insights.append({
                'title': 'Customer Segmentation Opportunity',
                'description': f'Identified {len(cluster_analysis)} distinct segments that can be used for targeted marketing and personalized experiences.',
                'category': 'Business',
                'priority': 'High',
                'actionable': True
            })
        
        # ETL integration
        insights.append({
            'title': 'ETL Pipeline Integration',
            'description': 'Clustering results can be integrated into ETL workflows for automated customer segmentation and real-time classification.',
            'category': 'ETL',
            'priority': 'Medium',
            'actionable': True
        })
        
        return insights

def add_clustering_routes(app, data_store, azure_client):
    """Add clustering routes to the Flask app"""
    
    clustering_engine = ClusteringAutoMLEngine(azure_client)
    
    @app.route('/clustering-automl-visual')
    def clustering_automl_visual():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Clustering AutoML Visual route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            # Set session for this tab
            session['session_id'] = session_id
            logger.info(f"Session set for Clustering AutoML Visual: {session_id}")
            return render_template('clustering-automl-visual.html')
        except Exception as e:
            logger.error(f"Error in clustering_automl_visual route: {str(e)}")
            return redirect(url_for('index'))
    
    @app.route('/api/clustering/dataset-info', methods=['GET'])
    def api_clustering_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Clustering dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Analyze columns for clustering suitability
            columns_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                unique_count = df[col].nunique()
                
                # Determine clustering suitability
                clustering_suitability = "High"
                if pd.api.types.is_numeric_dtype(df[col]):
                    if unique_count < 3:
                        clustering_suitability = "Low"
                    elif missing_pct > 50:
                        clustering_suitability = "Medium"
                elif pd.api.types.is_object_dtype(df[col]):
                    if unique_count > len(df) * 0.8:
                        clustering_suitability = "Low"
                    elif unique_count < 10:
                        clustering_suitability = "High"
                    else:
                        clustering_suitability = "Medium"
                
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
                    'clustering_suitability': clustering_suitability,
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
            logger.error(f"Error in api_clustering_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
    @app.route('/api/clustering/analyze', methods=['POST'])
    def clustering_analyze():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            selected_columns = data.get('selected_columns', [])
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No data found. Please upload a dataset first.'}), 400
            
            if not selected_columns:
                return jsonify({'error': 'Please select columns for clustering analysis.'}), 400
            
            df = data_store[session_id]['df']
            
            # Preprocessing
            start_time = time.time()
            X_scaled, X_original, scaler, label_encoders = clustering_engine.preprocess_data(df, selected_columns)
            
            # Auto-select best algorithm
            algorithm_results = clustering_engine.auto_select_algorithm(X_scaled)
            
            # Create visualizations
            visualizations = clustering_engine.create_visualizations(
                X_scaled, algorithm_results['best_labels'], X_original, selected_columns
            )
            
            # Analyze clusters
            cluster_analysis = clustering_engine.analyze_clusters(
                X_original, algorithm_results['best_labels'], selected_columns
            )
            
            # Generate AI insights
            ai_insights = clustering_engine.generate_llm_insights(
                cluster_analysis, algorithm_results, selected_columns
            )
            
            processing_time = round(time.time() - start_time, 2)
            
            # Create enhanced dataset
            enhanced_df = df.copy()
            enhanced_df['cluster_id'] = algorithm_results['best_labels']
            enhanced_df['clustering_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Store results
            analysis_id = str(uuid.uuid4())
            data_store[f"clustering_{analysis_id}"] = {
                'enhanced_df': enhanced_df,
                'results': {
                    'algorithm_info': algorithm_results,
                    'cluster_analysis': cluster_analysis,
                    'visualizations': visualizations,
                    'ai_insights': ai_insights,
                    'processing_time': processing_time
                },
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return jsonify({
                'analysis_id': analysis_id,
                'algorithm_used': algorithm_results['best_algorithm'],
                'silhouette_score': algorithm_results['best_score'],
                'num_clusters': len(cluster_analysis),
                'cluster_analysis': cluster_analysis,
                'visualizations': visualizations,
                'ai_insights': ai_insights,
                'processing_time': processing_time,
                'etl_benefits': get_clustering_etl_benefits()
            })
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/api/clustering/download', methods=['POST'])
    def clustering_download():
        try:
            data = request.json
            analysis_id = data.get('analysis_id')
            
            if not analysis_id:
                return jsonify({'error': 'Analysis ID required'}), 400
            
            clustering_key = f"clustering_{analysis_id}"
            if clustering_key not in data_store:
                return jsonify({'error': 'Analysis not found'}), 404
            
            enhanced_df = data_store[clustering_key]['enhanced_df']
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            enhanced_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            return jsonify({
                'filename': f'clustering_results_{analysis_id[:8]}.csv',
                'data': csv_data,
                'rows': len(enhanced_df),
                'columns': len(enhanced_df.columns)
            })
            
        except Exception as e:
            logger.error(f"Error in clustering download: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

def get_clustering_etl_benefits():
    """Get ETL benefits of clustering analysis"""
    return [
        {
            'category': 'Customer Segmentation',
            'benefit': 'Automated Customer Grouping',
            'description': 'Automatically segment customers for targeted marketing and personalized experiences.',
            'implementation': 'Integrate clustering into customer data pipeline for real-time segmentation'
        },
        {
            'category': 'Data Quality',
            'benefit': 'Anomaly Detection',
            'description': 'Identify outliers and anomalous data points that may indicate data quality issues.',
            'implementation': 'Use clustering to flag unusual patterns in ETL data validation'
        },
        {
            'category': 'Performance Optimization',
            'benefit': 'Data Partitioning',
            'description': 'Use cluster assignments to optimize data storage and query performance.',
            'implementation': 'Partition datasets based on cluster membership for faster processing'
        },
        {
            'category': 'Business Intelligence',
            'benefit': 'Pattern Discovery',
            'description': 'Discover hidden patterns and relationships in business data.',
            'implementation': 'Incorporate clustering insights into BI dashboards and reports'
        }
    ]
