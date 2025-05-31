from flask import Blueprint, request, jsonify, render_template
import os
import base64
import json
import re
import uuid
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from datetime import datetime

# Azure OpenAI API configuration
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = "2023-05-15"  # Update this to the latest version

# Create Blueprint for object detection
object_detection_bp = Blueprint('object_detection', __name__)

class ObjectDetectionService:
    def __init__(self):
        self.models = {
            "azure_openai_default": {
                "name": "Azure OpenAI (Default)",
                "endpoint": AZURE_OPENAI_ENDPOINT,
                "deployment_id": "gpt-4-vision-preview",  # Update with your actual deployment ID
                "api_version": AZURE_OPENAI_API_VERSION
            },
            "azure_openai_enhanced": {
                "name": "Azure OpenAI (Enhanced)",
                "endpoint": AZURE_OPENAI_ENDPOINT,
                "deployment_id": "gpt-4-vision-preview",  # Update with your actual deployment ID
                "api_version": AZURE_OPENAI_API_VERSION
            },
            "custom_model": {
                "name": "Custom Model",
                "endpoint": None  # Would be configured for a custom model
            }
        }
        
    def detect_objects_in_image(self, image_data, model_id, confidence_threshold=50, enhanced_detection=False):
        """
        Detect objects in an image using Azure OpenAI API
        
        Args:
            image_data: Base64 encoded image data
            model_id: ID of the model to use
            confidence_threshold: Minimum confidence threshold (0-100)
            enhanced_detection: Whether to use enhanced detection features
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Remove the data URL prefix if present
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            
            # Get model configuration
            model_config = self.models.get(model_id)
            if not model_config:
                return {"error": "Invalid model ID"}
            
            # For Azure OpenAI models
            if "azure_openai" in model_id:
                return self._detect_with_azure_openai(
                    image_data, 
                    model_config, 
                    confidence_threshold, 
                    enhanced_detection
                )
            # For custom models
            elif model_id == "custom_model":
                return self._detect_with_custom_model(
                    image_data, 
                    confidence_threshold, 
                    enhanced_detection
                )
            else:
                return {"error": "Unsupported model"}
                
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return {"error": str(e)}
    
    def _detect_with_azure_openai(self, image_data, model_config, confidence_threshold, enhanced_detection):
        """
        Detect objects using Azure OpenAI Vision API
        """
        # Check if Azure OpenAI credentials are configured
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
            print("Azure OpenAI credentials not configured, using simulated results")
            return self._simulate_detection_results()
        
        try:
            # Prepare the API request
            headers = {
                "Content-Type": "application/json",
                "api-key": AZURE_OPENAI_KEY
            }
            
            # Create system message based on settings
            system_message = "You are an expert computer vision system specialized in object detection."
            if enhanced_detection:
                system_message += " Use enhanced detection to identify small or partially visible objects."
            
            # Create user message with specific instructions
            user_message = f"""
            Analyze this image and detect all objects. 
            For each object, provide:
            1. A class label
            2. A confidence score (0.0 to 1.0)
            3. A bounding box in the format [x, y, width, height] where x,y is the top-left corner
            
            Only include objects with confidence above {confidence_threshold/100}.
            
            Format your response as a valid JSON object with this structure:
            {{
                "detections": [
                    {{
                        "id": 1,
                        "class": "object_class",
                        "confidence": 0.95,
                        "bbox": {{
                            "x": 100,
                            "y": 150,
                            "width": 200,
                            "height": 300
                        }}
                    }},
                    ...
                ]
            }}
            
            Return ONLY the JSON with no additional text.
            """
            
            # Prepare the request payload
            payload = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 800
            }
            
            # Make the API request
            endpoint = f"{model_config['endpoint']}/openai/deployments/{model_config['deployment_id']}/chat/completions?api-version={model_config['api_version']}"
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return self._simulate_detection_results()
            
            # Parse the response
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            # Clean up the content
            content = content.strip()
            if content.startswith('```') and content.endswith('```'):
                content = content[3:-3].strip()
            
            # Parse the JSON response
            detection_data = json.loads(content)
            
            # Add timestamp and model info
            detection_data["timestamp"] = datetime.now().isoformat()
            detection_data["model"] = model_config["name"]
            detection_data["success"] = True
            
            return detection_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return self._simulate_detection_results()
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            return self._simulate_detection_results()
        except Exception as e:
            print(f"Azure OpenAI API error: {str(e)}")
            return self._simulate_detection_results()
    
    def _detect_with_custom_model(self, image_data, confidence_threshold, enhanced_detection):
        """
        Detect objects using a custom model (simulated for demo)
        """
        # This would be implemented with your custom model
        # For demo purposes, return simulated results
        return self._simulate_detection_results()
    
    def _simulate_detection_results(self):
        """
        Generate simulated detection results for demo purposes
        """
        # Create random detections
        detections = []
        num_detections = np.random.randint(3, 7)
        
        classes = ["Person", "Car", "Dog", "Cat", "Bicycle", "Chair", "Table", "Plant", "Book", "Phone"]
        
        for i in range(num_detections):
            # Random values for demonstration
            class_name = np.random.choice(classes)
            confidence = np.random.uniform(0.7, 0.98)
            
            # Random bounding box (these would be pixel values in a real scenario)
            x = np.random.randint(50, 400)
            y = np.random.randint(50, 300)
            width = np.random.randint(80, 250)
            height = np.random.randint(80, 250)
            
            detections.append({
                "id": i + 1,
                "class": class_name,
                "confidence": confidence,
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            })
        
        return {
            "detections": detections,
            "timestamp": datetime.now().isoformat(),
            "model": "Simulated Model (Demo)",
            "success": True
        }
    
    def analyze_tabular_data(self, file_data, file_type, model_id, confidence_threshold=50):
        """
        Analyze tabular data for object detection
        
        Args:
            file_data: CSV or Excel file data
            file_type: Type of file ('csv' or 'excel')
            model_id: ID of the model to use
            confidence_threshold: Minimum confidence threshold (0-100)
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load the data
            if file_type == 'csv':
                df = pd.read_csv(BytesIO(file_data))
            elif file_type == 'excel':
                df = pd.read_excel(BytesIO(file_data))
            else:
                return {"error": "Unsupported file type"}
            
            # For demo purposes, return simulated results
            # In a real application, this would analyze the dataframe
            return self._simulate_tabular_results(df)
            
        except Exception as e:
            print(f"Error in tabular data analysis: {str(e)}")
            return {"error": str(e)}
    
    def _simulate_tabular_results(self, df):
        """
        Generate simulated tabular analysis results for demo purposes
        """
        # Get basic dataframe info
        num_rows, num_cols = df.shape
        
        # Create simulated detections
        detections = []
        
        # Simulate anomaly detection
        for i in range(min(5, max(1, num_rows // 20))):
            row = np.random.randint(0, num_rows)
            col = np.random.randint(0, num_cols)
            col_name = df.columns[col]
            
            detection_type = np.random.choice([
                "Anomaly", "Outlier", "Missing Value", "Duplicate", 
                "Data Quality Issue", "Pattern Deviation"
            ])
            confidence = np.random.uniform(0.8, 0.99)
            
            # Get the actual value from the dataframe
            try:
                cell_value = df.iloc[row, col]
                if pd.isna(cell_value):
                    value_str = "NULL/NaN"
                else:
                    value_str = str(cell_value)[:50]  # Limit length
            except:
                value_str = "N/A"
            
            detections.append({
                "id": i + 1,
                "class": detection_type,
                "confidence": confidence,
                "location": f"Row {row + 1}, Column '{col_name}'",
                "value": value_str
            })
        
        return {
            "detections": detections,
            "timestamp": datetime.now().isoformat(),
            "model": "Tabular Analysis (Demo)",
            "success": True,
            "data_summary": {
                "rows": num_rows,
                "columns": num_cols,
                "column_names": list(df.columns),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        }

# Initialize the service
detection_service = ObjectDetectionService()

# Route to serve the Object Detection HTML page
@object_detection_bp.route('/object-detection')
def object_detection():
    """Serve the Object Detection HTML page"""
    return render_template('object-detection.html')

# API Routes
@object_detection_bp.route('/api/object-detection', methods=['POST'])
def detect_objects():
    """API endpoint for object detection"""
    try:
        data = request.json
        
        # Validate request data
        if not data:
            return jsonify({"error": "No data provided", "success": False}), 400
        
        # Get parameters from request
        image_data = data.get('imageData')
        model_id = data.get('model', 'azure_openai_default')
        confidence_threshold = int(data.get('confidenceThreshold', 50))
        enhanced_detection = data.get('enhancedDetection', False)
        
        # Check if we have image data
        if not image_data:
            return jsonify({"error": "No image data provided", "success": False}), 400
        
        # Process the image
        results = detection_service.detect_objects_in_image(
            image_data, 
            model_id, 
            confidence_threshold,
            enhanced_detection
        )
        
        # Ensure success flag is set
        if "error" not in results:
            results["success"] = True
        else:
            results["success"] = False
        
        return jsonify(results)
        
    except ValueError as e:
        print(f"Value error: {str(e)}")
        return jsonify({"error": "Invalid parameter values", "success": False}), 400
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}", "success": False}), 500

@object_detection_bp.route('/api/tabular-detection', methods=['POST'])
def analyze_tabular():
    """API endpoint for tabular data analysis"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file provided", "success": False}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected", "success": False}), 400
        
        # Get parameters
        model_id = request.form.get('model', 'azure_openai_default')
        confidence_threshold = int(request.form.get('confidenceThreshold', 50))
        
        # Determine file type
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            file_type = 'csv'
        elif filename.endswith(('.xlsx', '.xls')):
            file_type = 'excel'
        else:
            return jsonify({"error": "Unsupported file type. Please upload CSV or Excel files.", "success": False}), 400
        
        # Read file data
        file_data = file.read()
        
        if len(file_data) == 0:
            return jsonify({"error": "Empty file provided", "success": False}), 400
        
        # Process the tabular data
        results = detection_service.analyze_tabular_data(
            file_data,
            file_type,
            model_id,
            confidence_threshold
        )
        
        # Ensure success flag is set
        if "error" not in results:
            results["success"] = True
        else:
            results["success"] = False
        
        return jsonify(results)
        
    except ValueError as e:
        print(f"Value error: {str(e)}")
        return jsonify({"error": "Invalid parameter values", "success": False}), 400
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}", "success": False}), 500

@object_detection_bp.route('/api/object-detection/models', methods=['GET'])
def get_available_models():
    """API endpoint to get available models"""
    try:
        models = []
        for model_id, config in detection_service.models.items():
            models.append({
                "id": model_id,
                "name": config["name"],
                "available": True if model_id == "custom_model" or (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY) else False
            })
        
        return jsonify({
            "models": models,
            "success": True
        })
        
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@object_detection_bp.route('/api/object-detection/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Object Detection",
        "timestamp": datetime.now().isoformat(),
        "azure_configured": bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY),
        "success": True
    })

# Error handlers for the blueprint
@object_detection_bp.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found", "success": False}), 404

@object_detection_bp.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({"error": "Method not allowed", "success": False}), 405

@object_detection_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "success": False}), 500