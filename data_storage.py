data_store = {}
"""
Data storage and session management module
"""
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataStore:
    """Thread-safe data storage for managing user sessions and datasets"""
    
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()
        logger.info("DataStore initialized")
    
    def create_session(self, df: pd.DataFrame, filename: str, file_path: str) -> str:
        """Create a new session with uploaded data"""
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self._data[session_id] = {
                'df': df,
                'filename': filename,
                'file_path': file_path,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processed_data': {},
                'models': {},
                'analyses': {}
            }
        
        logger.info(f"New session created: {session_id} for file: {filename}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID"""
        with self._lock:
            return self._data.get(session_id)
    
    def get_dataframe(self, session_id: str) -> Optional[pd.DataFrame]:
        """Get the main dataframe for a session"""
        session = self.get_session(session_id)
        return session['df'] if session else None
    
    def update_session(self, session_id: str, key: str, value: Any) -> bool:
        """Update a specific key in session data"""
        with self._lock:
            if session_id in self._data:
                self._data[session_id][key] = value
                logger.info(f"Updated session {session_id}: {key}")
                return True
            return False
    
    def store_processed_data(self, session_id: str, process_name: str, data: Any) -> bool:
        """Store processed data for a session"""
        with self._lock:
            if session_id in self._data:
                if 'processed_data' not in self._data[session_id]:
                    self._data[session_id]['processed_data'] = {}
                self._data[session_id]['processed_data'][process_name] = data
                logger.info(f"Stored processed data for session {session_id}: {process_name}")
                return True
            return False
    
    def store_model(self, session_id: str, model_name: str, model_data: Dict[str, Any]) -> bool:
        """Store trained model data for a session"""
        with self._lock:
            if session_id in self._data:
                if 'models' not in self._data[session_id]:
                    self._data[session_id]['models'] = {}
                self._data[session_id]['models'][model_name] = model_data
                logger.info(f"Stored model for session {session_id}: {model_name}")
                return True
            return False
    
    def store_analysis(self, session_id: str, analysis_name: str, analysis_data: Dict[str, Any]) -> bool:
        """Store analysis results for a session"""
        with self._lock:
            if session_id in self._data:
                if 'analyses' not in self._data[session_id]:
                    self._data[session_id]['analyses'] = {}
                self._data[session_id]['analyses'][analysis_name] = analysis_data
                logger.info(f"Stored analysis for session {session_id}: {analysis_name}")
                return True
            return False
    
    def get_processed_data(self, session_id: str, process_name: str) -> Optional[Any]:
        """Get processed data for a session"""
        session = self.get_session(session_id)
        if session and 'processed_data' in session:
            return session['processed_data'].get(process_name)
        return None
    
    def get_model(self, session_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model data for a session"""
        session = self.get_session(session_id)
        if session and 'models' in session:
            return session['models'].get(model_name)
        return None
    
    def get_analysis(self, session_id: str, analysis_name: str) -> Optional[Dict[str, Any]]:
        """Get analysis data for a session"""
        session = self.get_session(session_id)
        if session and 'analyses' in session:
            return session['analyses'].get(analysis_name)
        return None
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active sessions with basic info"""
        with self._lock:
            return {
                sid: {
                    'filename': data['filename'],
                    'upload_time': data['upload_time'],
                    'shape': data['df'].shape if 'df' in data else None
                }
                for sid, data in self._data.items()
            }
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data"""
        with self._lock:
            if session_id in self._data:
                del self._data[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than specified hours"""
        # This would be implemented with proper datetime comparison
        # For now, just return 0 as no cleanup performed
        return 0
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        df = session.get('df')
        if df is None:
            return None
        
        return {
            'session_id': session_id,
            'filename': session.get('filename', 'Unknown'),
            'upload_time': session.get('upload_time'),
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'shape': df.shape,
            'has_models': bool(session.get('models', {})),
            'has_analyses': bool(session.get('analyses', {}))
        }

# Global data store instance
data_store = DataStore()
