"""
Google Cloud Storage Utilities for Model Persistence
Provides functions to save and load models from GCS in Cloud Run environment
"""
import os
import json
import tempfile
from typing import Optional, Any, Dict, List
from datetime import datetime
import joblib

# Check if running in cloud environment
IS_CLOUD_ENVIRONMENT = os.getenv("GOOGLE_CLOUD_PROJECT") is not None or os.getenv("GCS_BUCKET_NAME") is not None

if IS_CLOUD_ENVIRONMENT:
    try:
        from google.cloud import storage
        GCS_AVAILABLE = True
    except ImportError:
        GCS_AVAILABLE = False
        print("Warning: google-cloud-storage not installed. GCS features disabled.")
else:
    GCS_AVAILABLE = False

# Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "stocktime-predictor-models")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/app/models")


def get_storage_client():
    """Get GCS client, returns None if not available."""
    if not GCS_AVAILABLE:
        return None
    try:
        return storage.Client()
    except Exception as e:
        print(f"Warning: Could not create GCS client: {e}")
        return None


def get_bucket():
    """Get the GCS bucket for model storage."""
    client = get_storage_client()
    if client is None:
        return None
    try:
        return client.bucket(GCS_BUCKET_NAME)
    except Exception as e:
        print(f"Warning: Could not access bucket {GCS_BUCKET_NAME}: {e}")
        return None


def save_model_to_gcs(
    model: Any,
    model_type: str,
    ticker: str,
    metadata: Optional[Dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    Save a trained model to GCS or local storage.
    
    Args:
        model: The trained model object (must be joblib serializable)
        model_type: Type of model ('rf', 'lstm', 'xgboost', 'prophet')
        ticker: Stock ticker symbol
        metadata: Optional metadata dict to save alongside model
        start_date: Training data start date
        end_date: Training data end date
    
    Returns:
        str: Path/URI where model was saved
    """
    # Generate model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if start_date and end_date:
        model_name = f"{model_type}_model_{ticker}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.joblib"
    else:
        model_name = f"{model_type}_model_{ticker}_{timestamp}.joblib"
    
    # Also save as "latest" for easy retrieval
    latest_name = f"{model_type}_model_{ticker}_latest.joblib"
    
    gcs_path = f"{model_type}_models/{ticker}/{model_name}"
    latest_path = f"{model_type}_models/{ticker}/{latest_name}"
    
    bucket = get_bucket()
    
    if bucket is not None:
        # Save to GCS
        try:
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                joblib.dump(model, tmp.name)
                
                # Upload model file
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(tmp.name)
                
                # Also upload as "latest"
                latest_blob = bucket.blob(latest_path)
                latest_blob.upload_from_filename(tmp.name)
                
                os.unlink(tmp.name)
            
            # Save metadata if provided
            if metadata:
                metadata['saved_at'] = datetime.now().isoformat()
                metadata['model_path'] = gcs_path
                metadata_path = gcs_path.replace('.joblib', '_metadata.json')
                metadata_blob = bucket.blob(metadata_path)
                metadata_blob.upload_from_string(
                    json.dumps(metadata, indent=2),
                    content_type='application/json'
                )
            
            print(f"Model saved to GCS: gs://{GCS_BUCKET_NAME}/{gcs_path}")
            return f"gs://{GCS_BUCKET_NAME}/{gcs_path}"
            
        except Exception as e:
            print(f"Error saving to GCS: {e}. Falling back to local storage.")
    
    # Fallback to local storage
    local_dir = os.path.join(LOCAL_MODEL_DIR, model_type, ticker)
    os.makedirs(local_dir, exist_ok=True)
    
    local_path = os.path.join(local_dir, model_name)
    latest_local_path = os.path.join(local_dir, latest_name)
    
    joblib.dump(model, local_path)
    joblib.dump(model, latest_local_path)
    
    if metadata:
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['model_path'] = local_path
        metadata_path = local_path.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Model saved locally: {local_path}")
    return local_path


def load_model_from_gcs(
    model_type: str,
    ticker: str,
    specific_model: Optional[str] = None
) -> Optional[Any]:
    """
    Load a model from GCS or local storage.
    
    Args:
        model_type: Type of model ('rf', 'lstm', 'xgboost', 'prophet')
        ticker: Stock ticker symbol
        specific_model: Optional specific model filename to load
    
    Returns:
        The loaded model object, or None if not found
    """
    if specific_model:
        gcs_path = f"{model_type}_models/{ticker}/{specific_model}"
    else:
        gcs_path = f"{model_type}_models/{ticker}/{model_type}_model_{ticker}_latest.joblib"
    
    bucket = get_bucket()
    
    if bucket is not None:
        try:
            blob = bucket.blob(gcs_path)
            if blob.exists():
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                    blob.download_to_filename(tmp.name)
                    model = joblib.load(tmp.name)
                    os.unlink(tmp.name)
                print(f"Model loaded from GCS: gs://{GCS_BUCKET_NAME}/{gcs_path}")
                return model
            else:
                print(f"Model not found in GCS: {gcs_path}")
        except Exception as e:
            print(f"Error loading from GCS: {e}. Trying local storage.")
    
    # Fallback to local storage
    if specific_model:
        local_path = os.path.join(LOCAL_MODEL_DIR, model_type, ticker, specific_model)
    else:
        local_path = os.path.join(LOCAL_MODEL_DIR, model_type, ticker, f"{model_type}_model_{ticker}_latest.joblib")
    
    if os.path.exists(local_path):
        model = joblib.load(local_path)
        print(f"Model loaded from local: {local_path}")
        return model
    
    print(f"Model not found: {local_path}")
    return None


def list_models_in_gcs(model_type: str, ticker: Optional[str] = None) -> List[Dict]:
    """
    List all available models in GCS for a given type and optionally ticker.
    
    Returns:
        List of dicts with model info (name, path, size, updated)
    """
    models = []
    prefix = f"{model_type}_models/"
    if ticker:
        prefix += f"{ticker}/"
    
    bucket = get_bucket()
    
    if bucket is not None:
        try:
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                if blob.name.endswith('.joblib'):
                    models.append({
                        'name': os.path.basename(blob.name),
                        'path': f"gs://{GCS_BUCKET_NAME}/{blob.name}",
                        'size_mb': round(blob.size / (1024 * 1024), 2) if blob.size else 0,
                        'updated': blob.updated.isoformat() if blob.updated else None
                    })
        except Exception as e:
            print(f"Error listing GCS models: {e}")
    
    # Also check local storage
    local_dir = os.path.join(LOCAL_MODEL_DIR, model_type)
    if ticker:
        local_dir = os.path.join(local_dir, ticker)
    
    if os.path.exists(local_dir):
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith('.joblib'):
                    file_path = os.path.join(root, file)
                    models.append({
                        'name': file,
                        'path': file_path,
                        'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                        'updated': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    })
    
    return models


def load_model_metadata(model_type: str, ticker: str) -> Optional[Dict]:
    """Load metadata for the latest model."""
    gcs_path = f"{model_type}_models/{ticker}/{model_type}_model_{ticker}_latest_metadata.json"
    
    bucket = get_bucket()
    
    if bucket is not None:
        try:
            blob = bucket.blob(gcs_path)
            if blob.exists():
                content = blob.download_as_string()
                return json.loads(content)
        except Exception as e:
            print(f"Error loading metadata from GCS: {e}")
    
    # Fallback to local
    local_path = os.path.join(LOCAL_MODEL_DIR, model_type, ticker, f"{model_type}_model_{ticker}_latest_metadata.json")
    if os.path.exists(local_path):
        with open(local_path, 'r') as f:
            return json.load(f)
    
    return None


def is_cloud_environment() -> bool:
    """Check if running in cloud environment."""
    return IS_CLOUD_ENVIRONMENT and GCS_AVAILABLE
