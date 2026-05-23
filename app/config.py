import os

class Config:
    """Configuration class for the ASR Application."""
    
    # Paths and Models
    ONNX_MODEL_PATH = os.environ.get('ONNX_MODEL_PATH', os.path.join(os.path.dirname(__file__), '../models/qn_final.onnx'))
    NEMO_MODEL_NAME = os.environ.get('NEMO_MODEL_NAME', 'QuartzNet15x5Base-En')
    
    # Inference Configuration
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
    
    # API Validation Limits
    MAX_FILE_SIZE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', 50))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Server Configuration
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
