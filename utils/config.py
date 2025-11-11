"""
Configuration module for OCR system.
Handles paths, environment variables, and system-specific settings.
"""

import os
import platform
import subprocess
from pathlib import Path


class Config:
    """Configuration class for OCR system."""
    
    # Python version requirement
    PYTHON_VERSION = "3.10"
    
    # Supported image formats
    SUPPORTED_FORMATS = ["png", "jpg", "jpeg"]
    MAX_FILE_SIZE_MB = 200
    
    # OCR languages
    LANGUAGES = {
        'en': 'English',
        'ru': 'Russian',
        'ch': 'Chinese (Simplified)',
    }
    
    @staticmethod
    def get_tesseract_path():
        """
        Get Tesseract executable path based on OS.
        Returns None if not found.
        """
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # Try common paths for macOS
            possible_paths = [
                "/opt/homebrew/bin/tesseract",  # M1/M2 Mac
                "/usr/local/bin/tesseract",     # Intel Mac or older
                "/opt/local/bin/tesseract",     # MacPorts
            ]
            
            # First, try to find using 'which'
            try:
                result = subprocess.run(
                    ["which", "tesseract"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if path and os.path.exists(path):
                        return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Fallback to checking common paths
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        elif system == "Windows":
            return r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        elif system == "Linux":
            return "/usr/bin/tesseract"
        
        return None
    
    @staticmethod
    def get_tesseract_version():
        """Get Tesseract version string."""
        tesseract_path = Config.get_tesseract_path()
        if not tesseract_path:
            return None
        
        try:
            result = subprocess.run(
                [tesseract_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Extract version from output (first line usually)
                version_line = result.stdout.split('\n')[0]
                return version_line.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return None
    
    @staticmethod
    def is_m1_mac():
        """Check if running on Apple Silicon (M1/M2/M3)."""
        if platform.system() != "Darwin":
            return False
        
        try:
            # Check processor architecture
            result = subprocess.run(
                ["uname", "-m"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                arch = result.stdout.strip()
                return arch == "arm64"
        except Exception:
            pass
        
        return False
    
    @staticmethod
    def get_env_vars():
        """Get recommended environment variables for macOS M1."""
        env_vars = {}
        
        if Config.is_m1_mac():
            env_vars.update({
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "MKL_SERVICE_FORCE_INTEL": "1",
                "OMP_NUM_THREADS": "4",
            })
        
        return env_vars

