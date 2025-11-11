"""
OCR Engine Manager - handles initialization and execution of different OCR engines.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pytesseract

from .config import Config


class OCREngineManager:
    """Manages multiple OCR engines and their execution."""
    
    def __init__(self):
        self.engines = {}
        self.engine_status = {}
        self.tesseract_path = Config.get_tesseract_path()
        
        # Set Tesseract path if found
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available OCR engines (English only)."""
        # PaddleOCR (English only)
        try:
            from paddleocr import PaddleOCR, PPStructure
            self.engines['paddleocr_en'] = {
                'type': 'paddleocr',
                'lang': 'en',
                'instance': None,
            }
            self.engines['ppstructure'] = {
                'type': 'ppstructure',
                'instance': None,
            }
            self.engine_status['paddleocr'] = True
        except Exception as e:
            self.engine_status['paddleocr'] = False
            print(f"PaddleOCR initialization error: {e}")
        
        # EasyOCR (English only)
        try:
            import easyocr
            self.engines['easyocr_en'] = {
                'type': 'easyocr',
                'langs': ['en'],
                'instance': None,
            }
            self.engine_status['easyocr'] = True
        except Exception as e:
            self.engine_status['easyocr'] = False
            print(f"EasyOCR initialization error: {e}")
        
        # Tesseract
        try:
            version = pytesseract.get_tesseract_version()
            self.engine_status['tesseract'] = True
            self.engine_status['tesseract_version'] = version
        except Exception:
            self.engine_status['tesseract'] = False
    
    def load_paddleocr(self, lang: str = 'en'):
        """Load PaddleOCR model (cached)."""
        if not self.engine_status.get('paddleocr', False):
            return None
        
        key = f'paddleocr_{lang}'
        if key not in self.engines:
            return None
        
        if self.engines[key]['instance'] is None:
            try:
                from paddleocr import PaddleOCR
                self.engines[key]['instance'] = PaddleOCR(
                    lang=lang,
                    use_angle_cls=True,
                    use_gpu=False,
                    show_log=False
                )
            except Exception as e:
                print(f"Error loading PaddleOCR {lang}: {e}")
                return None
        
        return self.engines[key]['instance']
    
    def load_ppstructure(self):
        """Load PPStructure model (cached)."""
        if not self.engine_status.get('paddleocr', False):
            return None
        
        key = 'ppstructure'
        if self.engines[key]['instance'] is None:
            try:
                from paddleocr import PPStructure
                self.engines[key]['instance'] = PPStructure(
                    lang='en',
                    layout=True,
                    use_gpu=False,
                    recovery=True,
                    return_ocr_result_in_table=True,
                    show_log=False
                )
            except Exception as e:
                print(f"Error loading PPStructure: {e}")
                return None
        
        return self.engines[key]['instance']
    
    def load_easyocr(self, langs: List[str]):
        """Load EasyOCR reader (cached)."""
        if not self.engine_status.get('easyocr', False):
            return None
        
        # Find matching engine config
        key = None
        for k, v in self.engines.items():
            if v.get('type') == 'easyocr' and v.get('langs') == langs:
                key = k
                break
        
        if not key:
            return None
        
        if self.engines[key]['instance'] is None:
            try:
                import easyocr
                self.engines[key]['instance'] = easyocr.Reader(
                    langs,
                    gpu=False,
                    download_enabled=True
                )
            except Exception as e:
                print(f"Error loading EasyOCR {langs}: {e}")
                return None
        
        return self.engines[key]['instance']
    
    def recognize_paddleocr(self, image: np.ndarray, lang: str = 'en') -> Tuple[List[str], float]:
        """
        Recognize text using PaddleOCR.
        Returns (text_lines, processing_time).
        """
        ocr = self.load_paddleocr(lang)
        if not ocr:
            return [], 0.0
        
        start_time = time.time()
        try:
            result = ocr.ocr(image, cls=True)
            processing_time = time.time() - start_time
            
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0].strip()
                        if text:
                            text_lines.append(text)
            
            return text_lines, processing_time
        except Exception as e:
            print(f"PaddleOCR recognition error: {e}")
            return [], time.time() - start_time
    
    def recognize_tesseract(self, image: np.ndarray, langs: str = "eng") -> Tuple[List[str], float]:
        """
        Recognize text using Tesseract.
        Returns (text_lines, processing_time).
        """
        if not self.engine_status.get('tesseract', False):
            return [], 0.0
        
        start_time = time.time()
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            raw_text = pytesseract.image_to_string(
                image_rgb,
                lang=langs,
                config="--oem 1 --psm 6"
            )
            processing_time = time.time() - start_time
            
            text_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            return text_lines, processing_time
        except Exception as e:
            print(f"Tesseract recognition error: {e}")
            return [], time.time() - start_time
    
    def recognize_easyocr(self, image: np.ndarray, langs: List[str]) -> Tuple[List[str], float]:
        """
        Recognize text using EasyOCR.
        Returns (text_lines, processing_time).
        """
        reader = self.load_easyocr(langs)
        if not reader:
            return [], 0.0
        
        start_time = time.time()
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            results = reader.readtext(image_rgb, detail=0, paragraph=False)
            processing_time = time.time() - start_time
            
            text_lines = [text.strip() for text in results if text.strip()]
            return text_lines, processing_time
        except Exception as e:
            print(f"EasyOCR recognition error: {e}")
            return [], time.time() - start_time
    
    def recognize_all(self, image: np.ndarray, progress_callback=None) -> Dict[str, Any]:
        """
        Recognize text using all available engines (English only).
        Returns dictionary with results from each engine.
        """
        results = {
            'paddleocr_en': {'text': [], 'time': 0.0, 'engine': 'PaddleOCR'},
            'tesseract': {'text': [], 'time': 0.0, 'engine': 'Tesseract'},
            'easyocr_en': {'text': [], 'time': 0.0, 'engine': 'EasyOCR'},
        }
        
        engines = [
            ('PaddleOCR', lambda: self.recognize_paddleocr(image, 'en'), 'paddleocr_en'),
            ('Tesseract', lambda: self.recognize_tesseract(image), 'tesseract'),
            ('EasyOCR', lambda: self.recognize_easyocr(image, ['en']), 'easyocr_en'),
        ]
        
        total = len(engines)
        for i, (name, func, key) in enumerate(engines):
            if progress_callback:
                progress = 0.2 + (i / total) * 0.5  # 20% to 70%
                progress_callback(f"{name}...", progress)
            
            try:
                text, proc_time = func()
                results[key]['text'] = text
                results[key]['time'] = proc_time
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[key]['time'] = 0.0
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all engines."""
        return {
            'paddleocr': self.engine_status.get('paddleocr', False),
            'easyocr': self.engine_status.get('easyocr', False),
            'tesseract': self.engine_status.get('tesseract', False),
            'tesseract_version': self.engine_status.get('tesseract_version', None),
            'tesseract_path': self.tesseract_path,
        }

