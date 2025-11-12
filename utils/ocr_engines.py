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
        self.load_times = {}  # Track model loading times
        self.init_times = {}  # Track initialization times
        
        # Set Tesseract path if found
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available OCR engines (English, Russian, Chinese)."""
        # PaddleOCR (English, Russian, Chinese)
        paddle_start = time.time()
        try:
            from paddleocr import PaddleOCR, PPStructure
            # English
            self.engines['paddleocr_en'] = {
                'type': 'paddleocr',
                'lang': 'en',
                'instance': None,
            }
            # Russian
            self.engines['paddleocr_ru'] = {
                'type': 'paddleocr',
                'lang': 'ru',
                'instance': None,
            }
            # Chinese
            self.engines['paddleocr_ch'] = {
                'type': 'paddleocr',
                'lang': 'ch',
                'instance': None,
            }
            self.engines['ppstructure'] = {
                'type': 'ppstructure',
                'instance': None,
            }
            self.engine_status['paddleocr'] = True
            self.init_times['paddleocr'] = time.time() - paddle_start
        except Exception as e:
            self.engine_status['paddleocr'] = False
            self.init_times['paddleocr'] = time.time() - paddle_start
            print(f"PaddleOCR initialization error: {e}")
        
        # EasyOCR (English, Russian, Chinese)
        easyocr_start = time.time()
        try:
            import easyocr
            # English
            self.engines['easyocr_en'] = {
                'type': 'easyocr',
                'langs': ['en'],
                'instance': None,
            }
            # Russian
            self.engines['easyocr_ru'] = {
                'type': 'easyocr',
                'langs': ['ru'],
                'instance': None,
            }
            # Chinese
            self.engines['easyocr_ch'] = {
                'type': 'easyocr',
                'langs': ['ch_sim'],
                'instance': None,
            }
            self.engine_status['easyocr'] = True
            self.init_times['easyocr'] = time.time() - easyocr_start
        except Exception as e:
            self.engine_status['easyocr'] = False
            self.init_times['easyocr'] = time.time() - easyocr_start
            print(f"EasyOCR initialization error: {e}")
        
        # Tesseract
        tesseract_start = time.time()
        try:
            version = pytesseract.get_tesseract_version()
            self.engine_status['tesseract'] = True
            self.engine_status['tesseract_version'] = version
            self.init_times['tesseract'] = time.time() - tesseract_start
        except Exception:
            self.engine_status['tesseract'] = False
            self.init_times['tesseract'] = time.time() - tesseract_start
    
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
                start_time = time.time()
                self.engines[key]['instance'] = PaddleOCR(
                    lang=lang,
                    use_angle_cls=True,
                    use_gpu=False,
                    show_log=False
                )
                load_time = time.time() - start_time
                self.load_times[key] = load_time
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
                start_time = time.time()
                self.engines[key]['instance'] = PPStructure(
                    lang='en',
                    layout=True,
                    use_gpu=False,
                    recovery=True,
                    return_ocr_result_in_table=True,
                    show_log=False
                )
                load_time = time.time() - start_time
                self.load_times[key] = load_time
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
                start_time = time.time()
                self.engines[key]['instance'] = easyocr.Reader(
                    langs,
                    gpu=False,
                    download_enabled=True
                )
                load_time = time.time() - start_time
                self.load_times[key] = load_time
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
    
    def recognize_paddleocr_multi(self, image: np.ndarray, langs: List[str] = ['en', 'ru', 'ch']) -> Dict[str, Tuple[List[str], float]]:
        """
        Recognize text using PaddleOCR with multiple languages.
        Returns dictionary {lang: (text_lines, processing_time)}.
        """
        results = {}
        for lang in langs:
            key = f'paddleocr_{lang}'
            if key in self.engines:
                text, proc_time = self.recognize_paddleocr(image, lang)
                results[key] = (text, proc_time)
        return results
    
    def recognize_easyocr_multi(self, image: np.ndarray, lang_configs: List[List[str]] = None) -> Dict[str, Tuple[List[str], float]]:
        """
        Recognize text using EasyOCR with multiple languages.
        Returns dictionary {lang_key: (text_lines, processing_time)}.
        """
        if lang_configs is None:
            lang_configs = [['en'], ['ru'], ['ch_sim']]
        
        results = {}
        for langs in lang_configs:
            # Find matching key
            key = None
            for k, v in self.engines.items():
                if v.get('type') == 'easyocr' and v.get('langs') == langs:
                    key = k
                    break
            
            if key:
                text, proc_time = self.recognize_easyocr(image, langs)
                results[key] = (text, proc_time)
        
        return results
    
    def recognize_tesseract_multi(self, image: np.ndarray, langs: List[str] = ['eng', 'rus', 'chi_sim']) -> Dict[str, Tuple[List[str], float]]:
        """
        Recognize text using Tesseract with multiple languages.
        Returns dictionary {lang: (text_lines, processing_time)}.
        """
        results = {}
        for lang in langs:
            try:
                text, proc_time = self.recognize_tesseract(image, lang)
                results[f'tesseract_{lang}'] = (text, proc_time)
            except Exception as e:
                print(f"Tesseract {lang} recognition error: {e}")
                results[f'tesseract_{lang}'] = ([], 0.0)
        return results
    
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
            # ---- Preprocess for better EasyOCR accuracy ----
            # 1) Ensure RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img = image
            
            # 2) Upscale small images to help the detector
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim < 1200:
                scale = 1200.0 / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 3) Light denoise and contrast normalization
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_dn = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)
            gray = cv2.cvtColor(img_dn, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # Convert back to RGB for EasyOCR
            image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # Use paragraph mode to merge small segments into lines
            results = reader.readtext(
                image_rgb,
                detail=0,
                paragraph=True,
                width_ths=0.6,
                ycenter_ths=0.5,
                height_ths=0.6,
                mag_ratio=1.5,
            )
            processing_time = time.time() - start_time
            
            # Postprocess: split paragraphs into clean lines
            text_lines = []
            for text in results:
                if not text:
                    continue
                # Normalize whitespace and split on newline if any
                for line in str(text).splitlines():
                    line = line.strip()
                    if line:
                        text_lines.append(line)
            return text_lines, processing_time
        except Exception as e:
            print(f"EasyOCR recognition error: {e}")
            return [], time.time() - start_time
    
    def recognize_all(self, image: np.ndarray, languages: List[str] = ['en'], progress_callback=None) -> Dict[str, Any]:
        """
        Recognize text using all available engines with specified languages.
        Returns dictionary with results from each engine.
        
        Args:
            image: Image to process
            languages: List of language codes ['en', 'ru', 'ch']
            progress_callback: Optional progress callback
        """
        results = {}
        
        # Try all three engines with all specified languages
        engines_to_try = []
        
        # PaddleOCR for each language
        for lang in languages:
            key = f'paddleocr_{lang}'
            if key in self.engines:
                engines_to_try.append((
                    f'PaddleOCR ({lang})',
                    lambda img=image, l=lang: self.recognize_paddleocr(img, l),
                    key
                ))
        
        # Tesseract - try to detect language automatically or use first language
        if self.engine_status.get('tesseract', False):
            # Map language codes to Tesseract codes
            tesseract_lang_map = {'en': 'eng', 'ru': 'rus', 'ch': 'chi_sim'}
            tesseract_langs = [tesseract_lang_map.get(lang, 'eng') for lang in languages if lang in tesseract_lang_map]
            if tesseract_langs:
                # Combine languages for Tesseract
                tesseract_lang_str = '+'.join(tesseract_langs)
                engines_to_try.append((
                    'Tesseract',
                    lambda img=image, l=tesseract_lang_str: self.recognize_tesseract(img, l),
                    'tesseract'
                ))
        
        # EasyOCR for each language
        easyocr_lang_map = {'en': ['en'], 'ru': ['ru'], 'ch': ['ch_sim']}
        for lang in languages:
            if lang in easyocr_lang_map:
                key = f'easyocr_{lang}'
                if key in self.engines:
                    langs = easyocr_lang_map[lang]
                    engines_to_try.append((
                        f'EasyOCR ({lang})',
                        lambda img=image, l=langs: self.recognize_easyocr(img, l),
                        key
                    ))
        
        # Initialize results structure
        for _, _, key in engines_to_try:
            engine_name = key.split('_')[0].capitalize()
            if 'paddleocr' in key:
                engine_name = 'PaddleOCR'
            elif 'easyocr' in key:
                engine_name = 'EasyOCR'
            elif 'tesseract' in key:
                engine_name = 'Tesseract'
            
            results[key] = {'text': [], 'time': 0.0, 'engine': engine_name}
        
        # Process each engine
        total = len(engines_to_try)
        for i, (name, func, key) in enumerate(engines_to_try):
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
    
    def get_load_times(self) -> Dict[str, float]:
        """Get model loading times."""
        return self.load_times.copy()
    
    def get_init_times(self) -> Dict[str, float]:
        """Get model initialization times."""
        return self.init_times.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all engines."""
        return {
            'paddleocr': self.engine_status.get('paddleocr', False),
            'easyocr': self.engine_status.get('easyocr', False),
            'tesseract': self.engine_status.get('tesseract', False),
            'tesseract_version': self.engine_status.get('tesseract_version', None),
            'tesseract_path': self.tesseract_path,
        }

