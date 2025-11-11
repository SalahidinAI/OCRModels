"""
Table processing module using PPStructure.
Handles table detection and cell-level OCR recognition.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pytesseract

from .config import Config


class TableProcessor:
    """Processes tables using PPStructure with hybrid OCR for cells."""
    
    def __init__(self, ocr_manager=None):
        """
        Initialize table processor.
        
        Args:
            ocr_manager: OCREngineManager instance for cell OCR
        """
        self.ocr_manager = ocr_manager
        self.ppstructure = None
        self.tesseract_path = Config.get_tesseract_path()
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
    
    def load_ppstructure(self):
        """Load PPStructure model (lazy loading)."""
        if self.ppstructure is not None:
            return self.ppstructure
        
        try:
            from paddleocr import PPStructure
            self.ppstructure = PPStructure(
                lang='en',
                layout=True,
                use_gpu=False,
                recovery=True,
                return_ocr_result_in_table=True,
                show_log=False
            )
            return self.ppstructure
        except Exception as e:
            print(f"Error loading PPStructure: {e}")
            return None
    
    def process_image(self, image_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process image to extract tables.
        
        Args:
            image_path: Path to image file
            progress_callback: Optional callback function(status, progress)
            
        Returns:
            Dictionary with tables list and processing time
        """
        import time
        table_engine = self.load_ppstructure()
        if not table_engine:
            return {'tables': [], 'time': 0.0}
        
        try:
            if progress_callback:
                progress_callback("PPStructure: детекция таблиц...", 0.7)
            
            # PPStructure processes the image
            start_time = time.time()
            table_result = table_engine(image_path)
            ppstructure_time = time.time() - start_time
            
            tables = []
            regions = [r for r in table_result if r.get("type") == "table"]
            total = len(regions)
            
            for i, region in enumerate(regions):
                if progress_callback and total > 0:
                    progress = 0.7 + (i / total) * 0.25
                    progress_callback(f"Обработка таблицы {i+1}/{total}...", progress)
                
                table_data = self._process_table_region(region, image_path, progress_callback)
                if table_data:
                    tables.append(table_data)
            
            return {'tables': tables, 'time': ppstructure_time}
        except Exception as e:
            print(f"Error processing tables: {e}")
            return {'tables': [], 'time': 0.0}
    
    def _process_table_region(self, region: Dict, image_path: str, progress_callback=None) -> Optional[Dict[str, Any]]:
        """
        Process a single table region.
        
        Args:
            region: Table region from PPStructure
            image_path: Path to original image
            
        Returns:
            Dictionary with table structure and recognized text
        """
        res = region.get("res", {})
        cells = res.get("cells", [])
        
        # If PPStructure only provided HTML, use it
        if not cells:
            html = res.get("html", "")
            if html:
                return {
                    'html': html,
                    'rows': None,
                    'has_cell_ocr': False,
                }
            return None
        
        # Load image for cell-level OCR
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return None
        
        # Process cells with hybrid OCR
        rows = {}
        for cell in cells:
            r = cell.get("row_start", 0)
            c = cell.get("col_start", 0)
            bbox = cell.get("bbox", [])
            
            text = ""
            if len(bbox) == 4:
                text = self._recognize_cell(img_cv, bbox)
            
            if r not in rows:
                rows[r] = {}
            rows[r][c] = text
        
        # Generate HTML table
        html = self._generate_html_table(rows)
        
        return {
            'html': html,
            'rows': rows,
            'has_cell_ocr': True,
            'num_rows': len(rows),
            'num_cols': max([max(rows[r].keys()) for r in rows] + [0]) + 1,
        }
    
    def _recognize_cell(self, image: np.ndarray, bbox: List[int]) -> str:
        """
        Recognize text in a single cell using hybrid OCR approach.
        
        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Recognized text
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return ""
        
        # Preprocess ROI
        roi = self._preprocess_roi(roi)
        
        # Try OCR engines in order of preference
        text = ""
        
        # 1. Tesseract (usually best for structured text)
        if self.tesseract_path:
            try:
                text = pytesseract.image_to_string(
                    roi,
                    lang="eng",
                    config="--psm 6 --oem 1"
                ).strip()
                if len(text) > 1:
                    return text
            except Exception:
                pass
        
        # 2. EasyOCR EN (if available)
        if self.ocr_manager and not text:
            try:
                easy_en = self.ocr_manager.load_easyocr(['en'])
                if easy_en:
                    results = easy_en.readtext(roi, detail=0)
                    if results:
                        text = " ".join([t.strip() for t in results if t.strip()])
                        if text:
                            return text
            except Exception:
                pass
        
        return text
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI for better OCR recognition.
        
        Args:
            roi: Region of interest image
            
        Returns:
            Preprocessed image
        """
        # Resize for better recognition (2x)
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Binarization using Otsu
        _, roi_bin = cv2.threshold(
            roi_gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return roi_bin
    
    def _generate_html_table(self, rows: Dict[int, Dict[int, str]]) -> str:
        """
        Generate HTML table from rows dictionary.
        
        Args:
            rows: Dictionary {row_index: {col_index: text}}
            
        Returns:
            HTML string
        """
        html_lines = ["<table border='1' style='border-collapse:collapse;'>"]
        
        for rr in sorted(rows.keys()):
            html_lines.append("<tr>")
            for cc in sorted(rows[rr].keys()):
                cell_text = rows[rr][cc]
                # Escape HTML special characters
                cell_text = (
                    cell_text.replace("&", "&amp;")
                             .replace("<", "&lt;")
                             .replace(">", "&gt;")
                             .replace('"', "&quot;")
                )
                html_lines.append(f"<td style='padding:6px;'>{cell_text}</td>")
            html_lines.append("</tr>")
        
        html_lines.append("</table>")
        return "".join(html_lines)

