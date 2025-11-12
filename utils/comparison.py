"""
OCR Comparison module - main class for comparing OCR results.
"""

from typing import Dict, List, Any, Optional
import time

from .ocr_engines import OCREngineManager
from .table_processor import TableProcessor
from .metrics import compare_results, calculate_metrics
from .text_utils import merge_texts, normalize_text


class OCRComparison:
    """
    Main class for OCR comparison system.
    Handles recognition, comparison, and analysis.
    """
    
    def __init__(self):
        """Initialize OCR comparison system."""
        self.ocr_manager = OCREngineManager()
        self.table_processor = TableProcessor(self.ocr_manager)
        self.last_results = None
        self.last_tables = None
    
    def process_image(
        self,
        image_path: str,
        recognize_text: bool = True,
        recognize_tables: bool = True,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Process image with all OCR engines.
        
        Args:
            image_path: Path to image file
            recognize_text: Whether to recognize text
            recognize_tables: Whether to recognize tables
            progress_callback: Optional callback function(status, progress)
            
        Returns:
            Dictionary with results from all engines
        """
        import cv2
        
        # Load and optimize image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Optimize image size for faster processing
        height, width = image.shape[:2]
        max_dimension = 2000  # Max width or height
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            if progress_callback:
                progress_callback(f"Изображение оптимизировано: {width}x{height} → {new_width}x{new_height}", 0.1)
        
        start_time = time.time()
        results = {}
        
        # Text recognition
        if recognize_text:
            if progress_callback:
                progress_callback("Распознавание текста...", 0.2)
            # Use all languages: English, Russian, Chinese
            ocr_results = self.ocr_manager.recognize_all(image, languages=['en', 'ru', 'ch'], progress_callback=progress_callback)
            results['text'] = ocr_results
        
        # Table recognition
        table_results = {}
        if recognize_tables:
            if progress_callback:
                progress_callback("Распознавание таблиц...", 0.7)
            table_results = self.table_processor.process_image(image_path, progress_callback=progress_callback)
            results['tables'] = table_results.get('tables', [])
            results['ppstructure_time'] = table_results.get('time', 0.0)
        
        processing_time = time.time() - start_time
        results['total_time'] = processing_time
        
        # Store for later use
        self.last_results = results
        self.last_tables = results.get('tables', [])
        
        if progress_callback:
            progress_callback("Завершено!", 1.0)
        
        return results
    
    def compare_text_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        reference_engine: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare text recognition results.
        
        Args:
            results: OCR results (uses last_results if None)
            reference_engine: Engine to use as reference
            
        Returns:
            Comparison metrics and analysis
        """
        if results is None:
            results = self.last_results
        
        if not results or 'text' not in results:
            return {}
        
        # Prepare data for comparison
        text_results = {}
        for engine_key, engine_data in results['text'].items():
            engine_name = engine_data.get('engine', engine_key)
            text_results[engine_name] = {
                'text': engine_data.get('text', []),
                'time': engine_data.get('time', 0.0),
            }
        
        # Compare
        comparison = compare_results(text_results, reference_engine)
        
        # Add additional analysis
        comparison['analysis'] = self._analyze_differences(text_results, comparison)
        
        return comparison
    
    def _analyze_differences(
        self,
        results: Dict[str, Dict[str, Any]],
        comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze differences between engines.
        
        Returns:
            Analysis dictionary with insights
        """
        analysis = {
            'best_engine': None,
            'fastest_engine': None,
            'most_detailed': None,
            'recommendations': [],
        }
        
        if not results:
            return analysis
        
        # Find fastest engine
        times = {name: data.get('time', float('inf')) for name, data in results.items()}
        if times:
            analysis['fastest_engine'] = min(times, key=times.get)
        
        # Find most detailed (most lines)
        line_counts = {name: len(data.get('text', [])) for name, data in results.items()}
        if line_counts:
            analysis['most_detailed'] = max(line_counts, key=line_counts.get)
        
        # Find best engine (highest average similarity to others)
        if 'average_similarity' in comparison:
            avg_sims = comparison['average_similarity']
            if avg_sims:
                analysis['best_engine'] = max(avg_sims, key=avg_sims.get)
        
        # Generate recommendations
        if comparison.get('similarities'):
            similarities = comparison['similarities']
            low_similarity = [eng for eng, sim in similarities.items() if sim < 50]
            if low_similarity:
                analysis['recommendations'].append(
                    f"Engines with low similarity to reference: {', '.join(low_similarity)}"
                )
        
        return analysis
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all OCR engines."""
        return self.ocr_manager.get_status()
    
    def get_merged_text(self, results: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        Get merged text from all engines (deduplicated).
        
        Returns:
            List of (engine, text) tuples
        """
        if results is None:
            results = self.last_results
        
        if not results or 'text' not in results:
            return []
        
        all_texts = []
        for engine_key, engine_data in results['text'].items():
            engine_name = engine_data.get('engine', engine_key)
            for text_line in engine_data.get('text', []):
                if text_line.strip():
                    all_texts.append((engine_name, text_line.strip()))
        
        # Merge duplicates
        merged = merge_texts(all_texts)
        return merged
    
    def export_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        comparison: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export results to various formats.
        
        Returns:
            Dictionary with format names and content
        """
        if results is None:
            results = self.last_results
        
        exports = {}
        
        # Export merged text
        merged_text = self.get_merged_text(results)
        if merged_text:
            text_content = "\n".join([text for _, text in merged_text])
            exports['text'] = text_content
        
        # Export per-engine text
        if results and 'text' in results:
            for engine_key, engine_data in results['text'].items():
                engine_name = engine_data.get('engine', engine_key)
                text_lines = engine_data.get('text', [])
                if text_lines:
                    exports[f'text_{engine_name}'] = "\n".join(text_lines)
        
        return exports

