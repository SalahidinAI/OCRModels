"""
Test script for OCR engines.
Tests all available OCR engines on a test image.
"""

import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import OCRComparison, Config, ReportGenerator


def test_ocr_engines(image_path: str = "test.jpeg"):
    """
    Test all OCR engines on a test image.
    
    Args:
        image_path: Path to test image
    """
    print("=" * 80)
    print("OCR ENGINES TEST")
    print("=" * 80)
    print()
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file '{image_path}' not found!")
        print("Please provide a valid image path.")
        return
    
    # Initialize OCR system
    print("⏳ Initializing OCR system...")
    ocr_system = OCRComparison()
    
    # Check engine status
    print("\n📊 Engine Status:")
    status = ocr_system.get_engine_status()
    
    print(f"  PaddleOCR: {'✅' if status.get('paddleocr') else '❌'}")
    print(f"  EasyOCR: {'✅' if status.get('easyocr') else '❌'}")
    print(f"  Tesseract: {'✅' if status.get('tesseract') else '❌'}")
    
    if status.get('tesseract_version'):
        print(f"    Version: {status['tesseract_version']}")
    if status.get('tesseract_path'):
        print(f"    Path: {status['tesseract_path']}")
    
    print()
    
    # Process image
    print(f"🔍 Processing image: {image_path}")
    print("-" * 80)
    
    try:
        start_time = time.time()
        results = ocr_system.process_image(
            image_path,
            recognize_text=True,
            recognize_tables=True
        )
        total_time = time.time() - start_time
        
        print(f"✅ Processing completed in {total_time:.2f} seconds")
        print()
        
        # Text results
        if 'text' in results:
            print("📄 TEXT RECOGNITION RESULTS:")
            print("-" * 80)
            
            for engine_key, engine_data in results['text'].items():
                engine_name = engine_data.get('engine', engine_key)
                text_lines = engine_data.get('text', [])
                proc_time = engine_data.get('time', 0.0)
                
                print(f"\n{engine_name}:")
                print(f"  Lines: {len(text_lines)}")
                print(f"  Time: {proc_time:.2f}s")
                if text_lines:
                    print(f"  Sample (first 3 lines):")
                    for line in text_lines[:3]:
                        print(f"    - {line[:80]}...")
            print()
        
        # Comparison
        print("📊 COMPARISON METRICS:")
        print("-" * 80)
        
        comparison = ocr_system.compare_text_results(results)
        
        if comparison:
            if 'pairwise_similarities' in comparison:
                print("\nPairwise Similarities:")
                for pair, similarity in comparison['pairwise_similarities'].items():
                    print(f"  {pair}: {similarity:.1f}%")
            
            if 'processing_times' in comparison:
                print("\nProcessing Times:")
                for engine, proc_time in comparison['processing_times'].items():
                    print(f"  {engine}: {proc_time:.2f}s")
            
            if 'analysis' in comparison:
                analysis = comparison['analysis']
                print("\nAnalysis:")
                if analysis.get('best_engine'):
                    print(f"  Best engine: {analysis['best_engine']}")
                if analysis.get('fastest_engine'):
                    print(f"  Fastest engine: {analysis['fastest_engine']}")
                if analysis.get('most_detailed'):
                    print(f"  Most detailed: {analysis['most_detailed']}")
        
        print()
        
        # Tables
        if 'tables' in results and results['tables']:
            print("📊 TABLE RECOGNITION:")
            print("-" * 80)
            print(f"Found {len(results['tables'])} table(s)")
            for i, table in enumerate(results['tables'], 1):
                num_rows = table.get('num_rows', 'Unknown')
                num_cols = table.get('num_cols', 'Unknown')
                print(f"  Table {i}: {num_rows} rows × {num_cols} columns")
            print()
        
        # Generate report
        print("📋 GENERATING REPORT...")
        report = ReportGenerator.generate_text_report(comparison, results)
        
        # Save report
        report_path = "ocr_test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_path}")
        print()
        
        # Export results
        print("💾 EXPORTING RESULTS...")
        exports = ocr_system.export_results(results)
        
        for export_name, content in exports.items():
            if content:
                export_path = f"ocr_test_{export_name}.txt"
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ {export_name} saved to: {export_path}")
        
        print()
        print("=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Get image path from command line or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpeg"
    test_ocr_engines(image_path)
