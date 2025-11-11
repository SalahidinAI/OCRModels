"""
Report generator for OCR comparison results.
Generates human-readable reports and summaries.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class ReportGenerator:
    """Generate reports from OCR comparison results."""
    
    @staticmethod
    def generate_text_report(
        comparison: Dict[str, Any],
        results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text report from comparison results.
        
        Args:
            comparison: Comparison metrics dictionary
            results: Original OCR results
            
        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("OCR COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        # Engine status
        if results and 'text' in results:
            lines.append("ENGINES USED:")
            for engine_key, engine_data in results['text'].items():
                engine_name = engine_data.get('engine', engine_key)
                proc_time = engine_data.get('time', 0.0)
                line_count = len(engine_data.get('text', []))
                lines.append(f"  - {engine_name}: {line_count} lines, {proc_time:.2f}s")
            lines.append("")
        
        # Similarity metrics
        if 'pairwise_similarities' in comparison:
            lines.append("PAIRWISE SIMILARITY METRICS:")
            for pair, similarity in comparison['pairwise_similarities'].items():
                lines.append(f"  {pair}: {similarity:.1f}%")
            lines.append("")
        
        # Error rates
        if 'error_rates' in comparison:
            lines.append("ERROR RATES (vs reference):")
            ref_engine = comparison.get('reference_engine', 'Unknown')
            lines.append(f"  Reference engine: {ref_engine}")
            for engine, rates in comparison['error_rates'].items():
                if engine != ref_engine:
                    cer = rates.get('cer', 0.0) * 100
                    wer = rates.get('wer', 0.0) * 100
                    lines.append(f"  {engine}: CER={cer:.2f}%, WER={wer:.2f}%")
            lines.append("")
        
        # Analysis
        if 'analysis' in comparison:
            analysis = comparison['analysis']
            lines.append("ANALYSIS:")
            if analysis.get('best_engine'):
                lines.append(f"  Best engine (highest consensus): {analysis['best_engine']}")
            if analysis.get('fastest_engine'):
                lines.append(f"  Fastest engine: {analysis['fastest_engine']}")
            if analysis.get('most_detailed'):
                lines.append(f"  Most detailed: {analysis['most_detailed']}")
            if analysis.get('recommendations'):
                lines.append("  Recommendations:")
                for rec in analysis['recommendations']:
                    lines.append(f"    - {rec}")
            lines.append("")
        
        # Processing time
        if 'processing_times' in comparison:
            lines.append("PROCESSING TIMES:")
            for engine, proc_time in comparison['processing_times'].items():
                lines.append(f"  {engine}: {proc_time:.2f}s")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_json_report(
        comparison: Dict[str, Any],
        results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate JSON report from comparison results.
        
        Args:
            comparison: Comparison metrics dictionary
            results: Original OCR results
            
        Returns:
            JSON string
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'comparison': comparison,
        }
        
        if results:
            # Include summary of results (not full text)
            if 'text' in results:
                report['results_summary'] = {
                    engine_key: {
                        'engine': engine_data.get('engine', engine_key),
                        'line_count': len(engine_data.get('text', [])),
                        'processing_time': engine_data.get('time', 0.0),
                    }
                    for engine_key, engine_data in results['text'].items()
                }
        
        return json.dumps(report, indent=2, ensure_ascii=False)
    
    @staticmethod
    def generate_summary_stats(comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for display.
        
        Returns:
            Dictionary with key statistics
        """
        stats = {
            'total_engines': len(comparison.get('engines', [])),
            'reference_engine': comparison.get('reference_engine', 'Unknown'),
            'average_similarity': {},
            'processing_times': comparison.get('processing_times', {}),
        }
        
        # Calculate average similarity
        if 'pairwise_similarities' in comparison:
            similarities = list(comparison['pairwise_similarities'].values())
            if similarities:
                stats['overall_avg_similarity'] = sum(similarities) / len(similarities)
        
        return stats

