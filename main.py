"""
OCR Comparison System - Main Streamlit Application
Система сравнения OCR-моделей с метриками и анализом результатов.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from io import BytesIO
import pandas as pd
import time

# Import utils
from utils import OCRComparison, ReportGenerator

# ========= Page Configuration =========
st.set_page_config(
    page_title="OCR Сравнение с подсветкой различий",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========= Title and Description =========
st.title("📸 OCR Сравнение с подсветкой различий")
st.caption("Посимвольное сравнение результатов разных движков")

# ========= Initialize OCR System =========
@st.cache_resource
def init_ocr_system():
    """Initialize OCR comparison system (cached)."""
    return OCRComparison()

with st.spinner("⏳ Инициализация системы OCR..."):
    ocr_system = init_ocr_system()

# ========= Engine Status =========
st.sidebar.header("🔧 Статус OCR-движков")

engine_status = ocr_system.get_engine_status()

if engine_status.get('paddleocr'):
    st.sidebar.success("✅ PaddleOCR подключён")
else:
    st.sidebar.error("❌ PaddleOCR недоступен")

if engine_status.get('easyocr'):
    st.sidebar.success("✅ EasyOCR подключён")
else:
    st.sidebar.warning("⚠️ EasyOCR недоступен")

if engine_status.get('tesseract'):
    version = engine_status.get('tesseract_version', 'Unknown')
    st.sidebar.success(f"✅ Tesseract: {version}")
else:
    st.sidebar.error("❌ Tesseract не найден")
    if engine_status.get('tesseract_path'):
        st.sidebar.info(f"Путь: {engine_status['tesseract_path']}")
    else:
        st.sidebar.warning("⚠️ Установите Tesseract: brew install tesseract")

# ========= Image Upload =========
st.header("📁 Загрузите изображение")
uploaded_file = st.file_uploader(
    "Выберите файл изображения",
    type=["png", "jpg", "jpeg"],
    help="Поддерживаемые форматы: PNG, JPG, JPEG. Максимальный размер: 200 МБ"
)

# ========= Processing Options =========
if uploaded_file:
    st.success(f"✅ Загружено: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
    
    # Preview uploaded image
    try:
        st.image(uploaded_file, caption=f"Предпросмотр: {uploaded_file.name}", use_column_width=True)
    except Exception:
        pass
    
    col1, col2 = st.columns(2)
    with col1:
        recognize_text = st.checkbox("Распознать текст", value=True)
    with col2:
        recognize_tables = st.checkbox(
            "Распознать таблицы", 
            value=True,
            help="⚠️ Обработка таблиц может занять больше времени (1-5 минут)"
        )
    
    if recognize_tables:
        st.info("ℹ️ Большие изображения автоматически оптимизируются для ускорения обработки")
    
    if st.button("🔍 Распознать и сравнить", type="primary", use_container_width=True):
        # Save uploaded file temporarily
        file_bytes = uploaded_file.read()
        suffix = Path(uploaded_file.name).suffix
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            img_path = tmp.name

        try:
            # Process image with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(status, progress):
                progress_bar.progress(progress)
                status_text.text(f"⏳ {status}")
            
            start_time = time.time()
            try:
                results = ocr_system.process_image(
                    img_path,
                    recognize_text=recognize_text,
                    recognize_tables=recognize_tables,
                    progress_callback=update_progress
                )
                processing_time = time.time() - start_time
            finally:
                progress_bar.empty()
                status_text.empty()
            
            # Show processing time
            st.success(f"✅ Обработка завершена за {processing_time:.2f} сек")
            
            # ========= Show Model Timings =========
            st.subheader("⏱️ Время работы моделей")
            
            timing_info = []
            
            # Get timing for text recognition models
            if recognize_text and 'text' in results:
                text_results = results['text']
                paddle_time = text_results.get('paddleocr_en', {}).get('time', 0.0)
                tesseract_time = text_results.get('tesseract', {}).get('time', 0.0)
                easyocr_time = text_results.get('easyocr_en', {}).get('time', 0.0)
                
                if paddle_time > 0:
                    timing_info.append(f"**PaddleOCR**: {paddle_time:.1f} сек.")
                if tesseract_time > 0:
                    timing_info.append(f"**Tesseract**: {tesseract_time:.1f} сек.")
                if easyocr_time > 0:
                    timing_info.append(f"**EasyOCR**: {easyocr_time:.1f} сек.")
            
            # Get timing for PPStructure
            if recognize_tables and 'ppstructure_time' in results:
                ppstructure_time = results['ppstructure_time']
                if ppstructure_time > 0:
                    timing_info.append(f"**PPStructure**: {ppstructure_time:.1f} сек.")
            
            if timing_info:
                st.markdown("\n".join(timing_info))
            
            # ========= Text Recognition Results =========
            if recognize_text and 'text' in results:
                st.divider()
                st.header("🔤 Сравнение результатов OCR")
                
                # Compare results
                comparison = ocr_system.compare_text_results(results)
                
                if comparison:
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Движков использовано", len(comparison.get('engines', [])))
                    
                    with col2:
                        if 'pairwise_similarities' in comparison:
                            similarities = list(comparison['pairwise_similarities'].values())
                            if similarities:
                                avg_sim = sum(similarities) / len(similarities)
                                st.metric("Средняя схожесть", f"{avg_sim:.1f}%")
                    
                    with col3:
                        ref_engine = comparison.get('reference_engine', 'Unknown')
                        st.metric("Эталонный движок", ref_engine)
                    
                    # Pairwise similarities
                    st.subheader("📊 Статистика совпадений")
                    if 'pairwise_similarities' in comparison:
                        similarity_data = []
                        for pair, similarity in comparison['pairwise_similarities'].items():
                            engines = pair.split(' ↔ ')
                            similarity_data.append({
                                'Движок 1': engines[0],
                                'Движок 2': engines[1],
                                'Схожесть (%)': f"{similarity:.1f}"
                            })
                        
                        if similarity_data:
                            df_sim = pd.DataFrame(similarity_data)
                            st.dataframe(df_sim, use_container_width=True, hide_index=True)
                    
                    
                    # Error rates
                    if 'error_rates' in comparison:
                        st.subheader("📈 Метрики ошибок")
                        error_data = []
                        for engine, rates in comparison['error_rates'].items():
                            if engine != comparison.get('reference_engine'):
                                error_data.append({
                                    'Движок': engine,
                                    'CER (%)': f"{rates.get('cer', 0) * 100:.2f}",
                                    'WER (%)': f"{rates.get('wer', 0) * 100:.2f}"
                                })
                        
                        if error_data:
                            df_errors = pd.DataFrame(error_data)
                            st.dataframe(df_errors, use_container_width=True, hide_index=True)
                    
                    # Analysis
                    if 'analysis' in comparison:
                        analysis = comparison['analysis']
                        st.subheader("🔍 Анализ результатов")
                        
                        if analysis.get('best_engine'):
                            st.info(f"🏆 Лучший движок (высокий консенсус): **{analysis['best_engine']}**")
                        
                        if analysis.get('fastest_engine'):
                            st.info(f"⚡ Самый быстрый: **{analysis['fastest_engine']}**")
                        
                        if analysis.get('most_detailed'):
                            st.info(f"📝 Наиболее детальный: **{analysis['most_detailed']}**")
                    
                    # Text results by engine
                    st.subheader("📄 Результаты по движкам")
                    merged_text = ocr_system.get_merged_text(results)
                    
                    if merged_text:
                        # Group by engine
                        engine_texts = {}
                        for engine, text in merged_text:
                            if engine not in engine_texts:
                                engine_texts[engine] = []
                            engine_texts[engine].append(text)
                        
                        # Display in tabs
                        tabs = st.tabs(list(engine_texts.keys()))
                        for tab, (engine, texts) in zip(tabs, engine_texts.items()):
                            with tab:
                                full_text = "\n".join(texts)
                                st.text_area(
                                    f"Текст от {engine}",
                                    full_text,
                                    height=300,
                                    key=f"text_{engine}"
                                )
                                
                                # Download button
                                st.download_button(
                                    f"📥 Скачать текст ({engine})",
                                    full_text,
                                    f"ocr_{engine.lower()}.txt",
                                    key=f"download_{engine}"
                                )
                    
                    # Download merged text
                    if merged_text:
                        all_text = "\n".join([text for _, text in merged_text])
                        st.download_button(
                            "📥 Скачать объединённый текст",
                            all_text,
                            "ocr_merged.txt",
                            use_container_width=True
                        )
                    
                    # Generate and download report
                    st.subheader("📋 Отчёт")
                    report_text = ReportGenerator.generate_text_report(comparison, results)
                    st.download_button(
                        "📥 Скачать текстовый отчёт",
                        report_text,
                        "ocr_report.txt",
                        use_container_width=True
                    )
                    
                    # JSON report
                    report_json = ReportGenerator.generate_json_report(comparison, results)
                    st.download_button(
                        "📥 Скачать JSON-отчёт",
                        report_json,
                        "ocr_report.json",
                        use_container_width=True
                    )
            
            # ========= Table Recognition Results =========
            if recognize_tables and 'tables' in results and results['tables']:
                st.divider()
                st.header("📊 Распознанные таблицы")
                
                tables = results['tables']
                for i, table_data in enumerate(tables, 1):
                    st.subheader(f"Таблица {i}")
                    
                    html = table_data.get('html', '')
                    if html:
                        try:
                            # Parse HTML to DataFrame
                            df = pd.read_html(html)[0]
                            
                            # Display table
                            st.dataframe(df, use_container_width=True)
                            
                            # Table info
                            num_rows = table_data.get('num_rows', len(df))
                            num_cols = table_data.get('num_cols', len(df.columns))
                            st.caption(f"Размер: {num_rows} строк × {num_cols} столбцов")
                            
                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                csv_data = df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    f"📥 CSV таблица {i}",
                                    csv_data,
                                    f"table_{i}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                buffer = BytesIO()
                                df.to_excel(buffer, index=False, engine='openpyxl')
                                buffer.seek(0)
                                st.download_button(
                                    f"📥 Excel таблица {i}",
                                    buffer.getvalue(),
                                    f"table_{i}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.warning(f"Ошибка парсинга таблицы {i}: {e}")
                            st.code(html[:1000])
            elif recognize_tables:
                st.info("Таблицы не найдены на изображении.")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(img_path)
            except:
                pass

# ========= Footer =========
st.divider()
st.caption("OCR Comparison System v1.0 | Поддержка: PaddleOCR, Tesseract, EasyOCR")
