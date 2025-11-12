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

# ========= Initialize OCR System =========
@st.cache_resource
def init_ocr_system():
    """Initialize OCR comparison system (cached)."""
    return OCRComparison()

init_status = st.empty()
init_status.info("⏳ Инициализация системы OCR...")

init_start = time.time()
ocr_system = init_ocr_system()
init_time = time.time() - init_start

if 'load_times' not in st.session_state:
    st.session_state.load_times = ocr_system.ocr_manager.get_load_times()

init_times = ocr_system.ocr_manager.get_init_times()
init_status.empty()

total_load_time = sum(st.session_state.load_times.values()) if st.session_state.load_times else 0.0

# Display initialization times
if init_times:
    paddle_init = init_times.get('paddleocr', 0.0)
    easyocr_init = init_times.get('easyocr', 0.0)
    tesseract_init = init_times.get('tesseract', 0.0)
    total_init = paddle_init + easyocr_init + tesseract_init
    
    # Compact display with total and individual times
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    
    with col1:
        st.metric("⏱️ Инициализация моделей", f"{total_init:.2f} сек")
    
    with col2:
        if paddle_init > 0:
            st.caption(f"**PaddleOCR:** {paddle_init:.2f} сек")
        else:
            st.caption("**PaddleOCR:** —")
    
    with col3:
        if tesseract_init > 0:
            st.caption(f"**Tesseract:** {tesseract_init:.2f} сек")
        else:
            st.caption("**Tesseract:** —")
    
    with col4:
        if easyocr_init > 0:
            st.caption(f"**EasyOCR:** {easyocr_init:.2f} сек")
        else:
            st.caption("**EasyOCR:** —")

if total_load_time > 0:
    st.caption(f"📦 Общее время загрузки моделей: {total_load_time:.2f} сек")
elif st.session_state.load_times:
    st.caption("ℹ️ Модели загружаются при первом использовании")

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
        st.image(uploaded_file, caption=f"Предпросмотр: {uploaded_file.name}", use_container_width=True)
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
            
            st.success(f"✅ Обработка завершена за {processing_time:.2f} сек")
            
            # ========= Show Model Timings =========
            st.subheader("⏱️ Время работы моделей")
            
            # Get timing for text recognition models
            paddle_time = 0.0
            tesseract_time = 0.0
            easyocr_time = 0.0
            ppstructure_time = 0.0
            
            if recognize_text and 'text' in results:
                text_results = results['text']
                
                paddle_times = []
                easyocr_times = []
                
                for key, data in text_results.items():
                    proc_time = data.get('time', 0.0)
                    if proc_time > 0:
                        if 'paddleocr' in key:
                            paddle_times.append(proc_time)
                        elif 'easyocr' in key:
                            easyocr_times.append(proc_time)
                        elif 'tesseract' in key:
                            tesseract_time = max(tesseract_time, proc_time)
                
                if paddle_times:
                    paddle_time = sum(paddle_times)
                if easyocr_times:
                    easyocr_time = sum(easyocr_times)
                
                cols = st.columns(3)
                
                if paddle_time > 0:
                    with cols[0]:
                        st.metric("PaddleOCR", f"{paddle_time:.2f} сек", 
                                 delta=f"{len(paddle_times)} языков" if len(paddle_times) > 1 else None)
                
                if tesseract_time > 0:
                    with cols[1]:
                        st.metric("Tesseract", f"{tesseract_time:.2f} сек")
                
                if easyocr_time > 0:
                    with cols[2]:
                        st.metric("EasyOCR", f"{easyocr_time:.2f} сек",
                                 delta=f"{len(easyocr_times)} языков" if len(easyocr_times) > 1 else None)
            
            if recognize_tables and 'ppstructure_time' in results:
                ppstructure_time = results['ppstructure_time']
                if ppstructure_time > 0:
                    st.divider()
                    st.metric("PPStructure (таблицы)", f"{ppstructure_time:.2f} сек")
            
            total_exec_time = paddle_time + tesseract_time + easyocr_time + ppstructure_time
            if total_exec_time > 0:
                st.divider()
                st.metric("⏱️ Общее время выполнения", f"{total_exec_time:.2f} сек")
            
            # ========= Text Recognition Results =========
            if recognize_text and 'text' in results:
                st.divider()
                st.header("🔤 Сравнение результатов OCR")
                
                # Get text results from each main engine (PaddleOCR, Tesseract, EasyOCR)
                text_results = results['text']
                
                # Find best result for each engine type (one per engine)
                paddle_text = ""
                tesseract_text = ""
                easyocr_text = ""
                
                # Collect all texts by engine type
                paddle_texts = []
                easyocr_texts = []
                
                for key, data in text_results.items():
                    text_lines = data.get('text', [])
                    full_text = "\n".join(text_lines)
                    
                    if 'paddleocr' in key and full_text:
                        paddle_texts.append((full_text, key))
                    elif 'tesseract' in key and full_text:
                        if not tesseract_text or len(full_text) > len(tesseract_text):
                            tesseract_text = full_text
                    elif 'easyocr' in key and full_text:
                        easyocr_texts.append((full_text, key))
                
                # Use longest text for each engine (best result)
                if paddle_texts:
                    paddle_text = max(paddle_texts, key=lambda x: len(x[0]))[0]
                if easyocr_texts:
                    easyocr_text = max(easyocr_texts, key=lambda x: len(x[0]))[0]
                
                engine_texts = []
                engine_names = []
                
                # Order: PaddleOCR, Tesseract, EasyOCR
                if paddle_text:
                    engine_texts.append(paddle_text)
                    engine_names.append("PaddleOCR")
                if tesseract_text:
                    engine_texts.append(tesseract_text)
                    engine_names.append("Tesseract")
                if easyocr_text:
                    engine_texts.append(easyocr_text)
                    engine_names.append("EasyOCR")
                
                if len(engine_texts) >= 2:
                    from utils.text_utils import compare_texts_word_by_word
                    word_comparisons = compare_texts_word_by_word(engine_texts)
                    
                    # Calculate match statistics across all engines
                    all_words = []
                    for engine_words in word_comparisons:
                        all_words.extend(engine_words)
                    
                    if all_words:
                        total_words = len(all_words)
                        all_match = sum(1 for w in all_words if w['match_count'] == w['total_engines'])
                        two_match = sum(1 for w in all_words if w['match_count'] >= 2 and w['match_count'] < w['total_engines'])
                        one_match = sum(1 for w in all_words if w['match_count'] == 1)
                        
                        match_percentage = (all_match / total_words * 100) if total_words > 0 else 0
                        partial_match_percentage = (two_match / total_words * 100) if total_words > 0 else 0
                        mismatch_percentage = (one_match / total_words * 100) if total_words > 0 else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Всего слов", total_words)
                        with col2:
                            st.metric("Полное совпадение", f"{match_percentage:.1f}%", 
                                     delta=f"{all_match} слов", delta_color="normal")
                        with col3:
                            st.metric("Частичное совпадение", f"{partial_match_percentage:.1f}%",
                                     delta=f"{two_match} слов", delta_color="off")
                        with col4:
                            st.metric("Расхождения", f"{mismatch_percentage:.1f}%",
                                     delta=f"{one_match} слов", delta_color="inverse")
                        
                        try:
                            import plotly.graph_objects as go
                            fig = go.Figure(data=[
                                go.Bar(name='Полное совпадение', x=['Совпадения'], y=[match_percentage], marker_color='green'),
                                go.Bar(name='Частичное совпадение', x=['Совпадения'], y=[partial_match_percentage], marker_color='yellow'),
                                go.Bar(name='Расхождения', x=['Совпадения'], y=[mismatch_percentage], marker_color='red')
                            ])
                            fig.update_layout(
                                barmode='stack',
                                title='Распределение совпадений по словам',
                                yaxis_title='Процент (%)',
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            pass
                    
                    st.subheader("📄 Результаты по движкам")
                    
                    # Display vertically: PaddleOCR, Tesseract, EasyOCR
                    for idx, (engine_name, engine_text, word_comparison) in enumerate(zip(engine_names, engine_texts, word_comparisons)):
                        st.markdown(f"### {engine_name}")
                        
                        # Create colored HTML
                        html_parts = []
                        for word_data in word_comparison:
                            word = word_data['word']
                            color = word_data['color']
                            
                            if color == "green":
                                bg_color = "#90EE90"
                            elif color == "yellow":
                                bg_color = "#FFE4B5"
                            else:
                                bg_color = "#FFB6C1"
                            
                            word_escaped = word.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            html_parts.append(f'<span style="background-color: {bg_color}; padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block;">{word_escaped}</span>')
                        
                        st.markdown(f'<div style="line-height: 1.8; word-wrap: break-word;">{" ".join(html_parts)}</div>', unsafe_allow_html=True)
                        
                        st.download_button(
                            f"📥 Скачать ({engine_name})",
                            engine_text,
                            f"ocr_{engine_name.lower()}.txt",
                            key=f"download_{engine_name}_{idx}"
                        )
                        
                        if idx < len(engine_names) - 1:
                            st.divider()
                    
                    st.caption("""
                    **Легенда:** 
                    🟢 Зелёный - все модели совпадают | 
                    🟡 Жёлтый - две модели совпадают | 
                    🔴 Красный - только одна модель или расхождения
                    """)
                else:
                    st.warning("Недостаточно результатов для сравнения (нужно минимум 2 движка)")
                    
                    # Still show individual results
                    st.subheader("📄 Результаты по движкам")
                    for engine_name, engine_text in zip(engine_names, engine_texts):
                        st.markdown(f"### {engine_name}")
                        st.text_area(f"Текст от {engine_name}", engine_text, height=200, key=f"text_{engine_name}")
            
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
