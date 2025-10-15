"""
صفحه استخراج متن از فایل صوتی با Whisper
Audio Transcription Page using Whisper
"""

import streamlit as st
import os
import tempfile
import whisper
from pathlib import Path
import time

# تنظیمات صفحه
st.set_page_config(
    page_title="🎤 استخراج متن از صدا",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استایل‌های سفارشی
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .step-header {
        color: #ff6b6b;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .transcription-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# هدر اصلی
st.markdown('<h1 class="main-header">🎤 استخراج متن از فایل صوتی</h1>', unsafe_allow_html=True)
st.markdown("### تبدیل فایل‌های صوتی به متن با استفاده از Whisper AI")

# نوار کناری برای تنظیمات
with st.sidebar:
    st.header("⚙️ تنظیمات Whisper")
    
    # انتخاب مدل Whisper
    model_size = st.selectbox(
        "اندازه مدل Whisper",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="مدل‌های بزرگتر دقت بالاتری دارند اما کندتر هستند"
    )
    
    # انتخاب زبان
    language = st.selectbox(
        "زبان فایل صوتی",
        ["Auto-detect", "English", "Persian", "Arabic", "French", "German", 
         "Spanish", "Italian", "Portuguese", "Russian", "Chinese", "Japanese", 
         "Korean", "Turkish", "Dutch", "Swedish", "Norwegian", "Danish"],
        index=0,
        help="انتخاب زبان صحیح دقت را افزایش می‌دهد"
    )
    
    # تنظیمات پیشرفته
    st.subheader("🔧 تنظیمات پیشرفته")
    
    # Word-level timestamps
    word_timestamps = st.checkbox(
        "زمان‌بندی کلمه‌ای",
        value=False,
        help="نمایش زمان شروع و پایان هر کلمه"
    )
    
    # Temperature
    temperature = st.slider(
        "Temperature (خلاقیت)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="مقدار بالاتر = خلاقیت بیشتر، مقدار پایین = دقت بیشتر"
    )
    
    # Best of
    best_of = st.slider(
        "تعداد تلاش (Best of)",
        min_value=1,
        max_value=5,
        value=1,
        help="تعداد تلاش برای بهترین نتیجه"
    )

# محتوای اصلی
st.markdown('<h2 class="step-header">📁 مرحله 1: آپلود فایل صوتی</h2>', unsafe_allow_html=True)

# آپلود فایل
uploaded_file = st.file_uploader(
    "فایل صوتی خود را آپلود کنید",
    type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'wma'],
    help="فرمت‌های پشتیبانی شده: MP3, WAV, M4A, FLAC, OGG, WMA"
)

if uploaded_file is not None:
    # نمایش اطلاعات فایل
    file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)  # MB
    st.success(f"✅ فایل آپلود شد: {uploaded_file.name} ({file_size:.2f} MB)")
    
    # ذخیره فایل موقت
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state['audio_file_path'] = str(temp_file_path)
    st.session_state['audio_file_name'] = uploaded_file.name

# مرحله 2: استخراج متن
if st.session_state.get('audio_file_path'):
    st.markdown('<h2 class="step-header">🔍 مرحله 2: استخراج متن</h2>', unsafe_allow_html=True)
    
    # نمایش تنظیمات انتخاب شده
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**مدل:** {model_size}")
    with col2:
        st.info(f"**زبان:** {language}")
    with col3:
        st.info(f"**فایل:** {st.session_state.get('audio_file_name', 'نامشخص')}")
    
    if st.button("🚀 شروع استخراج متن", type="primary"):
        with st.spinner("در حال بارگذاری مدل Whisper و استخراج متن..."):
            try:
                # بارگذاری مدل
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 بارگذاری مدل Whisper...")
                progress_bar.progress(20)
                
                model = whisper.load_model(model_size)
                progress_bar.progress(40)
                
                # تنظیمات Whisper
                whisper_options = {
                    "temperature": temperature,
                    "best_of": best_of,
                    "word_timestamps": word_timestamps
                }
                
                # تنظیم زبان
                if language != "Auto-detect":
                    language_map = {
                        "English": "en", "Persian": "fa", "Arabic": "ar", 
                        "French": "fr", "German": "de", "Spanish": "es",
                        "Italian": "it", "Portuguese": "pt", "Russian": "ru",
                        "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
                        "Turkish": "tr", "Dutch": "nl", "Swedish": "sv",
                        "Norwegian": "no", "Danish": "da"
                    }
                    whisper_options["language"] = language_map.get(language, "en")
                
                status_text.text("🎤 در حال استخراج متن از فایل صوتی...")
                progress_bar.progress(60)
                
                # استخراج متن
                result = model.transcribe(st.session_state['audio_file_path'], **whisper_options)
                progress_bar.progress(90)
                
                # ذخیره نتیجه
                st.session_state['transcription_result'] = result
                st.session_state['transcription_text'] = result["text"]
                
                progress_bar.progress(100)
                status_text.text("✅ استخراج متن با موفقیت انجام شد!")
                
                # پاکسازی فایل موقت
                if os.path.exists(st.session_state['audio_file_path']):
                    os.remove(st.session_state['audio_file_path'])
                
                st.success("🎉 متن با موفقیت استخراج شد!")
                
            except Exception as e:
                st.error(f"❌ خطا در استخراج متن: {str(e)}")
                st.error("لطفاً مطمئن شوید که فایل صوتی معتبر است و دوباره تلاش کنید.")

# مرحله 3: نمایش نتایج
if st.session_state.get('transcription_result'):
    st.markdown('<h2 class="step-header">📝 مرحله 3: نتایج استخراج</h2>', unsafe_allow_html=True)
    
    result = st.session_state['transcription_result']
    
    # تب‌های مختلف برای نمایش نتایج
    tab1, tab2, tab3 = st.tabs(["📄 متن کامل", "⏰ با زمان‌بندی", "📊 جزئیات فنی"])
    
    with tab1:
        st.markdown("### متن استخراج شده:")
        st.markdown(f'<div class="transcription-box">{st.session_state["transcription_text"]}</div>', 
                   unsafe_allow_html=True)
        
        # دکمه کپی
        if st.button("📋 کپی متن"):
            st.write("متن در کلیپ‌بورد کپی شد!")
            st.code(st.session_state["transcription_text"])
    
    with tab2:
        if result.get("segments"):
            st.markdown("### متن با زمان‌بندی:")
            for i, segment in enumerate(result["segments"]):
                start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
                end_time = f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}"
                
                st.markdown(f"**{start_time} - {end_time}:** {segment['text']}")
        else:
            st.info("زمان‌بندی در دسترس نیست. برای فعال‌سازی، گزینه 'زمان‌بندی کلمه‌ای' را در تنظیمات فعال کنید.")
    
    with tab3:
        st.markdown("### اطلاعات فنی:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("مدت زمان فایل", f"{result.get('duration', 0):.2f} ثانیه")
            st.metric("تعداد سگمنت‌ها", len(result.get('segments', [])))
            st.metric("زبان تشخیص داده شده", result.get('language', 'نامشخص'))
        
        with col2:
            st.metric("مدل استفاده شده", model_size)
            st.metric("Temperature", temperature)
            st.metric("Best of", best_of)
        
        # نمایش جزئیات کامل
        if st.checkbox("نمایش جزئیات کامل JSON"):
            st.json(result)
    
    # مرحله 4: دانلود نتایج
    st.markdown('<h2 class="step-header">💾 مرحله 4: دانلود نتایج</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # دانلود متن ساده
        st.download_button(
            label="📄 دانلود متن (TXT)",
            data=st.session_state["transcription_text"],
            file_name=f"transcription_{st.session_state.get('audio_file_name', 'audio')}.txt",
            mime="text/plain",
            type="primary"
        )
    
    with col2:
        # دانلود SRT
        if result.get("segments"):
            srt_content = []
            for i, segment in enumerate(result["segments"]):
                start_time = segment['start']
                end_time = segment['end']
                
                start_str = '{:02d}:{:02d}:{:02d},{:03d}'.format(
                    int(start_time // 3600),
                    int((start_time % 3600) // 60),
                    int(start_time % 60),
                    int((start_time % 1) * 1000)
                )
                end_str = '{:02d}:{:02d}:{:02d},{:03d}'.format(
                    int(end_time // 3600),
                    int((end_time % 3600) // 60),
                    int(end_time % 60),
                    int((end_time % 1) * 1000)
                )
                srt_content.append(f"{i+1}\n{start_str} --> {end_str}\n{segment['text']}\n")
            
            srt_text = '\n'.join(srt_content)
            
            st.download_button(
                label="📝 دانلود زیرنویس (SRT)",
                data=srt_text,
                file_name=f"subtitles_{st.session_state.get('audio_file_name', 'audio')}.srt",
                mime="text/plain",
                type="secondary"
            )
        else:
            st.info("برای دانلود SRT، زمان‌بندی کلمه‌ای را فعال کنید")
    
    with col3:
        # دانلود JSON
        import json
        json_data = json.dumps(result, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📊 دانلود JSON",
            data=json_data,
            file_name=f"transcription_data_{st.session_state.get('audio_file_name', 'audio')}.json",
            mime="application/json",
            type="secondary"
        )

# پاکسازی
if st.button("🧹 پاکسازی و شروع مجدد", type="secondary"):
    # پاکسازی session state
    for key in ['audio_file_path', 'audio_file_name', 'transcription_result', 'transcription_text']:
        if key in st.session_state:
            del st.session_state[key]
    
    # پاکسازی فایل‌های موقت
    temp_dir = Path("temp")
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            file.unlink()
    
    st.success("✅ پاکسازی انجام شد")
    st.rerun()

# راهنمای استفاده
with st.expander("ℹ️ راهنمای استفاده"):
    st.markdown("""
    ### نحوه استفاده:
    1. **آپلود فایل**: فایل صوتی خود را آپلود کنید (MP3, WAV, M4A, FLAC, OGG, WMA)
    2. **تنظیمات**: مدل و زبان مناسب را انتخاب کنید
    3. **استخراج**: دکمه "شروع استخراج متن" را فشار دهید
    4. **نتایج**: متن استخراج شده را مشاهده و دانلود کنید
    
    ### نکات مهم:
    - **مدل‌های Whisper:**
      - `tiny`: سریع‌ترین، کم‌دقت‌ترین
      - `base`: متعادل (توصیه می‌شود)
      - `small`: دقت بالاتر
      - `medium`: دقت بالا
      - `large`: بالاترین دقت، کندترین
    
    - **زبان**: انتخاب زبان صحیح دقت را افزایش می‌دهد
    - **زمان‌بندی کلمه‌ای**: برای ایجاد فایل SRT فعال کنید
    - **Temperature**: 0.0 برای دقت بالا، 1.0 برای خلاقیت بیشتر
    
    ### محدودیت‌ها:
    - فایل‌های خیلی طولانی ممکن است زمان زیادی نیاز داشته باشند
    - کیفیت صدا بر دقت تأثیر می‌گذارد
    - مدل‌های بزرگتر نیاز به حافظه بیشتری دارند
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎤 استخراج متن از صدا - ساخته شده با Streamlit و Whisper AI</p>
</div>
""", unsafe_allow_html=True)
