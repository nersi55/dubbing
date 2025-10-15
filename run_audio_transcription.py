#!/usr/bin/env python3
"""
راه‌انداز صفحه استخراج متن از صدا
Audio Transcription Page Launcher
"""

import subprocess
import sys
import os

def main():
    """راه‌اندازی صفحه استخراج متن از صدا"""
    print("🎤 راه‌اندازی صفحه استخراج متن از صدا...")
    print("=" * 50)
    
    # بررسی وجود فایل
    if not os.path.exists("audio_transcription.py"):
        print("❌ فایل audio_transcription.py یافت نشد!")
        return
    
    try:
        # اجرای Streamlit
        print("🚀 در حال اجرای Streamlit...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "audio_transcription.py",
            "--server.port", "8502",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 خروج از برنامه...")
    except Exception as e:
        print(f"❌ خطا در اجرا: {e}")

if __name__ == "__main__":
    main()
