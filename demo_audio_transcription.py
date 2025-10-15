#!/usr/bin/env python3
"""
نمایش نحوه استفاده از صفحه استخراج متن از صدا
Demo script for audio transcription page
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """نمایش نحوه استفاده از صفحه استخراج متن از صدا"""
    print("🎤 نمایش نحوه استفاده از صفحه استخراج متن از صدا")
    print("=" * 60)
    
    # بررسی وجود فایل‌های لازم
    required_files = [
        "audio_transcription.py",
        "run_audio_transcription.py",
        "AUDIO_TRANSCRIPTION_GUIDE.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ فایل‌های زیر یافت نشدند:")
        for file in missing_files:
            print(f"   - {file}")
        return
    
    print("✅ تمام فایل‌های لازم موجود هستند")
    print()
    
    # نمایش راهنمای استفاده
    print("📋 راهنمای استفاده:")
    print("1. برای اجرای صفحه استخراج متن از صدا:")
    print("   python run_audio_transcription.py")
    print()
    print("2. یا مستقیماً با Streamlit:")
    print("   streamlit run audio_transcription.py --server.port 8502")
    print()
    print("3. صفحه در مرورگر باز می‌شود:")
    print("   http://localhost:8502")
    print()
    
    # نمایش ویژگی‌ها
    print("✨ ویژگی‌های صفحه:")
    print("   • پشتیبانی از فرمت‌های مختلف صوتی (MP3, WAV, M4A, FLAC, OGG, WMA)")
    print("   • انتخاب مدل Whisper (tiny تا large)")
    print("   • تشخیص خودکار زبان یا انتخاب دستی")
    print("   • زمان‌بندی کلمه‌ای برای ایجاد زیرنویس")
    print("   • تنظیمات پیشرفته (Temperature, Best of)")
    print("   • دانلود نتایج در فرمت‌های مختلف (TXT, SRT, JSON)")
    print()
    
    # نمایش مدل‌های Whisper
    print("🤖 مدل‌های Whisper:")
    models = [
        ("tiny", "خیلی سریع", "کم", "~1 GB"),
        ("base", "سریع", "متوسط", "~1 GB"),
        ("small", "متوسط", "خوب", "~2 GB"),
        ("medium", "کند", "بالا", "~5 GB"),
        ("large", "خیلی کند", "عالی", "~10 GB")
    ]
    
    print("   مدل        سرعت        دقت        حافظه")
    print("   " + "-" * 40)
    for model, speed, accuracy, memory in models:
        print(f"   {model:<8} {speed:<12} {accuracy:<10} {memory}")
    print()
    
    # نمایش زبان‌های پشتیبانی شده
    print("🌍 زبان‌های پشتیبانی شده:")
    languages = [
        "انگلیسی", "فارسی", "عربی", "فرانسوی", "آلمانی", "اسپانیایی",
        "ایتالیایی", "پرتغالی", "روسی", "چینی", "ژاپنی", "کره‌ای",
        "ترکی", "هلندی", "سوئدی", "نروژی", "دانمارکی"
    ]
    print("   " + ", ".join(languages))
    print()
    
    # نمایش فرمت‌های خروجی
    print("📁 فرمت‌های خروجی:")
    print("   • TXT: متن ساده")
    print("   • SRT: زیرنویس با زمان‌بندی")
    print("   • JSON: داده‌های کامل")
    print()
    
    # سوال برای اجرا
    print("🚀 آیا می‌خواهید صفحه را اجرا کنید؟ (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes', 'بله', 'ب']:
            print("\n🎬 در حال اجرای صفحه...")
            print("   صفحه در مرورگر باز می‌شود...")
            print("   برای توقف، Ctrl+C را فشار دهید")
            print()
            
            # اجرای صفحه
            try:
                subprocess.run([
                    sys.executable, "run_audio_transcription.py"
                ])
            except KeyboardInterrupt:
                print("\n👋 خروج از برنامه...")
        else:
            print("👋 خروج از برنامه...")
    except KeyboardInterrupt:
        print("\n👋 خروج از برنامه...")

if __name__ == "__main__":
    main()
