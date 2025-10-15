"""
دوبله خودکار ویدیو - توابع اصلی
Auto Video Dubbing - Core Functions
"""

import os
import re
import time
import base64
import struct
import tempfile
import subprocess
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

import yt_dlp
import pysrt
import google.generativeai as genai
from google.genai import types
import google.genai as genai_client
try:
    from pydub import AudioSegment
except ImportError:
    # Fallback for Python 3.13 compatibility
    import subprocess
    import tempfile
    import os
    
    class AudioSegment:
        @staticmethod
        def from_file(file_path):
            # Simple fallback implementation
            return SimpleAudioSegment(file_path)
        
        @staticmethod
        def silent(duration):
            return SimpleAudioSegment(None, duration=duration)
    
    class SimpleAudioSegment:
        def __init__(self, file_path=None, duration=None):
            self.file_path = file_path
            self.duration = duration or 0
        
        def __len__(self):
            return int(self.duration * 1000)  # Convert to milliseconds
        
        def __add__(self, other):
            if isinstance(other, (int, float)):
                # Volume adjustment
                return self
            return self
        
        def overlay(self, other, position=0):
            return self
        
        def export(self, output_path, format="wav"):
            if self.file_path and os.path.exists(self.file_path):
                # Copy file
                subprocess.run(['cp', self.file_path, output_path], check=True)
            else:
                # Create silent audio with proper duration
                duration_seconds = max(self.duration, 0.1)  # حداقل 0.1 ثانیه
                subprocess.run([
                    'ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=duration={duration_seconds}',
                    '-ac', '2', '-ar', '44100', '-y', str(output_path)
                ], check=True, capture_output=True)
from youtube_transcript_api import YouTubeTranscriptApi
import whisper


class VideoDubbingApp:
    def __init__(self, api_key: str):
        """Initialize the dubbing application with Google API key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.client = genai_client.Client(api_key=api_key)
        
        # Create necessary directories
        self.work_dir = Path("dubbing_work")
        self.work_dir.mkdir(exist_ok=True)
        self.segments_dir = self.work_dir / "dubbed_segments"
        self.segments_dir.mkdir(exist_ok=True)
        
    def clean_previous_files(self):
        """پاکسازی فایل‌های قبلی"""
        files_to_clean = [
            "input_video.mp4", "audio.wav", "audio.srt", 
            "audio_fa.srt", "final_dubbed_video.mp4"
        ]
        
        for file_name in files_to_clean:
            file_path = self.work_dir / file_name
            if file_path.exists():
                file_path.unlink()
                
        # Clean segments directory
        if self.segments_dir.exists():
            for file in self.segments_dir.glob("*"):
                file.unlink()
    
    def download_youtube_video(self, url: str) -> bool:
        """دانلود ویدیو از یوتیوب"""
        try:
            # Clean previous files
            for file in self.work_dir.glob('temp_video*'):
                file.unlink()
            
            format_option = 'bestvideo+bestaudio/best'
            temp_filename = str(self.work_dir / 'temp_video.%(ext)s')
            
            video_opts = {
                'format': format_option,
                'outtmpl': temp_filename,
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_warnings': False,
                'quiet': False
            }
            
            with yt_dlp.YoutubeDL(video_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = ydl.prepare_filename(info)
            
            if os.path.exists(downloaded_file):
                _, file_extension = os.path.splitext(downloaded_file)
                final_filename = self.work_dir / f'input_video{file_extension}'
                os.rename(downloaded_file, str(final_filename))
                
                if file_extension.lower() != '.mp4':
                    mp4_path = self.work_dir / 'input_video.mp4'
                    subprocess.run([
                        'ffmpeg', '-i', str(final_filename), 
                        '-c', 'copy', str(mp4_path), '-y'
                    ], check=True, capture_output=True)
                    final_filename.unlink()
                
                # Extract audio
                audio_path = self.work_dir / 'audio.wav'
                subprocess.run([
                    'ffmpeg', '-i', str(self.work_dir / 'input_video.mp4'), 
                    '-vn', str(audio_path), '-y'
                ], check=True, capture_output=True)
                
                return True
            return False
            
        except Exception as e:
            print(f"خطا در دانلود: {str(e)}")
            return False
    
    def extract_transcript_from_youtube(self, url: str, language: str = "Auto-detect") -> bool:
        """استخراج زیرنویس از یوتیوب"""
        try:
            # Extract video ID
            video_id = None
            patterns = [
                r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})',
                r'(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    video_id = match.group(1)
                    break
            
            if not video_id:
                if 'shorts/' in url:
                    shorts_id = url.split('shorts/')[1].split('?')[0].split('&')[0]
                    if len(shorts_id) == 11:
                        video_id = shorts_id
                elif 'youtu.be/' in url:
                    video_id = url.split('youtu.be/')[1].split('?')[0].split('&')[0]
                elif 'v=' in url:
                    video_id = url.split('v=')[1].split('&')[0].split('?')[0]
            
            if not video_id or len(video_id) != 11:
                return False
            
            # Language mapping
            language_map = {
                "Auto-detect": None,
                "English (EN)": "en",
                "Persian (FA)": "fa",
                "German (DE)": "de",
                "French (FR)": "fr",
                "Italian (IT)": "it",
                "Spanish (ES)": "es",
                "Chinese (ZH)": "zh",
                "Korean (KO)": "ko",
                "Russian (RU)": "ru",
                "Arabic (AR)": "ar",
                "Japanese (JA)": "ja",
                "Hindi (HI)": "hi"
            }
            selected_language = language_map.get(language)
            
            # Get transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            if selected_language:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[selected_language])
            else:
                # Auto-detect
                transcript_data = None
                for transcript in transcript_list:
                    if transcript.is_generated:
                        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[transcript.language_code])
                        break
                    elif not transcript.is_translatable:
                        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[transcript.language_code])
                        break
                
                if not transcript_data:
                    for transcript in transcript_list:
                        if transcript.is_translatable:
                            transcript_data = transcript.translate('en').fetch()
                            break
            
            if transcript_data:
                # Get video duration
                result = subprocess.run([
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(self.work_dir / 'input_video.mp4')
                ], capture_output=True, text=True)
                
                video_duration = float(result.stdout.strip())
                
                # Process transcript data
                processed_data = []
                for entry in transcript_data:
                    processed_data.append({
                        'start': entry['start'],
                        'duration': entry.get('duration', 0),
                        'text': entry['text']
                    })
                
                # Sort by start time
                processed_data.sort(key=lambda x: x['start'])
                
                # Clean overlapping subtitles
                cleaned_data = []
                if processed_data:
                    cleaned_data.append(processed_data[0])
                    for i in range(1, len(processed_data)):
                        current = processed_data[i]
                        previous = cleaned_data[-1]
                        prev_end = previous['start'] + previous['duration']
                        
                        if current['start'] < prev_end:
                            if current['start'] + current['duration'] <= prev_end:
                                previous['text'] += " " + current['text']
                            else:
                                overlap = prev_end - current['start']
                                new_duration = current['duration'] - overlap
                                
                                if new_duration > 0.3:
                                    current['start'] = prev_end
                                    current['duration'] = new_duration
                                    cleaned_data.append(current)
                                else:
                                    previous['text'] += " " + current['text']
                                    previous['duration'] = max(previous['duration'],
                                                             (current['start'] + current['duration']) - previous['start'])
                        else:
                            cleaned_data.append(current)
                
                # Convert to SRT format
                srt_content = []
                for i, entry in enumerate(cleaned_data):
                    start_time = entry['start']
                    duration = entry['duration']
                    end_time = start_time + duration
                    
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
                    srt_content.append(f"{i+1}\n{start_str} --> {end_str}\n{entry['text']}\n")
                
                # Save SRT file
                srt_path = self.work_dir / 'audio.srt'
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(srt_content))
                
                return True
            
            return False
            
        except Exception as e:
            print(f"خطا در استخراج زیرنویس: {str(e)}")
            return False
    
    def extract_audio_with_whisper(self) -> bool:
        """استخراج متن از صدا با Whisper"""
        try:
            audio_path = self.work_dir / 'audio.wav'
            if not audio_path.exists():
                return False
            
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path))
            
            # Convert to SRT format
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
            
            srt_path = self.work_dir / 'audio.srt'
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            return True
            
        except Exception as e:
            print(f"خطا در استخراج صدا با Whisper: {str(e)}")
            return False
    
    def compress_srt_dialogues(self, merge_count: int = 3) -> bool:
        """فشرده‌سازی دیالوگ‌های SRT"""
        try:
            srt_path = self.work_dir / 'audio.srt'
            if not srt_path.exists():
                return False
            
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT
            subtitle_blocks = content.strip().split('\n\n')
            subtitles = []
            
            for block in subtitle_blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 2:
                    try:
                        time_line_index = -1
                        for i, line in enumerate(lines):
                            if '-->' in line:
                                time_line_index = i
                                break
                        if time_line_index != -1:
                            time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[time_line_index])
                            start_time, end_time = time_match.groups()
                            text = '\n'.join(lines[time_line_index+1:])
                            subtitles.append({'start': start_time, 'end': end_time, 'text': text})
                    except Exception:
                        continue
            
            if not subtitles:
                return False
            
            # Merge subtitles
            merged_subs = []
            new_index = 1
            for i in range(0, len(subtitles), merge_count):
                chunk = subtitles[i:i+merge_count]
                if not chunk:
                    continue
                start_time = chunk[0]['start']
                end_time = chunk[-1]['end']
                # ترکیب بهتر متن‌ها با نقطه‌گذاری مناسب
                combined_text = ' '.join([sub['text'].replace('\n', ' ').strip() for sub in chunk])
                # حذف فاصله‌های اضافی
                combined_text = ' '.join(combined_text.split())
                # اضافه کردن نقطه در انتها اگر وجود نداشته باشد
                if combined_text and not combined_text.endswith(('.', '!', '?', '،', ':')):
                    combined_text += '.'
                merged_subs.append({'index': new_index, 'start': start_time, 'end': end_time, 'text': combined_text})
                new_index += 1
            
            # Format SRT
            srt_output = []
            for sub in merged_subs:
                srt_output.append(str(sub['index']))
                srt_output.append(f"{sub['start']} --> {sub['end']}")
                srt_output.append(sub['text'])
                srt_output.append('')
            
            # Save compressed SRT
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_output))
            
            return True
            
        except Exception as e:
            print(f"خطا در فشرده‌سازی: {str(e)}")
            return False
    
    def translate_subtitles(self, target_language: str = "Persian (FA)") -> bool:
        """ترجمه زیرنویس‌ها"""
        try:
            srt_path = self.work_dir / 'audio.srt'
            if not srt_path.exists():
                return False
            
            subs = pysrt.open(str(srt_path), encoding='utf-8')
            
            # Translation models (بهترتیب کیفیت)
            translation_models = [
                "gemini-2.5-flash",        # بهترین کیفیت
                "gemini-2.5-flash-lite",   # کیفیت خوب و سریع
                "gemini-flash-lite-latest" # پشتیبان
            ]
            
            def translate_with_fallback(text):
                for model_name in translation_models:
                    try:
                        model = genai.GenerativeModel(
                            model_name,
                            safety_settings={
                                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                            }
                        )
                        
                        if target_language == "Persian (FA)":
                            prompt = f"""
                            شما یک مترجم حرفه‌ای و باتجربه هستید که در ترجمه زیرنویس‌های سینمایی و تلویزیونی تخصص دارید. 
                            
                            دستورالعمل‌های ترجمه:
                            1. متن را به فارسی روان، طبیعی و قابل فهم ترجمه کنید
                            2. از اصطلاحات و عبارات روزمره فارسی استفاده کنید
                            3. ساختار جمله‌ها را متناسب با فارسی تغییر دهید
                            4. از کلمات فارسی معادل استفاده کنید (نه کلمات عربی)
                            5. لحن متن را حفظ کنید (رسمی/غیررسمی، جدی/شوخ)
                            6. اصطلاحات و کنایه‌ها را به معادل فارسی ترجمه کنید
                            7. از نقطه‌گذاری صحیح فارسی استفاده کنید
                            8. متن نهایی باید کاملاً روان و طبیعی باشد

                            پاسخ شما باید **فقط و فقط** شامل متن ترجمه شده باشد.

                            مثال‌های ترجمه خوب:
                            ورودی: <text>What's up, dude?</text>
                            خروجی: چطوری رفیق؟

                            ورودی: <text>I can't believe this is happening.</text>
                            خروجی: باور نمی‌کنم این داره اتفاق می‌افته.

                            ورودی: <text>Let's get this party started!</text>
                            خروجی: بیا این مهمونی رو راه بندازیم!

                            حالا متن زیر را با رعایت تمام دستورالعمل‌ها ترجمه کن:
                            ورودی: <text>{text}</text>
                            خروجی:
                            """
                        else:
                            language_map = {
                                "English (EN)": "English", "German (DE)": "German", 
                                "French (FR)": "French", "Italian (IT)": "Italian", 
                                "Spanish (ES)": "Spanish", "Chinese (ZH)": "Chinese", 
                                "Korean (KO)": "Korean", "Russian (RU)": "Russian", 
                                "Arabic (AR)": "Arabic", "Japanese (JA)": "Japanese", 
                                "Hindi (HI)": "Hindi"
                            }
                            target_lang_name = language_map.get(target_language, "English")
                            prompt = f"""
                            You are an expert subtitle translator. Your only task is to translate the text inside the <text> tag to {target_lang_name}.
                            Your response must ONLY be the translated text. Do not add any explanation, prefix, or extra words.

                            Example:
                            Input: <text>سلام، این یک آزمایش است.</text>
                            Output: Hello, this is a test.

                            Now, translate the following:
                            Input: <text>{text}</text>
                            Output:
                            """
                        
                        response = model.generate_content(prompt)
                        time.sleep(2)  # Rate limiting
                        return response.text.strip()
                        
                    except Exception as e:
                        print(f"خطا در مدل {model_name}: {str(e)}")
                        time.sleep(3)
                        continue
                
                return text  # Return original text if all models fail
            
            # Translate all subtitles with better context
            print(f"🔄 شروع ترجمه {len(subs)} زیرنویس...")
            for i, sub in enumerate(subs):
                print(f"📝 ترجمه زیرنویس {i+1}/{len(subs)}: {sub.text[:50]}...")
                sub.text = translate_with_fallback(sub.text)
                time.sleep(1)  # کاهش سرعت برای کیفیت بهتر
            
            # Save translated subtitles
            translated_path = self.work_dir / 'audio_fa.srt'
            subs.save(str(translated_path), encoding='utf-8')
            
            return True
            
        except Exception as e:
            print(f"خطا در ترجمه: {str(e)}")
            return False
    
    def parse_audio_mime_type(self, mime_type: str) -> dict:
        """Parse audio MIME type for conversion"""
        parts = mime_type.split(";")
        details = {'bits_per_sample': 16, 'rate': 24000}
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                details['rate'] = int(param.split("=", 1)[1])
            elif param.startswith("audio/L"):
                details['bits_per_sample'] = int(param.split("L", 1)[1])
        return details
    
    def convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Convert audio data to WAV format"""
        parameters = self.parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
            num_channels, sample_rate, byte_rate, block_align,
            bits_per_sample, b"data", data_size
        )
        return header + audio_data
    
    def generate_tts_segment(self, text: str, voice: str, model: str, output_path: str, 
                           speech_prompt: str = "", max_retries: int = 3) -> Optional[str]:
        """تولید سگمنت صوتی با Gemini TTS"""
        for attempt in range(1, max_retries + 1):
            try:
                if speech_prompt and speech_prompt.strip():
                    final_text = f"{speech_prompt.strip()}: \"{text}\""
                else:
                    final_text = text
                
                contents = [types.Content(role="user", parts=[types.Part.from_text(text=final_text)])]
                generate_content_config = types.GenerateContentConfig(
                    response_modalities=["audio"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                )
                
                stream = self.client.models.generate_content_stream(
                    model=model, contents=contents, config=generate_content_config,
                )
                
                audio_data_buffer = b""
                mime_type = ""
                for chunk in stream:
                    if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                        part = chunk.candidates[0].content.parts[0]
                        if part.inline_data:
                            audio_data_buffer += part.inline_data.data
                            mime_type = part.inline_data.mime_type
                
                if audio_data_buffer and mime_type:
                    final_wav_data = self.convert_to_wav(audio_data_buffer, mime_type)
                    with open(output_path, 'wb') as f:
                        f.write(final_wav_data)
                    return output_path
                else:
                    raise Exception("هیچ داده صوتی از API دریافت نشد.")
                    
            except Exception as e:
                print(f"خطا در تولید صدای Gemini (تلاش {attempt}/{max_retries}): {str(e)}")
                if attempt < max_retries:
                    wait_time = 9 * attempt
                    print(f"انتظار برای {wait_time} ثانیه...")
                    time.sleep(wait_time)
                else:
                    print(f"تولید صدا برای قطعه '{text[:50]}...' ناموفق بود.")
                    return None
        return None
    
    def create_audio_segments(self, voice: str = "Fenrir", model: str = "gemini-2.5-flash-preview-tts",
                            speech_prompt: str = "", sleep_between_requests: int = 30) -> bool:
        """ایجاد سگمنت‌های صوتی با مدیریت هوشمند محدودیت‌ها"""
        try:
            srt_path = self.work_dir / 'audio_fa.srt'
            if not srt_path.exists():
                return False
            
            subs = pysrt.open(str(srt_path), encoding='utf-8')
            total_segments = len(subs)
            
            # محاسبه فشرده‌سازی خودکار
            if total_segments > 15:  # اگر بیشتر از محدودیت روزانه باشد
                auto_merge_count = min(15, max(3, total_segments // 10))
                print(f"⚠️ تعداد سگمنت‌ها ({total_segments}) بیشتر از محدودیت API است.")
                print(f"🔄 فشرده‌سازی خودکار با ضریب {auto_merge_count} فعال می‌شود...")
                self.compress_srt_dialogues(auto_merge_count)
                subs = pysrt.open(str(srt_path), encoding='utf-8')
                total_segments = len(subs)
                print(f"✅ تعداد سگمنت‌ها به {total_segments} کاهش یافت.")
            
            # مدیریت batch ها
            batch_size = 3
            batch_delay = 60
            
            for batch_start in range(0, total_segments, batch_size):
                batch_end = min(batch_start + batch_size, total_segments)
                batch_segments = subs[batch_start:batch_end]
                
                print(f"📦 پردازش batch {batch_start//batch_size + 1}: سگمنت‌های {batch_start+1}-{batch_end}")
                
                for i, sub in enumerate(batch_segments):
                    segment_index = batch_start + i + 1
                    print(f"🎧 پردازش سگمنت {segment_index}/{total_segments}...")
                    
                    temp_audio_path = self.segments_dir / f"temp_{segment_index}.wav"
                    final_segment_path = self.segments_dir / f"dub_{segment_index}.wav"
                    
                    # تولید صدا با مدیریت خطا
                    generated_path = self.generate_tts_segment(
                        sub.text, voice, model, str(temp_audio_path), speech_prompt
                    )
                    
                    # انتظار بین سگمنت‌ها
                    if i < len(batch_segments) - 1:
                        print(f"⏱️ استراحت برای {sleep_between_requests} ثانیه...")
                        time.sleep(sleep_between_requests)
                    
                    # مدیریت فایل‌های خالی
                    if not generated_path or not os.path.exists(generated_path):
                        print(f"⚠️ تولید صدای Gemini برای سگمنت {segment_index} ناموفق بود. فایل سکوت ایجاد می‌شود.")
                        start_ms = sub.start.hours * 3600000 + sub.start.minutes * 60000 + sub.start.seconds * 1000 + sub.start.milliseconds
                        end_ms = sub.end.hours * 3600000 + sub.end.minutes * 60000 + sub.end.seconds * 1000 + sub.end.milliseconds
                        target_duration_ms = max(end_ms - start_ms, 100)  # حداقل 100ms
                        
                        # ایجاد فایل سکوت با FFmpeg
                        try:
                            subprocess.run([
                                'ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=duration={target_duration_ms/1000.0}',
                                '-ac', '2', '-ar', '44100', '-y', str(final_segment_path)
                            ], check=True, capture_output=True)
                            print(f"   ✅ فایل سکوت برای سگمنت {segment_index} ایجاد شد.")
                        except Exception as e:
                            print(f"   ❌ خطا در ایجاد فایل سکوت: {e}")
                        continue
                    
                    try:
                        # تنظیم زمان‌بندی
                        start_ms = sub.start.hours * 3600000 + sub.start.minutes * 60000 + sub.start.seconds * 1000 + sub.start.milliseconds
                        end_ms = sub.end.hours * 3600000 + sub.end.minutes * 60000 + sub.end.seconds * 1000 + sub.end.milliseconds
                        target_duration = (end_ms - start_ms) / 1000.0
                        if target_duration <= 0:
                            target_duration = 0.5
                        
                        sound = AudioSegment.from_file(generated_path)
                        original_duration = len(sound) / 1000.0
                        
                        if original_duration == 0:
                            raise ValueError("فایل صوتی تولید شده خالی است.")
                        
                        speed_factor = original_duration / target_duration
                        speed_factor = max(0.5, min(speed_factor, 2.5))
                        
                        print(f"   - زمان هدف: {target_duration:.2f}s | زمان اصلی: {original_duration:.2f}s | ضریب سرعت: {speed_factor:.2f}")
                        
                        subprocess.run([
                            'ffmpeg', '-i', generated_path,
                            '-filter:a', f'rubberband=tempo={speed_factor}',
                            '-y', str(final_segment_path)
                        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        print(f"   ✅ سگمنت {segment_index} با موفقیت ساخته و زمان‌بندی شد.")
                        
                    except Exception as e:
                        print(f"   ❌ خطا در زمان‌بندی سگمنت {segment_index}: {e}")
                        if os.path.exists(generated_path):
                            os.rename(generated_path, str(final_segment_path))
                
                # انتظار بین batch ها
                if batch_end < total_segments:
                    print(f"⏳ انتظار {batch_delay} ثانیه قبل از batch بعدی...")
                    time.sleep(batch_delay)
            
            print("="*50)
            print("🎉 تمام سگمنت‌های صوتی با مدیریت هوشمند محدودیت‌ها ساخته شدند!")
            return True
            
        except Exception as e:
            print(f"خطا در ایجاد سگمنت‌های صوتی: {str(e)}")
            return False
    
    def create_final_video(self, keep_original_audio: bool = False, 
                          original_audio_volume: float = 0.8) -> Optional[str]:
        """ایجاد ویدیو نهایی دوبله شده"""
        try:
            video_path = self.work_dir / 'input_video.mp4'
            srt_path = self.work_dir / 'audio_fa.srt'
            
            if not video_path.exists() or not srt_path.exists():
                print("❌ فایل ویدیو یا زیرنویس یافت نشد")
                return None
            
            subs = pysrt.open(str(srt_path), encoding='utf-8')
            print(f"📝 تعداد زیرنویس‌ها: {len(subs)}")
            
            # بررسی فایل‌های صوتی موجود
            available_segments = []
            for i in range(1, len(subs) + 1):
                segment_path = self.segments_dir / f"dub_{i}.wav"
                if segment_path.exists():
                    available_segments.append((i, segment_path))
            
            print(f"🎵 فایل‌های صوتی موجود: {len(available_segments)} از {len(subs)}")
            
            if not available_segments:
                print("❌ هیچ فایل صوتی یافت نشد")
                return None
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Extract original audio
                print("🎵 استخراج صدای اصلی...")
                original_audio_path = temp_dir / "original_audio.wav"
                subprocess.run([
                    'ffmpeg', '-i', str(video_path), '-vn',
                    '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                    '-y', str(original_audio_path)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Get video duration
                result = subprocess.run([
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(video_path)
                ], capture_output=True, text=True)
                video_duration = float(result.stdout.strip())
                print(f"⏱️ مدت ویدیو: {video_duration:.2f} ثانیه")
                
                # Create base audio (silent or original)
                if keep_original_audio:
                    print("🔊 حفظ صدای اصلی...")
                    base_audio = AudioSegment.from_file(str(original_audio_path))
                    volume_reduction = - (60 * (1 - original_audio_volume))
                    base_audio = base_audio + volume_reduction
                else:
                    print("🔇 ایجاد صدای سکوت...")
                    base_audio = AudioSegment.silent(duration=int(video_duration * 1000))
                
                # Overlay dubbing segments
                print("🎤 اضافه کردن سگمنت‌های دوبله...")
                final_audio = base_audio
                
                for i, (segment_num, segment_path) in enumerate(available_segments):
                    try:
                        print(f"   📁 پردازش سگمنت {segment_num}...")
                        segment_audio = AudioSegment.from_file(str(segment_path))
                        
                        # محاسبه زمان شروع
                        sub = subs[segment_num - 1]
                        start_time_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
                        
                        if start_time_ms < 0:
                            start_time_ms = 0
                        
                        print(f"      ⏰ زمان شروع: {start_time_ms/1000:.2f}s")
                        print(f"      🎵 مدت صدا: {len(segment_audio)/1000:.2f}s")
                        
                        # اضافه کردن به صدا
                        final_audio = final_audio.overlay(segment_audio, position=start_time_ms)
                        print(f"      ✅ سگمنت {segment_num} اضافه شد")
                        
                    except Exception as e:
                        print(f"      ❌ خطا در سگمنت {segment_num}: {str(e)}")
                        continue
                
                # Export final audio
                print("💾 ذخیره صدای نهایی...")
                merged_audio_path = temp_dir / "merged_audio.wav"
                final_audio.export(str(merged_audio_path), format="wav")
                
                # Create final video using the working method
                print("🎬 ایجاد ویدیو نهایی...")
                output_path = self.work_dir / 'final_dubbed_video.mp4'
                
                # روش کارآمد: استفاده از concat برای ترکیب فایل‌های صوتی
                # ابتدا فایل لیست صوتی ایجاد می‌کنیم
                audio_list_file = temp_dir / "audio_list.txt"
                with open(audio_list_file, 'w') as f:
                    for i, (segment_num, segment_path) in enumerate(available_segments):
                        f.write(f"file '{segment_path.absolute()}'\n")
                
                # ترکیب فایل‌های صوتی
                combined_audio = temp_dir / "combined_audio.wav"
                subprocess.run([
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(audio_list_file),
                    '-c', 'copy', '-y', str(combined_audio)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # بررسی و تنظیم زمان‌بندی صدا
                print("⏱️ بررسی زمان‌بندی صدا...")
                
                # دریافت مدت زمان ویدیو
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
                    str(video_path)
                ], capture_output=True, text=True)
                
                import json
                video_info = json.loads(result.stdout)
                video_duration = float(video_info['format']['duration'])
                
                # دریافت مدت زمان صدا
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
                    str(combined_audio)
                ], capture_output=True, text=True)
                
                audio_info = json.loads(result.stdout)
                audio_duration = float(audio_info['format']['duration'])
                
                print(f"   📹 مدت ویدیو: {video_duration:.2f} ثانیه")
                print(f"   🎵 مدت صدا: {audio_duration:.2f} ثانیه")
                
                # تنظیم سرعت صدا اگر لازم باشد
                if audio_duration > video_duration:
                    speed_factor = audio_duration / video_duration
                    print(f"   ⚡ تنظیم سرعت صدا: {speed_factor:.2f}x")
                    
                    adjusted_audio = temp_dir / "adjusted_audio.wav"
                    subprocess.run([
                        'ffmpeg', '-i', str(combined_audio),
                        '-filter:a', f'rubberband=tempo={speed_factor}',
                        '-y', str(adjusted_audio)
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # بررسی مدت زمان بعد از تنظیم
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
                        str(adjusted_audio)
                    ], capture_output=True, text=True)
                    
                    adjusted_info = json.loads(result.stdout)
                    adjusted_duration = float(adjusted_info['format']['duration'])
                    print(f"   🎵 مدت صدا بعد از تنظیم: {adjusted_duration:.2f} ثانیه")
                    
                    final_audio = adjusted_audio
                else:
                    print("   ✅ زمان‌بندی صدا مناسب است")
                    final_audio = combined_audio
                
                # ایجاد ویدیو نهایی
                subprocess.run([
                    'ffmpeg', '-i', str(video_path), '-i', str(final_audio),
                    '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v', '-map', '1:a',
                    '-shortest', '-y', str(output_path)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                print(f"✅ ویدیو نهایی ایجاد شد: {output_path}")
                return str(output_path)
                
        except Exception as e:
            print(f"❌ خطا در ایجاد ویدیو نهایی: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_subtitled_video(self, subtitle_config: dict = None) -> Optional[str]:
        """ایجاد ویدیو با زیرنویس ترجمه شده با تنظیمات سفارشی"""
        try:
            video_path = self.work_dir / 'input_video.mp4'
            srt_path = self.work_dir / 'audio_fa.srt'
            
            if not video_path.exists() or not srt_path.exists():
                print("❌ فایل ویدیو یا زیرنویس یافت نشد")
                return None
            
            subs = pysrt.open(str(srt_path), encoding='utf-8')
            print(f"📝 تعداد زیرنویس‌ها: {len(subs)}")
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # تنظیمات پیش‌فرض
                default_config = {
                    "font": "Arial",
                    "fontsize": 24,
                    "color": "white",
                    "background_color": "none",
                    "outline_color": "black",
                    "outline_width": 2,
                    "position": "bottom_center",
                    "margin_v": 20,
                    "shadow": 0,
                    "shadow_color": "black",
                    "bold": False,
                    "italic": False
                }
                
                # ادغام تنظیمات سفارشی با پیش‌فرض
                if subtitle_config:
                    config = {**default_config, **subtitle_config}
                else:
                    config = default_config
                
                print(f"🎨 تنظیمات زیرنویس:")
                print(f"   📝 فونت: {config['font']}")
                print(f"   📏 اندازه: {config['fontsize']}px")
                print(f"   🎨 رنگ: {config['color']}")
                if config['background_color'] != 'none':
                    print(f"   🎨 زمینه: {config['background_color']}")
                print(f"   🔲 حاشیه: {config['outline_width']}px {config['outline_color']}")
                print(f"   📍 موقعیت: {config['position']}")
                
                # ایجاد فایل SRT موقت با encoding صحیح
                temp_srt = temp_dir / "temp_subtitles.srt"
                with open(temp_srt, 'w', encoding='utf-8') as f:
                    f.write(srt_path.read_text(encoding='utf-8'))
                
                # ایجاد ویدیو با زیرنویس
                output_path = self.work_dir / 'custom_subtitled_video.mp4'
                print("🎬 ایجاد ویدیو با زیرنویس...")
                
                # ساخت فیلتر زیرنویس با تنظیمات کامل
                style_parts = [
                    f"FontName={config['font']}",
                    f"FontSize={config['fontsize']}",
                    f"PrimaryColour=&H{self._color_to_hex(config['color'])}",
                    f"OutlineColour=&H{self._color_to_hex(config['outline_color'])}",
                    f"Outline={config['outline_width']}",
                    f"MarginV={config['margin_v']}",
                    f"Shadow={config['shadow']}",
                    f"ShadowColour=&H{self._color_to_hex(config['shadow_color'])}",
                    f"Bold={1 if config['bold'] else 0}",
                    f"Italic={1 if config['italic'] else 0}",
                    f"Alignment={self._get_alignment(config['position'])}"
                ]
                
                # اضافه کردن رنگ زمینه اگر انتخاب شده باشد
                if config['background_color'] != 'none':
                    style_parts.append(f"BackColour=&H{self._color_to_hex(config['background_color'])}")
                    style_parts.append("BorderStyle=4")  # جعبه گرد
                
                subtitle_filter = f"subtitles={temp_srt.absolute()}:force_style='{','.join(style_parts)}'"
                
                subprocess.run([
                    'ffmpeg', '-i', str(video_path),
                    '-vf', subtitle_filter,
                    '-c:v', 'libx264', '-c:a', 'copy',
                    '-y', str(output_path)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                print(f"✅ ویدیو با زیرنویس ایجاد شد: {output_path}")
                return str(output_path)
                
        except Exception as e:
            print(f"❌ خطا در ایجاد ویدیو با زیرنویس: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _color_to_hex(self, color_name: str) -> str:
        """تبدیل نام رنگ به فرمت hex برای FFmpeg"""
        color_map = {
            "white": "ffffff",
            "yellow": "ffff00",
            "red": "ff0000",
            "green": "00ff00",
            "blue": "0000ff",
            "black": "000000",
            "orange": "ffa500",
            "purple": "800080",
            "pink": "ffc0cb",
            "cyan": "00ffff",
            "lime": "00ff00",
            "magenta": "ff00ff",
            "silver": "c0c0c0",
            "gold": "ffd700",
            "gray": "808080",
            "none": "00000000"  # شفاف
        }
        return color_map.get(color_name.lower(), "ffffff")
    
    def _get_alignment(self, position: str) -> int:
        """تبدیل موقعیت به کد alignment برای FFmpeg"""
        # FFmpeg subtitle alignment codes:
        # 1=bottom_left, 2=bottom_center, 3=bottom_right
        # 4=middle_left, 5=middle_center, 6=middle_right  
        # 7=top_left, 8=top_center, 9=top_right
        alignment_map = {
            "top_left": 7,
            "top_center": 8,
            "top_right": 9,
            "middle_left": 4,
            "middle_center": 5,
            "middle_right": 6,
            "bottom_left": 1,
            "bottom_center": 2,
            "bottom_right": 3,
            "top": 8,
            "bottom": 2,
            "center": 5,
            "left": 4,
            "right": 6
        }
        return alignment_map.get(position.lower(), 2)  # پیش‌فرض: پایین وسط
