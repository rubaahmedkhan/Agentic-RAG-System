import os
from faster_whisper import WhisperModel

#video_path = r"D:\RAG\ragwebsite\downloaded_video.mp4"
video_path = r"D:\RAG\ragwebsite\downloaded_video.mp4"

if not os.path.exists(video_path):
    print("❌ File not found!")
else:
    print("✅ File found. Transcribing...")

    model = WhisperModel("base", compute_type="int8")  # Fast on CPU

    segments, info = model.transcribe(video_path)

    full_text = ""
    for segment in segments:
        full_text += segment.text + " "

    print("📄 Transcript:\n", full_text)


