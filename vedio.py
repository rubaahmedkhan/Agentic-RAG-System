import yt_dlp

def download_youtube_video(url, output_path="downloaded_video.mp4"):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': output_path
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Example:
download_youtube_video("https://www.youtube.com/watch?v=-pqzyvRp3Tc&t=31s")
