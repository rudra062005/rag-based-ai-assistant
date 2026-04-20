import os
import subprocess

# folders
video_folder = "videos"
audio_folder = "audios"

# create audios folder if not exists
os.makedirs(audio_folder, exist_ok=True)

# loop through files
for file in os.listdir(video_folder):
    if file.endswith(".mp4"):
        video_path = os.path.join(video_folder, file)
        
        # change extension to .mp3
        audio_name = os.path.splitext(file)[0] + ".mp3"
        audio_path = os.path.join(audio_folder, audio_name)

        # ffmpeg command
        command = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",   # best audio quality
            "-map", "a",
            audio_path
        ]

        print(f"Converting {file} → {audio_name}")
        subprocess.run(command)

print("All files converted!")