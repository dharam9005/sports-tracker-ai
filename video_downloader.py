"""
Video downloader utility using yt-dlp.
Downloads public videos from YouTube or other sources.
"""

import subprocess
import os


def download_video(url, output_path="input_video.mp4", max_duration=120):
    """
    Download a video from a public URL.

    Args:
        url: Public video URL (YouTube, etc.)
        output_path: Where to save the downloaded video
        max_duration: Maximum duration in seconds to download

    Returns:
        Path to the downloaded video
    """
    try:
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",
            "--download-sections", f"*0-{max_duration}",
            "-o", output_path,
            "--no-playlist",
            "--quiet",
            url,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError:
        # Fallback: try without section download
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",
            "-o", output_path,
            "--no-playlist",
            "--quiet",
            url,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path


def get_video_info(url):
    """Get video metadata without downloading."""
    try:
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        import json
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}
