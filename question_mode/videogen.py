import httpx
import time
import argparse
import sys
import os
import base64
import mimetypes
import cv2
import numpy as np
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False
import re
from gtts import gTTS
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip, concatenate_audioclips, CompositeVideoClip
from moviepy.audio.fx import AudioFadeOut
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Runway API Key
API_KEY = os.getenv("RUNWAY_API_KEY", "")

def select_file_dialog():
    if not HAS_TK:
        print("Error: Tkinter not installed. Cannot use file dialog.")
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    print("Please select your product image...")
    file_path = filedialog.askopenfilename(
        title="Select Product Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def capture_image_from_camera(output_path="captured_product.jpg"):
    print(f"Opening camera... Press 'Space' to take a picture, or 'Esc' to cancel.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key == 32: # Space
            cv2.imwrite(output_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            return output_path
        elif key == 27: # Esc
            break
    cap.release()
    cv2.destroyAllWindows()
    return None

def split_into_sentences(text):
    """Clean and split text into meaningful sentences/taglines."""
    noise = r'Scene\s*\d+|[\d.]+s|Cinematic|4k|8k|shot\s*of|close\s*up|angle|realistic|render'
    clean = re.sub(noise, '', text, flags=re.IGNORECASE).strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 3]
    if not sentences:
        sentences = [s.strip() for s in re.split(r'[,;]\s+', clean) if len(s.strip()) > 3]
    return sentences if sentences else [clean]

def extract_key_feature(sentence):
    """Extract 2-4 punchy words from a sentence for the text overlay."""
    # Common stop words to remove for a cleaner tagline
    stops = r'\b(a|an|the|is|are|was|were|has|have|with|features|delivers|provides|and|of|for|in|on|at|to)\b'
    clean = re.sub(stops, '', sentence, flags=re.IGNORECASE).strip()
    # Remove technical metadata
    noise = r'Cinematic|4k|8k|HD|High\s*Quality|High\s*Res'
    clean = re.sub(noise, '', clean, flags=re.IGNORECASE).strip()
    # Split and take up to 4 meaningful words
    words = [w for w in clean.split() if len(w) > 2]
    if not words: return sentence.upper()[:25] # Fallback
    
    # Try to find a noun-heavy core or just take the first few
    feature = " ".join(words[:4]).upper()
    return feature

def generate_sentence_audio(text, index):
    """Generate audio for a single sentence and return its path and clip."""
    path = f"sentence_{index}.mp3"
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(path)
        clip = AudioFileClip(path)
        return path, clip
    except Exception as e:
        print(f"TTS Error for '{text}': {e}")
        return None, None

def generate_video_segment(prompt_text: str, duration: int = 8, image_path: str = None) -> str:
    # Runway supports duration 10 if we specify 8
    api_duration = 8 if duration >= 7 else (6 if duration >= 5 else 4)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Runway-Version": "2024-11-06"
    }
    
    payload = {
        "model": "gen3a_turbo" if image_path else "veo3.1_fast",
        "ratio": "1280:720",
        "duration": api_duration
    }

    if image_path:
        image_path = image_path.strip('"\'')
        if not os.path.exists(image_path): return None
        mime_type, _ = mimetypes.guess_type(image_path) or ("image/jpeg", None)
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        payload["promptImage"] = f"data:{mime_type};base64,{b64_data}"
        payload["promptText"] = f"{prompt_text[:800]}. Maintain high visual consistency."
        endpoint = "https://api.dev.runwayml.com/v1/image_to_video"
    else:
        payload["promptText"] = prompt_text[:950]
        endpoint = "https://api.dev.runwayml.com/v1/text_to_video"

    try:
        response = httpx.post(endpoint, headers=headers, json=payload, timeout=45.0)
        if response.status_code >= 400: print(f"API Error ({response.status_code}): {response.text}", file=sys.stderr)
        response.raise_for_status()
        task_id = response.json()["id"]
        print(f"Task {task_id} started (Duration: {api_duration}s)...")
        while True:
            status_resp = httpx.get(f"https://api.dev.runwayml.com/v1/tasks/{task_id}", headers=headers)
            status = status_resp.json()
            if status.get("status") == "SUCCEEDED": return status.get("output")[0]
            if status.get("status") == "FAILED":
                print(f"Runway Error: {status.get('error')}")
                return None
            print(".", end="", flush=True)
            time.sleep(5)
    except Exception as e:
        print(f"Error during video segment generation: {e}", file=sys.stderr)
        return None

def download_video(url, filename):
    with httpx.stream("GET", url) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_bytes(): f.write(chunk)

def create_text_overlay_clip(text, duration, start_time, video_w, video_h):
    """Creates a punchy key-feature text overlay as a MoviePy clip."""
    def make_banner():
        # High-Impact Settings for Key Features
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = video_w / 1000.0 * 1.3  # Larger scale for features
        thick = max(2, int(scale * 3.5)) # Thicker
        
        # Key features are usually short, but we still wrap just in case
        words = text.split()
        lines, cur = [], ""
        for ow in words:
            test = (cur + " " + ow).strip()
            (tw, _), _ = cv2.getTextSize(test, font, scale, thick)
            if tw < video_w - 120: cur = test
            else: 
                if cur: lines.append(cur)
                cur = ow
        if cur: lines.append(cur)
        
        line_data = []
        total_h = 0
        for line in lines:
            (tw, th), base = cv2.getTextSize(line, font, scale, thick)
            line_data.append((line, tw, th, base))
            total_h += th + 40
            
        bh = total_h + 60
        frame = np.zeros((bh, video_w, 3), dtype=np.uint8) + 5 # Nearly black
        
        draw_y = 40
        for line, tw, th, base in line_data:
            tx = (video_w - tw) // 2
            draw_y += th
            # Cinematic Shadow
            cv2.putText(frame, line, (tx + 4, draw_y + 4), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
            # Pure White Feature Text
            cv2.putText(frame, line, (tx, draw_y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            draw_y += 40
        return frame

    banner = make_banner()
    banner_rgb = cv2.cvtColor(banner, cv2.COLOR_BGR2RGB)
    
    from moviepy import ImageClip
    clip = ImageClip(banner_rgb, duration=duration)
    clip = clip.with_start(start_time).with_position(("center", "bottom"))
    clip = clip.with_opacity(0.85)
    return clip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, nargs="?")
    parser.add_argument("--images", type=str, nargs="+")
    parser.add_argument("--duration", type=int, default=8)
    args = parser.parse_args()

    images = []
    if args.images: images.extend(args.images)
    
    if not args.prompt and not images:
        num_str = input("Select product photo? [Y/N]: ").strip().upper()
        if num_str == 'Y':
            choice = input(f"[F]ile/[C]amera: ").upper()
            path = select_file_dialog() if choice == 'F' else capture_image_from_camera()
            if path: images.append(path)
        
        args.prompt = input("Enter Ad Prompt: ").strip()

    if not args.prompt: 
        print("Error: Prompt is required.")
        sys.exit(1)

    print(f"\n--- Generating Unified 8s Video ---")
    img_path = images[0] if images else None
    video_url = generate_video_segment(args.prompt, args.duration, img_path)
    
    if not video_url:
        print("\nCRITICAL: Video generation failed. Check API credits or connection.", file=sys.stderr)
        sys.exit(1)

    video_fname = "raw_video.mp4"
    download_video(video_url, video_fname)

    print("\n--- Extracting Key Features & Syncing Audio ---")
    try:
        video_clip = VideoFileClip(video_fname)
        w, h = video_clip.size
        
        sentences = split_into_sentences(args.prompt)
        print(f"Processing {len(sentences)} key features.")
        
        audio_clips = []
        overlay_clips = []
        current_time = 0.0
        
        for i, sentence in enumerate(sentences):
            # 1. Generate full sentence audio for VO
            a_path, a_clip = generate_sentence_audio(sentence, i)
            if a_clip:
                duration = a_clip.duration
                
                # 2. Extract punchy Key Feature for text overlay
                key_feature = extract_key_feature(sentence)
                print(f"VO: '{sentence[:40]}...' -> UI: '{key_feature}'")
                
                # 3. Create timed text overlay
                t_clip = create_text_overlay_clip(key_feature, duration, current_time, w, h)
                overlay_clips.append(t_clip)
                
                a_clip = a_clip.with_start(current_time)
                audio_clips.append(a_clip)
                
                current_time += duration
                if current_time >= video_clip.duration: break

        final_audio = concatenate_audioclips(audio_clips) if audio_clips else None
        if final_audio:
            if final_audio.duration > video_clip.duration:
                final_audio = final_audio.subclipped(0, video_clip.duration)
            final_audio = final_audio.with_effects([AudioFadeOut(0.5)])
        
        final_video = CompositeVideoClip([video_clip] + overlay_clips)
        if final_audio:
            final_video = final_video.with_audio(final_audio)

        print("\nExporting Key-Feature Synchronized Ad...")
        final_video.write_videofile("final_ad_film.mp4", codec="libx264", audio_codec="aac", bitrate="6000k", fps=24)
        
        video_clip.close()
        for c in audio_clips: c.close()
        final_video.close()
        
        for i in range(len(sentences)):
            temp_p = f"sentence_{i}.mp3"
            if os.path.exists(temp_p): os.remove(temp_p)
            
        print(f"\nSUCCESS! Key-Feature Ad saved as final_ad_film.mp4")
    except Exception as e:
        print(f"Final error: {e}")
        import traceback
        traceback.print_exc()
