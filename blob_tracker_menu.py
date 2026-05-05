#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

# Get the directory where the app is located
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
    base_path = sys._MEIPASS
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = APP_DIR

# Find path to blob_tracker_app.py
script_path = os.path.join(base_path, "blob_tracker_app.py")

# Path to embedded ffmpeg (for Windows exe)
if getattr(sys, 'frozen', False) and sys.platform == 'win32':
    FFMPEG_PATH = os.path.join(base_path, "ffmpeg.exe")
else:
    FFMPEG_PATH = "ffmpeg"

OUTPUT_DIR = os.path.join(APP_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors for terminal
class Colors:
    HOT_PINK = '\033[95m'    # Bright magenta / hot pink
    RESET = '\033[0m'

# ASCII Art Whale
WHALE = r"""
                __       __
                .'--.--'.-'
  _,.------.___,   \' r'
  ', '-._a      '-' .'
   '.    '-'Y \_  /
    '--;____'--.'-,
      /..'       '''
"""

CREDIT = "  by cgazou(c) + deepseek"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu():
    """Display menu with whale ASCII art in hot pink"""
    clear_screen()
    
    # Print everything in hot pink
    for line in WHALE.split('\n'):
        print(Colors.HOT_PINK + line + Colors.RESET)
    
    print()
    print(Colors.HOT_PINK + "BLOB TRACKER 4K" + Colors.RESET)
    print(Colors.HOT_PINK + CREDIT + Colors.RESET)
    print(Colors.HOT_PINK + "=" * 50 + Colors.RESET)
    print()
    print(Colors.HOT_PINK + "  Supported formats: MP4 (best), MOV" + Colors.RESET)
    print(Colors.HOT_PINK + "  Output: MP4 video with tracking overlays" + Colors.RESET)
    print()
    print(Colors.HOT_PINK + "  1. Quick export (MP4 video only)" + Colors.RESET)
    print(Colors.HOT_PINK + "  2. Full export (MP4 + PNG sequence with alpha)" + Colors.RESET)
    print(Colors.HOT_PINK + "  3. Exit" + Colors.RESET)
    print()

def check_video_format(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.mp4':
        return True
    elif ext == '.mov':
        print(Colors.HOT_PINK + "\n[INFO] MOV format detected. This should work but MP4 is recommended." + Colors.RESET)
        response = input(Colors.HOT_PINK + "Continue anyway? (Y/n): " + Colors.RESET).strip().lower()
        return response != 'n'
    else:
        print(Colors.HOT_PINK + f"\n[WARNING] Unsupported format: {ext}" + Colors.RESET)
        print(Colors.HOT_PINK + "Recommended format: MP4 (H.264)" + Colors.RESET)
        response = input(Colors.HOT_PINK + "Continue anyway? (y/N): " + Colors.RESET).strip().lower()
        return response == 'y' or response == 'yes'

def check_ffmpeg():
    """Check if ffmpeg is available (embedded or system)"""
    if os.path.exists(FFMPEG_PATH):
        return True
    try:
        subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def convert_to_prores(png_folder, fps=25):
    """Convert PNG sequence to ProRes 4444 using ffmpeg"""
    folder_name = os.path.basename(png_folder.rstrip('/\\'))
    output_mov = os.path.join(os.path.dirname(png_folder), f"{folder_name}_prores.mov")
    
    print(Colors.HOT_PINK + f"\nConverting to ProRes 4444..." + Colors.RESET)
    print(Colors.HOT_PINK + f"Input: {png_folder}/{folder_name}_*.png" + Colors.RESET)
    print(Colors.HOT_PINK + f"Output: {output_mov}" + Colors.RESET)
    print(Colors.HOT_PINK + f"FPS: {fps}" + Colors.RESET)
    print()
    
    cmd = [
        FFMPEG_PATH, "-y",
        "-framerate", str(fps),
        "-i", f"{png_folder}/{folder_name}_%06d.png",
        "-c:v", "prores_ks",
        "-profile:v", "4444",
        "-pix_fmt", "yuva444p10le",
        "-vendor", "apl0",
        "-bits_per_mb", "8000",
        output_mov
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0, output_mov

def main():
    while True:
        show_menu()
        
        export_choice = input(Colors.HOT_PINK + "Your choice (1/2/3): " + Colors.RESET).strip()
        
        if export_choice == '3':
            print(Colors.HOT_PINK + "\nGoodbye!" + Colors.RESET)
            break
        
        if export_choice not in ['1', '2']:
            print(Colors.HOT_PINK + "Invalid choice" + Colors.RESET)
            input(Colors.HOT_PINK + "Press Enter to continue..." + Colors.RESET)
            continue
        
        print()
        video_path = input(Colors.HOT_PINK + "Video file path (drag and drop): " + Colors.RESET).strip().strip('"').strip("'")
        video_path = os.path.expanduser(video_path)
        
        if not video_path:
            print(Colors.HOT_PINK + "No file selected" + Colors.RESET)
            input(Colors.HOT_PINK + "Press Enter to continue..." + Colors.RESET)
            continue
        
        if not os.path.exists(video_path):
            print(Colors.HOT_PINK + f"File not found: {video_path}" + Colors.RESET)
            input(Colors.HOT_PINK + "Press Enter to continue..." + Colors.RESET)
            continue
        
        if not check_video_format(video_path):
            print(Colors.HOT_PINK + "Operation cancelled" + Colors.RESET)
            input(Colors.HOT_PINK + "Press Enter to continue..." + Colors.RESET)
            continue
        
        if export_choice == '2':
            print()
            folder_name = input(Colors.HOT_PINK + "Enter folder name for PNG sequence: " + Colors.RESET).strip()
            if not folder_name:
                folder_name = "frames_output"
            
            output_path = os.path.join(OUTPUT_DIR, folder_name)
            mp4_path = os.path.join(OUTPUT_DIR, f"{folder_name}.mp4")
            
            cmd = [
                sys.executable, script_path,
                "--input", video_path,
                "--output", mp4_path,
                "--png-output", output_path
            ]
            print(Colors.HOT_PINK + f"\nMP4 output: {mp4_path}" + Colors.RESET)
            print(Colors.HOT_PINK + f"PNG output: {output_path}/" + Colors.RESET)
            
        else:
            print()
            mp4_name = input(Colors.HOT_PINK + "Enter MP4 file name (without .mp4): " + Colors.RESET).strip()
            if not mp4_name:
                mp4_name = "output"
            
            output_path = os.path.join(OUTPUT_DIR, f"{mp4_name}.mp4")
            
            cmd = [
                sys.executable, script_path,
                "--input", video_path,
                "--output", output_path
            ]
            print(Colors.HOT_PINK + f"\nMP4 output: {output_path}" + Colors.RESET)
        
        print(Colors.HOT_PINK + "\n" + "=" * 40 + Colors.RESET)
        print(Colors.HOT_PINK + "Processing..." + Colors.RESET)
        print(Colors.HOT_PINK + "=" * 40 + Colors.RESET)
        print()
        
        result = subprocess.run(cmd)
        
        print()
        if result.returncode == 0:
            print(Colors.HOT_PINK + "[SUCCESS] Tracking completed!" + Colors.RESET)
            
            if export_choice == '2':
                ffmpeg_ok = check_ffmpeg()
                
                if ffmpeg_ok:
                    print()
                    print(Colors.HOT_PINK + "=" * 40 + Colors.RESET)
                    convert = input(Colors.HOT_PINK + "Convert PNG sequence to ProRes 4444? (y/N): " + Colors.RESET).strip().lower()
                    if convert == 'y' or convert == 'yes':
                        fps_input = input(Colors.HOT_PINK + "Frame rate (default: 25): " + Colors.RESET).strip()
                        fps = int(fps_input) if fps_input.isdigit() else 25
                        
                        success, prores_path = convert_to_prores(output_path, fps)
                        if success:
                            print(Colors.HOT_PINK + f"\n[SUCCESS] ProRes 4444 created: {prores_path}" + Colors.RESET)
                        else:
                            print(Colors.HOT_PINK + "\n[ERROR] ProRes conversion failed" + Colors.RESET)
                else:
                    print(Colors.HOT_PINK + "\n[INFO] ffmpeg not found." + Colors.RESET)
                    print(Colors.HOT_PINK + "Download: https://ffmpeg.org/download.html" + Colors.RESET)
        else:
            print(Colors.HOT_PINK + "[ERROR] Tracking failed" + Colors.RESET)
        
        input(Colors.HOT_PINK + "\nPress Enter to continue..." + Colors.RESET)

if __name__ == '__main__':
    main()