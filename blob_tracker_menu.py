#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

# Python command (works both in script and exe)
if getattr(sys, 'frozen', False):
    PYTHON_CMD = "python"
else:
    PYTHON_CMD = sys.executable

# Get the directory where the app is located
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
    base_path = sys._MEIPASS
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = APP_DIR

# Find path to blob_tracker_app.py
script_path = os.path.join(base_path, "blob_tracker_app.py")

OUTPUT_DIR = os.path.join(APP_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors for terminal
class Colors:
    HOT_PINK = '\033[95m'
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
    clear_screen()
    
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
                PYTHON_CMD, script_path,
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
                PYTHON_CMD, script_path,
                "--input", video_path,
                "--output", output_path
            ]
            print(Colors.HOT_PINK + f"\nMP4 output: {output_path}" + Colors.RESET)
        
        print(Colors.HOT_PINK + "\n" + "=" * 40 + Colors.RESET)
        print(Colors.HOT_PINK + "Processing..." + Colors.RESET)
        print(Colors.HOT_PINK + "=" * 40 + Colors.RESET)
        print()
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(Colors.HOT_PINK + "\n[SUCCESS] Tracking completed!" + Colors.RESET)
            print(Colors.HOT_PINK + f"\nFiles saved in: {OUTPUT_DIR}" + Colors.RESET)
        else:
            print(Colors.HOT_PINK + "\n[ERROR] Tracking failed" + Colors.RESET)
        
        input(Colors.HOT_PINK + "\nPress Enter to continue..." + Colors.RESET)

if __name__ == '__main__':
    main()