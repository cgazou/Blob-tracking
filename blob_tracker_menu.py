#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

# Get the directory where the app is located
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Find path to blob_tracker_app.py
if getattr(sys, 'frozen', False):
    # Running as exe - files are in _MEIPASS
    base_path = sys._MEIPASS
else:
    # Running as script
    base_path = os.path.dirname(os.path.abspath(__file__))

script_path = os.path.join(base_path, "blob_tracker_app.py")

OUTPUT_DIR = os.path.join(APP_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("=" * 50)
    print("     BLOB TRACKER 4K")
    print("=" * 50)
    print()
    print("  1. Quick export (MP4 video only)")
    print("  2. Full export (MP4 + PNG sequence with alpha)")
    print()
    
    export_choice = input("Your choice (1/2): ").strip()
    
    print()
    video_path = input("Video file path (drag and drop): ").strip().strip('"')
    
    if not video_path:
        print("No file selected")
        input("Press Enter to quit...")
        return
    
    if export_choice == '2':
        folder_name = input("Enter folder name for PNG sequence: ").strip()
        if not folder_name:
            folder_name = "frames_output"
        
        output_path = os.path.join(OUTPUT_DIR, folder_name)
        mp4_path = os.path.join(OUTPUT_DIR, f"{folder_name}.mp4")
        
        cmd = [
            "python", script_path,
            "--input", video_path,
            "--output", mp4_path,
            "--png-output", output_path
        ]
    else:
        mp4_name = input("Enter MP4 file name (without .mp4): ").strip()
        if not mp4_name:
            mp4_name = "output"
        
        output_path = os.path.join(OUTPUT_DIR, f"{mp4_name}.mp4")
        
        cmd = [
            "python", script_path,
            "--input", video_path,
            "--output", output_path
        ]
    
    print("\n" + "=" * 40)
    print("Processing...")
    print("=" * 40)
    print()
    
    result = subprocess.run(cmd)
    
    print()
    if result.returncode == 0:
        print("[SUCCESS] Tracking completed!")
    else:
        print("[ERROR] Tracking failed")
    
    input("\nPress Enter to exit...")

if __name__ == '__main__':
    main()