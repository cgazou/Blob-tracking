#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blob Tracker 4K - Simple Menu
"""

import os
import sys
import subprocess

# Get the directory where the app is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "output")

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[INFO] Created output directory: output/")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_config():
    """Display current configuration"""
    config_path = os.path.join(APP_DIR, 'config.txt')
    if not os.path.exists(config_path):
        print("[ERROR] config.txt not found!")
        return False
    
    clear_screen()
    print("=" * 50)
    print("     BLOB TRACKER 4K - CONFIGURATION")
    print("=" * 50)
    print()
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                print(f"  {line}")
    
    print()
    print("=" * 50)
    return True

def main():
    print("\n" + "=" * 50)
    print("        IMPORTANT - VIDEO REQUIREMENT")
    print("=" * 50)
    print("\n  This tracker is designed for 4K video ONLY.")
    print("  Input resolution must be 3840x2160 or higher.")
    print("  Lower resolutions may not work properly.\n")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 50)
    input("Press Enter to continue...")
    
    if not show_config():
        input("Press Enter to quit...")
        return
    
    print()
    print("Is this configuration correct?")
    print("  [Y]es - Launch tracking")
    print("  [N]o - Quit")
    print()
    
    choice = input("Your choice (Y/N): ").strip().lower()
    
    if choice == 'y' or choice == 'yes':
        clear_screen()
        print("=" * 50)
        print("        EXPORT OPTIONS")
        print("=" * 50)
        print()
        print("  1. Quick export (MP4 video only)")
        print("  2. Full export (MP4 + PNG sequence with alpha)")
        print()
        
        export_choice = input("Your choice (1/2): ").strip()
        
        # Get video file
        print()
        video_path = input("Video file path (drag and drop): ").strip().strip('"')
        
        if not video_path:
            print("No file selected")
            input("Press Enter to quit...")
            return
        
        # Build command - files go into output/ folder
        if export_choice == '2':
            # PNG + MP4 export
            print()
            print("Output will be saved in: output/")
            folder_name = input("Enter folder name for PNG sequence: ").strip()
            if not folder_name:
                folder_name = "frames_output"
            
            # Full path inside output folder
            output_path = os.path.join(OUTPUT_DIR, folder_name)
            
            # Also create MP4 with same name
            mp4_name = f"{folder_name}.mp4"
            mp4_path = os.path.join(OUTPUT_DIR, mp4_name)
            
            cmd = [
                sys.executable, os.path.join(APP_DIR, "blob_tracker_app.py"),
                "--input", video_path,
                "--output", mp4_path,
                "--png-output", output_path
            ]
            print(f"\nMP4 output: {mp4_path}")
            print(f"PNG output: {output_path}/")
        else:
            # MP4 only
            print()
            print("Output will be saved in: output/")
            mp4_name = input("Enter MP4 file name (without .mp4): ").strip()
            if not mp4_name:
                mp4_name = "output"
            if not mp4_name.endswith('.mp4'):
                mp4_name += '.mp4'
            
            output_path = os.path.join(OUTPUT_DIR, mp4_name)
            
            cmd = [
                sys.executable, os.path.join(APP_DIR, "blob_tracker_app.py"),
                "--input", video_path,
                "--output", output_path
            ]
            print(f"\nMP4 output: {output_path}")
        
        print("\nCommand: " + " ".join(cmd))
        print("\nProcessing...\n")
        
        result = subprocess.run(cmd)
        
        print("\n" + "=" * 40)
        if result.returncode == 0:
            print("[SUCCESS] Tracking completed!")
            print(f"Files saved in: {OUTPUT_DIR}")
        else:
            print("[ERROR] Tracking failed")
        
        input("\nPress Enter to quit...")
    
    else:
        print("\nGoodbye!")

if __name__ == '__main__':
    main()