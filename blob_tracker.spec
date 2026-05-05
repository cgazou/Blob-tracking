# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['blob_tracker_menu.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('blob_tracker_app.py', '.'),
        ('config.txt', '.') if __import__('os').path.exists('config.txt') else (),
    ],
    hiddenimports=['numpy', 'cv2', 're', 'subprocess', 'collections'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.purities)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BlobTracker4K',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # True = fenetre console visible, False = invisible
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)