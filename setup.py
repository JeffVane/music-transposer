from cx_Freeze import setup, Executable
import os

# Inclui as DLLs necessárias do ffmpeg e o cache
include_files = [
    ("bin", "bin"),  # ffmpeg/ffprobe
    "cache_tonalidades.json"
]

# Se tiver ícone
base = "Win32GUI"  # Oculta terminal
icon = "icon.ico"  # Substitua por seu ícone

executables = [
    Executable("interface.py", base=base, icon=icon, target_name="Transpositor.exe")
]

build_exe_options = {
    "packages": ["os", "sys", "librosa", "sounddevice", "soundfile", "numpy", "PyQt6"],
    "include_files": include_files,
    "include_msvcr": True,
    "excludes": ["tkinter"],
}

setup(
    name="Music Transposer Pro",
    version="1.0",
    description="Aplicativo para transposição musical com player integrado",
    options={"build_exe": build_exe_options},
    executables=executables
)
