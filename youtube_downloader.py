# youtube_downloader.py
import os
import subprocess
import traceback

def validar_url_youtube(url):
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        return any(site in parsed.netloc for site in ["youtube.com", "youtu.be"])
    except:
        return False


def limpar_cache_yt_dlp():
    try:
        subprocess.run(["yt-dlp", "--rm-cache-dir"], text=True, encoding="utf-8")
        return True
    except Exception as e:
        print(f"[YT-DLP] Falha ao limpar cache: {e}")
        return False


def obter_titulo_e_url_real(url):
    comando_info = [
        "yt-dlp", "--no-playlist", "--skip-download",
        "--print", "%(title)s", "--print", "%(webpage_url)s",
        url
    ]
    info = subprocess.run(comando_info, capture_output=True, text=True, encoding="utf-8", timeout=15)
    if info.returncode != 0:
        raise RuntimeError(f"yt-dlp falhou ao obter título:\n{info.stderr}")
    linhas = info.stdout.strip().splitlines()
    if len(linhas) < 2:
        raise ValueError("Saída inesperada ao obter informações do vídeo.")
    return linhas[0], linhas[1]


def baixar_audio_youtube(url, ffmpeg_dir=None, nome_fixo="temp_download.mp3"):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_path = os.path.join(ffmpeg_dir or "", "ffmpeg.exe")
        ffprobe_path = os.path.join(ffmpeg_dir or "", "ffprobe.exe")

        if not os.path.isfile(ffmpeg_path) or not os.path.isfile(ffprobe_path):
            raise FileNotFoundError("ffmpeg.exe ou ffprobe.exe não encontrados.")

        # 1ª tentativa com --print filename
        comando_print = [
            "yt-dlp",
            "--ffmpeg-location", ffmpeg_dir or "",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "mp3",
            "--format", "bestaudio[ext=m4a]/bestaudio",
            "--print", "filename",
            url
        ]
        processo = subprocess.run(comando_print, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", timeout=120)
        nome_arquivo = processo.stdout.strip()
        if processo.returncode == 0 and nome_arquivo and os.path.isfile(nome_arquivo):
            return nome_arquivo

        # 2ª tentativa: nome fixo
        saida_temp = os.path.join(base_dir, nome_fixo)
        comando_fallback = [
            "yt-dlp",
            "--ffmpeg-location", ffmpeg_dir or "",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "mp3",
            "--format", "bestaudio[ext=m4a]/bestaudio",
            "--output", saida_temp,
            url
        ]
        processo2 = subprocess.run(comando_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", timeout=120)

        if not os.path.isfile(saida_temp):
            raise FileNotFoundError("Arquivo não encontrado após o download (fallback).")

        return saida_temp

    except Exception as e:
        raise RuntimeError(f"Erro ao baixar áudio:\n{traceback.format_exc()}")
