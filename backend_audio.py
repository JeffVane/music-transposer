import threading
import queue
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox,QDialog,QLabel,QLineEdit,QHBoxLayout
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
import traceback
from PyQt6.QtWidgets import QInputDialog
import os
from urllib.parse import urlparse
import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QVBoxLayout
from components import RoundedButton, COLORS  # se também usar aqui
import subprocess
from auto_processador import detectar_tonalidade_aprimorado
import unicodedata
import re


def calcular_relativa(nota_idx, tipo_escala):
    """Retorna a relativa menor ou maior"""
    if tipo_escala == "Maior":
        relativa_idx = (nota_idx - 3) % 12  # 3 semitons abaixo
        tipo_relativa = "menor"
    else:
        relativa_idx = (nota_idx + 3) % 12  # 3 semitons acima
        tipo_relativa = "Maior"

    return f"{notas[relativa_idx]} {tipo_relativa}"


def normalizar_nome_musica(titulo):
    # 1. Normaliza usando NFC para preservar acentos e símbolos visuais
    titulo = unicodedata.normalize("NFC", titulo)

    # 2. Remove apenas os caracteres realmente proibidos
    titulo = re.sub(r'[<>:"/\\|?*\n\r\t]', '', titulo)

    # 3. Remove espaços duplos
    titulo = re.sub(r'\s+', ' ', titulo).strip()

    return titulo

# Notas musicais
notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

COLORS = {
    "primary": "#1DB954",  # Verde Spotify
    "secondary": "#4A90E2",  # Azul suave
    "danger": "#FF5252",  # Vermelho vibrante
    "background": "#121212",  # Fundo escuro
    "card_bg": "#212121",  # Fundo de card
    "text_primary": "#FFFFFF",  # Texto principal
    "text_secondary": "#B3B3B3",  # Texto secundário
    "slider_groove": "#535353",  # Trilho do slider
    "slider_handle": "#1DB954",  # Manipulador do slider
    "progress_bg": "#535353",  # Fundo da barra de progresso
    "accent": "#E53935",  # Roxo para destaque
    "hover": "#333333",  # Cor para hover
}


class TranspositorBase(QWidget):
    def __init__(self):
        super().__init__()
        self.tom_em_reproducao = None
        self.iniciar_processamento_em_background()
        self.limite_transposicao = 6  # pode ser alterado para 5, 7 etc.

    def pitch_shift_resample(self, y, semitons):
        """Muda o pitch por resample, alterando a duração"""
        rate = 2.0 ** (semitons / 12.0)
        return librosa.resample(y, orig_sr=self.sr, target_sr=int(self.sr * rate))

    def iniciar_worker_thread(self):
        """Inicia uma thread de trabalho para processar a fila de tons"""

        def worker():
            while True:
                try:
                    semitons, inicial, progresso_idx = self.ton_queue.get()
                    self.processando_tons = True
                    self.loading_label.setText(f"Processando tom {semitons:+}...")

                    if semitons not in self.audio_transposto:
                        # Aplica pitch shift: resample para tons muito baixos ou altos
                        if abs(semitons) <= 3:
                            y_mod = librosa.effects.pitch_shift(self.y_original, sr=self.sr, n_steps=semitons)
                        else:
                            y_mod = self.pitch_shift_resample(self.y_original, semitons)

                        self.audio_transposto[semitons] = y_mod

                    if inicial:
                        self.progress_bar.setValue(progresso_idx)
                        if progresso_idx == 3:
                            self.progress_bar.setVisible(False)
                            self.label_status.setText("Música pronta para reprodução!")
                            self.botao_play.setEnabled(True)
                            self.botao_subir.setEnabled(True)
                            self.botao_descer.setEnabled(True)
                            self.botao_exportar.setEnabled(True)

                    self.loading_label.setText("")
                    self.processando_tons = False
                    self.botao_play.setEnabled(True)
                    self.ton_queue.task_done()

                except Exception as e:
                    QMessageBox.critical(self, "Erro ao processar tom", str(e))
                    self.loading_label.setText("")
                    self.ton_queue.task_done()
                    self.processando_tons = False

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def carregar_musica(self):
        caminho, _ = QFileDialog.getOpenFileName(self, "Selecionar música", "", "Áudio (*.mp3 *.wav)")
        if not caminho:
            return

        self.botao_carregar.setEnabled(False)
        self.label_status.setText("Carregando música...")

        # Inicia animação da barra de progresso (modo indeterminado)
        self._progress_anim_value = 0
        self._progress_anim_direction = 1
        self.progress_bar.setMaximum(0)  # Modo indeterminado
        self.progress_bar.setVisible(True)
        self.progress_animation_timer.start()

        # Thread para carregar o áudio
        def carregar_arquivo():
            try:
                y, sr = librosa.load(caminho, sr=None, mono=True)
                return y, sr
            except Exception as e:
                return None, str(e)

        def on_load_finished(result):
            # Para animação da barra
            self.progress_animation_timer.stop()
            self.progress_bar.setMaximum(3)  # Prepara para transposição inicial
            self.progress_bar.setValue(0)

            if result is None or isinstance(result[1], str):
                QMessageBox.critical(self, "Erro ao carregar arquivo", result[1])
                self.botao_carregar.setEnabled(True)
                self.progress_bar.setVisible(False)
                self.label_status.setText("Erro ao carregar música")
                return

            y, sr = result
            self.y_original = y
            self.sr = sr
            self.tom_em_reproducao = None
            self.ton_atual = 0
            self.reproduzindo = False
            self.posicao_amostral = 0

            # Detecta a tonalidade
            self.loading_label.setText("Detectando tonalidade...")
            worker = AudioWorker(self.detectar_tonalidade_thread, y, sr)
            worker.signals.finished.connect(self.on_tonalidade_detectada)
            worker.signals.error.connect(self.on_error)
            worker.start()

        # Inicia o worker
        worker = AudioWorker(carregar_arquivo)
        worker.signals.finished.connect(on_load_finished)
        worker.signals.error.connect(self.on_error)
        worker.start()

    def carregar_arquivo_local(self, caminho):
        if self.carregar_tons_processados(caminho):
            return  # tons já estão carregados

        self.botao_carregar.setEnabled(False)
        self.label_status.setText("Carregando música...")

        self._progress_anim_value = 0
        self._progress_anim_direction = 1
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(True)
        self.progress_animation_timer.start()


        def carregar():
            try:
                y, sr = librosa.load(caminho, sr=None, mono=True)
                return y, sr
            except Exception as e:
                return None, str(e)

        def ao_final(resultado):
            try:
                self.progress_bar.setVisible(False)
                self.loading_label.setText("")

                if not isinstance(resultado, str):
                    raise ValueError(f"Tipo de retorno inesperado: {type(resultado)}")

                if not os.path.isfile(resultado):
                    raise FileNotFoundError(f"Arquivo baixado não encontrado: {resultado}")

                # ✅ Sugere o nome da música com base no nome do arquivo baixado
                nome_sugerido = os.path.splitext(os.path.basename(resultado))[0]

                dialog = SalvarMusicaDialog(nome_sugerido)  # passa o nome para o diálogo
                # Criar novo diálogo para salvar a música
                salvar_dialog = SalvarMusicaDialog(nome_sugerido=titulo_limpo)
                if salvar_dialog.exec() != QDialog.DialogCode.Accepted:
                    return

                nome, pasta = salvar_dialog.get_dados()

                if not nome or not pasta:
                    QMessageBox.warning(self, "Nome inválido", "O nome ou pasta não foi especificado corretamente.")
                    return

                destino = os.path.join(pasta, f"{nome}.mp3")
                if os.path.exists(destino):
                    os.remove(destino)

                os.rename(resultado, destino)
                self.label_status.setText("Música salva com sucesso!")

                # ✅ Tenta carregar tons prontos, ou inicia detecção do tom
                if self.carregar_arquivo_local(destino):
                    print("[DEBUG] Tons pré-processados carregados com sucesso.")
                else:
                    print("[DEBUG] Carregando manualmente para detectar o tom...")
                    self.botao_carregar.setEnabled(False)
                    self.label_status.setText("Carregando música...")

                    def carregar():
                        try:
                            y, sr = librosa.load(destino, sr=None, mono=True)
                            return y, sr
                        except Exception as e:
                            return None, str(e)

                    def ao_final_manual(result):
                        self.progress_bar.setVisible(False)
                        if result is None or isinstance(result[1], str):
                            QMessageBox.critical(self, "Erro ao carregar", result[1])
                            self.label_status.setText("Erro ao carregar")
                            self.botao_carregar.setEnabled(True)
                            return

                        self.y_original, self.sr = result
                        self.tom_em_reproducao = None  # <--- ESSENCIAL
                        self.ton_atual = 0
                        self.reproduzindo = False
                        self.posicao_amostral = 0

                        self.label_status.setText("Detectando tom...")
                        worker = AudioWorker(self.detectar_tonalidade_thread, self.y_original, self.sr)
                        worker.signals.finished.connect(self.on_tonalidade_detectada)
                        worker.signals.error.connect(self.on_error)
                        worker.start()

                    worker = AudioWorker(carregar)
                    worker.signals.finished.connect(ao_final_manual)
                    worker.signals.error.connect(self.on_error)
                    worker.start()

                self.listar_musicas_local()

            except Exception as e:
                erro_str = f"[ERRO ao finalizar download]\n{traceback.format_exc()}"
                print(erro_str)
                QMessageBox.critical(self, "Erro", erro_str)

    @staticmethod
    def validar_url_youtube(url):
        try:
            parsed = urlparse(url)
            return any(site in parsed.netloc for site in ["youtube.com", "youtu.be"])
        except:
            return False



    def baixar_youtube(self):
        try:
            # Verificação básica do yt-dlp
            check = subprocess.run(["yt-dlp", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   timeout=3)
            if check.returncode != 0:
                QMessageBox.warning(self, "yt-dlp não disponível", "O yt-dlp não está funcionando corretamente.")
                return

            # Diálogo para URL
            dialog = YoutubeDialog()
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            url = dialog.get_url().strip()
            if not url or not self.validar_url_youtube(url):
                QMessageBox.warning(self, "URL inválida", "A URL informada não é válida para o YouTube.")
                return

            self.label_status.setText("Obtendo informações do vídeo...")
            self.loading_label.setText("Analisando link...")

            # Consulta título e link real
            comando_info = [
                "yt-dlp", "--no-playlist", "--skip-download",
                "--print", "%(title)s", "--print", "%(webpage_url)s",
                url
            ]
            info = subprocess.run(comando_info, capture_output=True, text=True, encoding="utf-8", timeout=15)


            if info.returncode != 0:
                QMessageBox.critical(self, "Erro", f"Não foi possível obter as informações do vídeo.")
                return

            if info.returncode != 0 or not info.stdout:
                QMessageBox.critical(self, "Erro", "Não foi possível obter as informações do vídeo (yt-dlp falhou).")
                print("[YT-DLP STDERR]:", info.stderr)
                return

            linhas = info.stdout.strip().splitlines()
            if len(linhas) < 2:
                QMessageBox.critical(self, "Erro", "Saída inesperada ao obter informações do vídeo.")
                print("[YT-DLP RAW STDOUT]:", info.stdout)
                return

            titulo, url_real = linhas

            titulo_limpo = normalizar_nome_musica(titulo)

            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Confirmar Download")
            msg_box.setText(f"O vídeo detectado foi:\n\n🎵 {titulo_limpo}\n\nDeseja continuar?")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            # Estilo com texto branco e fundo escuro
            msg_box.setStyleSheet("""
                QWidget {
                    background-color: #2a2a2a;
                    color: white;
                    font-size: 12pt;
                }
                QLabel {
                    color: white;
                }
                QPushButton {
                    background-color: #444;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
            """)

            confirm = msg_box.exec()

            if confirm != QMessageBox.StandardButton.Yes:
                return

            # Limpa o cache do yt-dlp para evitar bugs com nomes, redirecionamentos ou arquivos antigos
            try:
                subprocess.run(["yt-dlp", "--rm-cache-dir"], text=True, encoding="utf-8")
                print("[YT-DLP] Cache limpo com sucesso.")
            except Exception as e:
                print(f"[YT-DLP] Falha ao limpar cache: {e}")

            self.label_status.setText("Baixando do YouTube...")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            def baixar_audio():
                try:
                    # Caminho da pasta do script + bin
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    ffmpeg_dir = os.path.join(base_dir, "bin")

                    # Verificar se ffmpeg e ffprobe existem
                    ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")
                    ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe.exe")

                    if not os.path.isfile(ffmpeg_path) or not os.path.isfile(ffprobe_path):
                        raise FileNotFoundError("ffmpeg.exe ou ffprobe.exe não encontrados na pasta 'bin'.")

                    print("[DEBUG] Usando ffmpeg de:", ffmpeg_dir)

                    # 1ª tentativa: usar --print filename para capturar nome final
                    comando_print = [
                        "yt-dlp",
                        "--ffmpeg-location", ffmpeg_dir,
                        "--no-playlist",
                        "--extract-audio",
                        "--audio-format", "mp3",
                        "--format", "bestaudio[ext=m4a]/bestaudio",
                        "--print", "filename",
                        url_real
                    ]

                    processo = subprocess.run(
                        comando_print,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        timeout=120
                    )

                    stdout = processo.stdout.strip()
                    stderr = processo.stderr.strip()
                    print("[YT-DLP STDOUT]:", stdout)
                    print("[YT-DLP STDERR]:", stderr)
                    print("[YT-DLP Retcode]:", processo.returncode)

                    nome_arquivo = stdout

                    # Verifica se o nome retornado é um caminho de arquivo válido
                    if processo.returncode == 0 and nome_arquivo and os.path.isfile(nome_arquivo):
                        print("[DEBUG] Arquivo detectado com sucesso:", nome_arquivo)
                        return nome_arquivo

                    # 2ª tentativa: nome fixo como fallback
                    print("[DEBUG] Usando fallback: nome fixo 'temp_download.mp3'")
                    saida_temp = os.path.join(base_dir, "temp_download.mp3")
                    comando_fallback = [
                        "yt-dlp",
                        "--ffmpeg-location", ffmpeg_dir,
                        "--no-playlist",
                        "--extract-audio",
                        "--audio-format", "mp3",
                        "--format", "bestaudio[ext=m4a]/bestaudio",
                        "--output", saida_temp,
                        url_real
                    ]

                    processo2 = subprocess.run(
                        comando_fallback,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        timeout=120
                    )

                    print("[YT-DLP FALLBACK STDOUT]:", processo2.stdout.strip())
                    print("[YT-DLP FALLBACK STDERR]:", processo2.stderr.strip())

                    if not os.path.isfile(saida_temp):
                        raise FileNotFoundError("Arquivo não encontrado após o download (fallback).")

                    return saida_temp

                except Exception as e:
                    raise RuntimeError(f"Erro ao baixar áudio:\n{traceback.format_exc()}")

            def ao_final(resultado):
                self.progress_bar.setVisible(False)
                self.loading_label.setText("")

                if not isinstance(resultado, str):
                    QMessageBox.critical(self, "Erro", f"Falha ao baixar: tipo inesperado ({type(resultado)})")
                    return

                if not os.path.isfile(resultado):
                    QMessageBox.critical(self, "Erro", f"Arquivo baixado não encontrado:\n{resultado}")
                    return

                # Criar e exibir o diálogo para salvar a música com o nome sugerido
                salvar_dialog = SalvarMusicaDialog(nome_sugerido=titulo_limpo)
                if salvar_dialog.exec() != QDialog.DialogCode.Accepted:
                    return

                nome, pasta = salvar_dialog.get_dados()
                if not nome or not pasta:
                    QMessageBox.warning(self, "Nome inválido", "O nome ou pasta não foi especificado corretamente.")
                    return

                destino = os.path.join(pasta, f"{nome}.mp3")
                if os.path.exists(destino):
                    os.remove(destino)

                os.rename(resultado, destino)
                self.label_status.setText("Música salva com sucesso!")

                if self.carregar_arquivo_local(destino):
                    print("[DEBUG] Tons pré-processados carregados com sucesso.")
                else:
                    print("[DEBUG] Carregando áudio manualmente para detectar tom...")
                    self.botao_carregar.setEnabled(False)
                    self.label_status.setText("Carregando música...")

                    def carregar():
                        try:
                            y, sr = librosa.load(destino, sr=None, mono=True)
                            return y, sr
                        except Exception as e:
                            return None, str(e)

                    def ao_final_manual(result):
                        self.progress_bar.setVisible(False)
                        if result is None or isinstance(result[1], str):
                            QMessageBox.critical(self, "Erro ao carregar", result[1])
                            self.label_status.setText("Erro ao carregar")
                            self.botao_carregar.setEnabled(True)
                            return

                        self.y_original, self.sr = result
                        self.tom_em_reproducao = None  # <--- ESSENCIAL
                        self.ton_atual = 0
                        self.reproduzindo = False
                        self.posicao_amostral = 0

                        self.label_status.setText("Detectando tom...")
                        worker = AudioWorker(self.detectar_tonalidade_thread, self.y_original, self.sr)
                        worker.signals.finished.connect(self.on_tonalidade_detectada)
                        worker.signals.error.connect(self.on_error)
                        worker.start()

                    worker = AudioWorker(carregar)
                    worker.signals.finished.connect(ao_final_manual)
                    worker.signals.error.connect(self.on_error)
                    worker.start()

                # Mostra a música na lista
                self.listar_musicas_local()

            worker = AudioWorker(baixar_audio)
            worker.signals.finished.connect(ao_final)
            worker.signals.error.connect(self.on_error)
            worker.start()

        except Exception as e:
            erro = f"Erro inesperado: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Erro crítico", erro)
            self.progress_bar.setVisible(False)
            self.loading_label.setText("")

    # Verifique se esta classe já possui o signal 'status' em WorkerSignals
    # Se não tiver, adicione essa linha à classe WorkerSignals:
    # status = pyqtSignal(str)

    def listar_musicas_local(self):
        """Lista os arquivos de áudio no diretório atual"""
        try:
            musicas = [f for f in os.listdir() if f.lower().endswith((".mp3", ".wav"))]
            if not musicas:
                QMessageBox.information(self, "Nenhuma música encontrada",
                                        "Nenhuma música foi encontrada na pasta atual.")
                return

            lista = "\n".join(musicas)
            QMessageBox.information(self, "Músicas Disponíveis", f"As músicas encontradas:\n\n{lista}")
        except Exception as e:
            QMessageBox.critical(self, "Erro ao listar músicas", f"Erro: {str(e)}")

    # Método de diagnóstico para depuração do yt-dlp
    def diagnosticar_yt_dlp(self):
        """Executa diagnósticos no yt-dlp e mostra resultados"""
        try:
            resultado = {}

            # Verificar se yt-dlp está instalado
            try:
                check = subprocess.run(
                    ["yt-dlp", "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                resultado["instalado"] = check.returncode == 0
                resultado["versao"] = check.stdout.strip() if check.returncode == 0 else "N/A"
            except FileNotFoundError:
                resultado["instalado"] = False
                resultado["versao"] = "Não encontrado"
            except Exception as e:
                resultado["instalado"] = False
                resultado["erro"] = str(e)

            # Verificar permissões de escrita no diretório atual
            try:
                test_file = "yt_dlp_test.txt"
                with open(test_file, "w") as f:
                    f.write("teste")
                os.remove(test_file)
                resultado["dir_escrita"] = True
            except:
                resultado["dir_escrita"] = False

            # Verificar conectividade com YouTube
            try:
                import urllib.request
                urllib.request.urlopen("https://www.youtube.com", timeout=5)
                resultado["conectividade_youtube"] = True
            except:
                resultado["conectividade_youtube"] = False

            # Mostrar resultados
            msg = f"""Diagnóstico do yt-dlp:
    - Instalado: {resultado['instalado']}
    - Versão: {resultado['versao']}
    - Permissão de escrita: {resultado['dir_escrita']}
    - Conectividade com YouTube: {resultado.get('conectividade_youtube', 'Não verificado')}
    """
            if 'erro' in resultado:
                msg += f"- Erro encontrado: {resultado['erro']}\n"

            msg += "\nSoluções possíveis:\n"
            if not resultado['instalado']:
                msg += "- Instale o yt-dlp: pip install -U yt-dlp\n"
            if not resultado.get('dir_escrita', True):
                msg += "- Execute o aplicativo com permissões de escrita no diretório\n"
            if not resultado.get('conectividade_youtube', True):
                msg += "- Verifique sua conexão com a internet ou firewall\n"

            QMessageBox.information(self, "Diagnóstico yt-dlp", msg)
            return resultado

        except Exception as e:
            QMessageBox.critical(self, "Erro de diagnóstico", f"Falha ao diagnosticar: {str(e)}")
            return {"erro": str(e)}

    # Adicione no __init__ da classe TranspositorBase:
    def inicializar_youtube_downloader(self):
        """Configura o botão de download do YouTube"""
        self.botao_youtube = QPushButton("Baixar do YouTube", self)
        self.layout.addWidget(self.botao_youtube)

        if self.verificar_yt_dlp():
            self.botao_youtube.clicked.connect(self.baixar_youtube)
        else:
            self.botao_youtube.setEnabled(False)
            self.botao_youtube.setToolTip("yt-dlp não encontrado. Instale com 'pip install yt-dlp'")
            print("yt-dlp não está instalado. Para usar o download do YouTube, instale com: pip install yt-dlp")

    def detectar_tonalidade_thread(self, y, sr):
        resultado = detectar_tonalidade_aprimorado(y, sr, mostrar_debug=False)

        nota, modo = resultado.split()
        nota_base = notas.index(nota)
        tipo_escala = modo
        relativa = calcular_relativa(nota_base, tipo_escala)

        tonalidade = f"{nota} {modo} (relativa: {relativa})"

        return {
            "nota_base": nota_base,
            "tipo_escala": tipo_escala,
            "tonalidade": tonalidade
        }

    def on_tonalidade_detectada(self, resultado):
        """Callback quando a tonalidade é detectada"""
        self.nota_base = resultado["nota_base"]
        self.tipo_escala = resultado["tipo_escala"]
        self.tonalidade_detectada = resultado["tonalidade"]

        self.label_tom.setText(f"Tom detectado: {self.tonalidade_detectada}")
        self.atualizar_tom_atual_na_interface()

        self.audio_transposto.clear()

        self.loading_label.setText("Calculando espectrograma base...")
        try:
            stft = librosa.stft(self.y_original)
            self.audio_transposto[0] = librosa.istft(stft)
            self.stft_base = stft

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Erro ao calcular o tom original", str(e))
            self.loading_label.setText("Erro")
            return

        tons_iniciais = [-1, 1]
        self.tons_carregados = 1
        self.progress_bar.setValue(1)
        self.progress_bar.setMaximum(3)
        self.progress_bar.setVisible(True)

        def transpor(semitons):
            try:
                self.loading_label.setText(f"Processando tom {semitons:+}...")

                y_mod = librosa.effects.pitch_shift(self.y_original, sr=self.sr, n_steps=semitons)

                if y_mod is None or len(y_mod) == 0:
                    raise ValueError("Resultado de transposição está vazio.")

                self.audio_transposto[semitons] = (y_mod, self.sr)

            except Exception as e:
                import traceback
                erro_str = f"Erro ao processar tom {semitons:+}:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                print(erro_str)
                try:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "Erro ao processar tom", erro_str)
                except Exception as ui_error:
                    print(f"Erro ao mostrar QMessageBox: {ui_error}")
            finally:
                self.tons_carregados += 1
                self.progress_bar.setValue(self.tons_carregados)
                if self.tons_carregados == 3:
                    self.progress_bar.setVisible(False)
                    self.label_status.setText("Música pronta para reprodução!")
                    self.botao_play.setEnabled(True)
                    self.botao_subir.setEnabled(True)
                    self.botao_descer.setEnabled(True)
                    self.botao_exportar.setEnabled(True)
                self.loading_label.setText("")

        for semitons in tons_iniciais:
            threading.Thread(target=transpor, args=(semitons,), daemon=True).start()

        self.duracao_total = int(len(self.y_original) / self.sr)
        self.slider.setRange(0, self.duracao_total)
        self.label_duracao.setText(self.formatar_tempo(self.duracao_total))
        self.slider.setValue(0)
        self.label_tempo.setText("0:00")
        self.slider.setEnabled(True)

        self.botao_carregar.setEnabled(True)

    def on_error(self, erro):
        print("Erro capturado pela thread:\n", erro)  # Log no console
        QMessageBox.critical(self, "Erro em thread", erro)
        self.loading_label.setText("")
        self.botao_carregar.setEnabled(True)
        self.progress_bar.setVisible(False)

    def gerar_tom(self, semitons, inicial=False, progresso_idx=0, callback=None):
        print(f"[DEBUG] Tentando gerar tom {semitons:+}")
        audio_existente = self.audio_transposto.get(semitons)
        if audio_existente is not None and self.audio_valido(audio_existente):
            print(f"[DEBUG] Tom {semitons:+} já existe e está válido.")
            if callable(callback):
                QTimer.singleShot(0, callback)
            return

        def processar():
            try:
                print(f"[DEBUG] Iniciando processamento de tom {semitons:+}...")
                if self.y_original is None or self.sr is None:
                    raise ValueError("Áudio original ou taxa de amostragem não definidos")
                y_mod = librosa.effects.pitch_shift(self.y_original, sr=self.sr, n_steps=semitons)
                sr_mod = self.sr
                if not self.audio_valido(y_mod):
                    raise ValueError("Resultado de transposição inválido")
                self.audio_transposto[semitons] = (y_mod, sr_mod)
                print(f"[DEBUG] Tom {semitons:+} gerado com sucesso ({len(y_mod)} amostras)")
            except Exception as e:
                erro_str = f"[ERRO] ao gerar tom {semitons:+}: {e}"
                print(erro_str)
                if self.isVisible():
                    QMessageBox.critical(self, "Erro ao processar tom", erro_str)
            finally:
                if self.isVisible():
                    self.loading_label.setText("")
                    self.botao_play.setEnabled(True)
                    self.botao_exportar.setEnabled(True)
                if callback:
                    QTimer.singleShot(0, callback)  # ✅ Garante que será executado na thread principal

        threading.Thread(target=processar, daemon=True).start()

    def tocar_ou_pausar(self):
        if self.reproduzindo:
            self.pausar_audio()
            return

        semitons_para_reproduzir = self.tom_em_reproducao if self.tom_em_reproducao is not None else self.ton_atual

        def ao_gerar():
            audio = self.audio_transposto.get(semitons_para_reproduzir)
            if self.audio_valido(audio):
                self.loading_label.setText("")
                self.botao_play.setEnabled(True)
                self.reproduzir_audio(semitons_para_reproduzir)
            else:
                self.loading_label.setText("Erro ao carregar o tom")
                QMessageBox.warning(self, "Erro ao tocar",
                                    f"Não foi possível tocar o tom {semitons_para_reproduzir:+}.")

        self.botao_play.setEnabled(False)
        self.loading_label.setText("Gerando tom sob demanda...")

        self.gerar_tom(semitons_para_reproduzir, callback=ao_gerar)

    def pausar_audio(self):
        self.reproduzindo = False
        self.chegou_ao_fim = False  # ← Evita reset no pause manual
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.botao_play.setText("▶ Tocar")
        self.timer.stop()

    def reproduzir_audio(self, semitons):
        try:
            dados = self.audio_transposto.get(semitons)

            if dados is None:
                raise ValueError("Tom não encontrado")

            if isinstance(dados, tuple):
                audio, sr = dados
            else:
                audio = dados
                sr = self.sr  # fallback para versões antigas

            if not isinstance(audio, np.ndarray):
                raise ValueError("Áudio não é um array numpy.")

            if audio.ndim != 1:
                raise ValueError("Áudio deve ser mono (1D array).")

            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                raise ValueError("Áudio contém valores inválidos.")

            if len(audio) == 0:
                raise ValueError("Áudio transposto está vazio.")

            print(f"[DEBUG] Reproduzindo tom {semitons}, {len(audio)} amostras.")

            self.buffer_audio = audio
            self.tom_em_reproducao = semitons

            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass

            self.chegou_ao_fim = False

            self.stream = sd.OutputStream(
                samplerate=sr,
                channels=1,
                dtype='float32',
                callback=self.audio_callback,
                finished_callback=self.audio_finished
            )

            self.reproduzindo = True
            self.botao_play.setText("⏸ Pausar")
            self.stream.start()
            self.timer.start()

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Erro ao reproduzir", str(e))
            self.reproduzindo = False
            self.botao_play.setText("▶ Tocar")

    def audio_callback(self, outdata, frames, time, status):
        try:
            if not self.reproduzindo or self.buffer_audio is None:
                outdata[:] = np.zeros((frames, 1))
                return

            fim = self.posicao_amostral + frames

            if self.posicao_amostral >= len(self.buffer_audio):
                outdata[:] = np.zeros((frames, 1))
                return

            if fim > len(self.buffer_audio):
                bloco = self.buffer_audio[self.posicao_amostral:]
                bloco = np.pad(bloco, (0, frames - len(bloco)), mode='constant')
            else:
                bloco = self.buffer_audio[self.posicao_amostral:fim]

            if bloco.ndim != 1 or np.any(np.isnan(bloco)) or np.any(np.isinf(bloco)):
                raise ValueError("Bloco de áudio inválido")

            outdata[:] = bloco.reshape(-1, 1).astype(np.float32)
            self.posicao_amostral += frames

            if self.posicao_amostral >= len(self.buffer_audio):
                self.reproduzindo = False
                self.chegou_ao_fim = True  # ← Sinaliza fim real da música

        except Exception as e:
            print("[ERRO CALLBACK]", e)
            outdata[:] = np.zeros((frames, 1))
            self.reproduzindo = False

    def audio_finished(self):
        if getattr(self, 'chegou_ao_fim', False):
            self.chegou_ao_fim = False
            self.pausar_audio()
            self.posicao_amostral = 0
            self.slider.setValue(0)
            self.label_tempo.setText("0:00")

    def atualizar_slider(self):
        if not self.reproduzindo or self.slider_sendo_arrastado:
            return

        segundos = int(self.posicao_amostral / self.sr)
        if segundos > self.duracao_total:
            segundos = self.duracao_total

        self.slider.setValue(segundos)
        self.label_tempo.setText(self.formatar_tempo(segundos))

        if segundos >= self.duracao_total:
            self.pausar_audio()
            self.posicao_amostral = 0
            self.slider.setValue(0)
            self.label_tempo.setText("0:00")

    def slider_pressionado(self):
        self.slider_sendo_arrastado = True

    def slider_solto(self):
        nova_posicao = self.slider.value()
        self.posicao_amostral = int(nova_posicao * self.sr)
        self.slider_sendo_arrastado = False

        if self.reproduzindo:
            if self.stream:
                self.stream.stop()
                self.stream.close()

            # ❗Reinicia o stream apenas com o buffer já carregado
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.sr,
                    channels=1,
                    dtype='float32',
                    callback=self.audio_callback,
                    finished_callback=self.audio_finished
                )
                self.stream.start()
            except Exception as e:
                QMessageBox.critical(self, "Erro ao continuar reprodução", str(e))

    def atualizar_tempo_slider(self, valor):
        self.label_tempo.setText(self.formatar_tempo(valor))

    def formatar_tempo(self, s):
        return f"{s // 60}:{s % 60:02}"

    def audio_valido(self, audio):
        try:
            if isinstance(audio, tuple):
                y, _ = audio
            else:
                y = audio
            return (
                    isinstance(y, np.ndarray)
                    and y.ndim == 1
                    and len(y) > 0
                    and not np.any(np.isnan(y))
                    and not np.any(np.isinf(y))
            )
        except Exception as e:
            print(f"[DEBUG] Falha ao validar áudio: {e}")
            return False

    def atualizar_tom_atual_na_interface(self):
        nota_atual = (self.nota_base + self.ton_atual) % 12
        nome_nota = notas[nota_atual]
        modo = self.tipo_escala

        if modo == "Maior":
            # Relativa menor está 9 semitons abaixo da maior
            relativa = notas[(nota_atual - 3) % 12] + " menor"
        else:
            # Relativa maior está 3 semitons acima da menor
            relativa = notas[(nota_atual + 3) % 12] + " Maior"

        texto = f"Tom atual: {nome_nota} {modo} (relativa: {relativa})"
        self.label_tom.setText(texto)

    def subir_tom(self):
        if self.ton_atual >= self.limite_transposicao:
            QMessageBox.information(self, "Limite atingido",
                                    f"O tom máximo permitido é +{self.limite_transposicao} semitons.")
            return

        self.ton_atual += 1
        self.atualizar_tom_atual_na_interface()

        if self.reproduzindo:
            self.pausar_audio()

        self.botao_play.setEnabled(False)
        self.botao_exportar.setEnabled(False)
        self.loading_label.setText(f"Gerando tom {self.ton_atual:+}...")

        def ao_gerar():
            audio = self.audio_transposto.get(self.ton_atual)
            if self.audio_valido(audio):
                self.loading_label.setText("")
                self.botao_play.setEnabled(True)
                self.botao_exportar.setEnabled(True)
                self.reproduzir_audio(self.ton_atual)
            else:
                self.loading_label.setText("Erro: áudio inválido")
                QMessageBox.warning(self, "Falha ao gerar tom",
                                    f"O tom {self.ton_atual:+} não pôde ser gerado corretamente.")

        self.gerar_tom(self.ton_atual, callback=ao_gerar)

    def descer_tom(self):
        if self.ton_atual <= -self.limite_transposicao:
            QMessageBox.information(self, "Limite atingido",
                                    f"O tom mínimo permitido é -{self.limite_transposicao} semitons.")
            return

        self.ton_atual -= 1
        self.atualizar_tom_atual_na_interface()

        if self.reproduzindo:
            self.pausar_audio()

        self.botao_play.setEnabled(False)
        self.botao_exportar.setEnabled(False)
        self.loading_label.setText(f"Gerando tom {self.ton_atual:+}...")

        def ao_gerar():
            audio = self.audio_transposto.get(self.ton_atual)
            if self.audio_valido(audio):
                self.loading_label.setText("")
                self.botao_play.setEnabled(True)
                self.botao_exportar.setEnabled(True)
                self.reproduzir_audio(self.ton_atual)
            else:
                self.loading_label.setText("Erro: áudio inválido")
                QMessageBox.warning(self, "Falha ao gerar tom",
                                    f"O tom {self.ton_atual:+} não pôde ser gerado corretamente.")

        self.gerar_tom(self.ton_atual, callback=ao_gerar)

    def atualizar_botoes_tom(self):
        self.botao_subir.setEnabled(self.ton_atual < self.limite_transposicao)
        self.botao_descer.setEnabled(self.ton_atual > -self.limite_transposicao)

    def exportar_tom_atual(self):
        try:
            audio = self.audio_transposto.get(self.ton_atual)
            if not isinstance(audio, np.ndarray) or len(audio) == 0:
                raise ValueError("O áudio do tom atual ainda não foi processado.")

            # Calcular o fator de correção de tempo baseado na transposição
            # Invertemos a relação para compensar o efeito
            pitch_factor = 2.0 ** (-self.ton_atual / 12.0)  # Observe o sinal negativo

            # Para tons baixos (negativos), o pitch_factor será > 1, aumentando a taxa de amostragem
            sr_ajustado = int(self.sr * pitch_factor)

            print(f"[DEBUG] Tom atual: {self.ton_atual:+} semitons")
            print(f"[DEBUG] Amostras originais: {len(self.y_original)}")
            print(f"[DEBUG] Amostras atuais: {len(audio)}")
            print(f"[DEBUG] Taxa original: {self.sr} Hz")
            print(f"[DEBUG] Taxa ajustada: {sr_ajustado} Hz (fator {pitch_factor:.3f})")

            # Normalizar áudio para evitar clipping
            if np.max(np.abs(audio)) > 0.98:
                audio = audio / np.max(np.abs(audio)) * 0.98
                print("[DEBUG] Áudio normalizado para evitar clipping")

            # Caixa de diálogo para salvar
            caminho, _ = QFileDialog.getSaveFileName(
                self, "Salvar áudio", f"musica_{self.ton_atual:+}.wav", "WAV files (*.wav)"
            )
            if not caminho:
                return

            # Verificar se o arquivo já existe e remover
            import os
            if os.path.exists(caminho):
                os.remove(caminho)

            # Usar a taxa de amostragem ajustada para manter a duração correta
            sf.write(
                caminho,
                audio,
                sr_ajustado,  # Taxa ajustada pelo fator de pitch
                subtype='PCM_16',
                format='WAV'
            )

            # Verificação final da duração do arquivo gerado
            duracao_arquivo = sf.info(caminho).duration
            duracao_original = len(self.y_original) / self.sr
            print(f"[DEBUG] Duração original: {duracao_original:.2f}s")
            print(f"[DEBUG] Duração do arquivo final: {duracao_arquivo:.2f}s")
            print(f"[DEBUG] Correção de tempo: {duracao_original / duracao_arquivo:.3f}x")

            self.loading_label.setText("Arquivo exportado com sucesso!")

        except Exception as e:
            erro_str = f"Erro ao exportar áudio:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(erro_str)
            QMessageBox.critical(self, "Erro ao exportar", erro_str)

    def iniciar_processamento_em_background(self):
        import subprocess, sys, os
        caminho_script = os.path.join(os.path.dirname(__file__), "auto_processador.py")
        try:
            subprocess.Popen(
                [sys.executable, caminho_script],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
        except Exception as e:
            print(f"[ERRO] Processador de tons não iniciou: {e}")

    def closeEvent(self, event):
        # Garantir que o stream de áudio seja encerrado corretamente
        if self.stream:
            self.stream.stop()
            self.stream.close()
        event.accept()

    def carregar_tons_processados(self, nome_arquivo):
        """Carrega os tons já processados em tons_processados/NOME e mostra o tom do cache"""
        import os, json

        try:
            with open("cache_tonalidades.json", "r", encoding="utf-8") as f:
                self.cache_tonalidades = json.load(f)
        except:
            self.cache_tonalidades = {}

        base = os.path.splitext(os.path.basename(nome_arquivo))[0]
        pasta = os.path.join("tons_processados", base)

        if not os.path.exists(pasta):
            return False  # Nada pronto

        self.audio_transposto = {}
        self.tom_em_reproducao = None
        self.ton_atual = 0
        self.reproduzindo = False
        self.posicao_amostral = 0

        for semitons in range(-12, 13):
            caminho = os.path.join(pasta, f"{base}_{semitons:+}.wav")
            if os.path.exists(caminho):
                try:
                    y, sr = sf.read(caminho)
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    self.audio_transposto[semitons] = (y, sr)  # <-- salva o sr junto

                except Exception as e:
                    print(f"[ERRO] ao carregar tom {semitons:+}: {e}")

        if 0 not in self.audio_transposto:
            return False

        self.y_original, self.sr = self.audio_transposto[0]
        self.duracao_total = int(len(self.y_original) / self.sr)
        self.slider.setRange(0, self.duracao_total)
        self.slider.setValue(0)
        self.label_duracao.setText(self.formatar_tempo(self.duracao_total))
        self.label_tempo.setText("0:00")
        self.slider.setEnabled(True)

        self.botao_play.setEnabled(True)
        self.botao_exportar.setEnabled(True)
        self.botao_subir.setEnabled(True)
        self.botao_descer.setEnabled(True)
        self.label_status.setText("Tons carregados do disco.")

        # 🧠 Tenta ler a tonalidade diretamente do cache
        info = self.cache_tonalidades.get(os.path.basename(nome_arquivo))
        if isinstance(info, dict):
            # Preferência para 'tonalidade_escolhida' (que é o campo salvo pelo auto_processador)
            tonalidade = info.get("tonalidade_escolhida") or info.get("tonalidade_avancada") or info.get(
                "tonalidade_original")
            if tonalidade:
                self.tonalidade_detectada = tonalidade
                self.label_tom.setText(f"Tom detectado: {tonalidade}")

                # 🎯 Atualiza nota_base com base na tonalidade carregada do cache
                nome = tonalidade.split()[0] if tonalidade else None
                if nome in notas:
                    self.nota_base = notas.index(nome)
                else:
                    self.nota_base = 0  # fallback para C se não reconhecido

            else:
                self.label_tom.setText("Tom detectado: desconhecido")
        else:
            self.label_tom.setText("Tom detectado: desconhecido")

            # Confirma se todos os tons de -6 a +6 estão presentes
            for semitons in range(-6, 7):
                if semitons not in self.audio_transposto:
                    print(f"[DEBUG] Tom faltando: {semitons:+}")
                    return False  # ainda não pronto

        return True


class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

class AudioWorker(threading.Thread):
    """Worker thread para processar tarefas em segundo plano"""
    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.daemon = True

    def run(self):
        try:
            result = self.task(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            erro_completo = traceback.format_exc()
            self.signals.error.emit(erro_completo)


class YoutubeDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 Baixar do YouTube")
        self.setFixedSize(400, 150)
        self.setStyleSheet(f"""
            background-color: {COLORS['card_bg']};
            color: {COLORS['text_primary']};
            border-radius: 12px;
        """)

        layout = QVBoxLayout(self)

        label = QLabel("Cole a URL do vídeo do YouTube:")
        label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(label)

        self.input = QLineEdit()
        self.input.setPlaceholderText("https://youtube.com/watch?v=...")
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #2a2a2a;
                border: 1px solid #444;
                padding: 8px;
                border-radius: 6px;
                color: {COLORS['text_primary']};
            }}
        """)
        layout.addWidget(self.input)

        buttons = QHBoxLayout()
        self.btn_ok = RoundedButton("Baixar", color=COLORS["primary"])
        self.btn_cancel = RoundedButton("Cancelar", color=COLORS["danger"])
        buttons.addWidget(self.btn_ok)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_url(self):
        return self.input.text()

class SalvarMusicaDialog(QDialog):
    def __init__(self,nome_sugerido=""):
        super().__init__()
        self.setWindowTitle("🎶 Salvar música")
        self.setFixedSize(420, 200)
        self.setStyleSheet(f"""
            background-color: {COLORS['card_bg']};
            color: {COLORS['text_primary']};
            border-radius: 12px;
        """)

        layout = QVBoxLayout(self)

        label_nome = QLabel("Nome da música:")
        layout.addWidget(label_nome)

        self.input_nome = QLineEdit()
        self.input_nome.setPlaceholderText("Ex: Minha Música Linda")
        self.input_nome.setText(nome_sugerido)  # 👈 AQUI
        self.input_nome.setStyleSheet(f"""
            QLineEdit {{
                background-color: #2a2a2a;
                border: 1px solid #444;
                padding: 8px;
                border-radius: 6px;
                color: {COLORS['text_primary']};
            }}
        """)
        layout.addWidget(self.input_nome)

        label_destino = QLabel("Salvar em:")
        layout.addWidget(label_destino)

        caminho_layout = QHBoxLayout()
        self.input_caminho = QLineEdit()
        self.input_caminho.setReadOnly(True)
        self.input_caminho.setStyleSheet("background-color: #2a2a2a; padding: 5px;")
        btn_escolher = RoundedButton("Escolher pasta", color=COLORS["secondary"])
        btn_escolher.clicked.connect(self.selecionar_pasta)
        caminho_layout.addWidget(self.input_caminho)
        caminho_layout.addWidget(btn_escolher)
        layout.addLayout(caminho_layout)

        botoes = QHBoxLayout()
        btn_ok = RoundedButton("Salvar", color=COLORS["primary"])
        btn_cancelar = RoundedButton("Cancelar", color=COLORS["danger"])
        btn_ok.clicked.connect(self.accept)
        btn_cancelar.clicked.connect(self.reject)
        botoes.addWidget(btn_ok)
        botoes.addWidget(btn_cancelar)
        layout.addLayout(botoes)

        self.caminho = ""

    def selecionar_pasta(self):
        pasta = QFileDialog.getExistingDirectory(self, "Escolher pasta")
        if pasta:
            self.caminho = pasta
            self.input_caminho.setText(pasta)

    def get_dados(self):
        return self.input_nome.text().strip(), self.caminho


import sys

def excecao_global(tipo, valor, tb):
    msg = ''.join(traceback.format_exception(tipo, valor, tb))
    print("Exceção não capturada:\n", msg)
    try:
        from PyQt6.QtWidgets import QMessageBox, QApplication
        app = QApplication.instance()
        if app is not None and app.activeWindow():
            QMessageBox.critical(app.activeWindow(), "Erro não capturado", msg)
        else:
            print("Erro não pôde ser mostrado em QMessageBox (sem janela ativa).")
    except Exception:
        print("Erro não pôde ser mostrado em QMessageBox.")


sys.excepthook = excecao_global