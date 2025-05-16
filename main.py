import sys
import threading
import queue
import numpy as np
import librosa
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QSlider, QHBoxLayout,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette, QPixmap, QLinearGradient, QGradient
import soundfile as sf
from PyQt6.QtWidgets import QFileDialog
import sys
import traceback

# Cores para a interface moderna
COLORS = {
    "primary": "#3498db",  # Azul principal
    "secondary": "#2ecc71",  # Verde para ações positivas
    "danger": "#e74c3c",  # Vermelho para ações negativas
    "background": "#f5f5f5",  # Cinza claro para o fundo
    "dark": "#2c3e50",  # Azul escuro para texto e detalhes
    "light": "#ecf0f1",  # Cinza muito claro para áreas de destaque
    "accent": "#9b59b6",  # Roxo para destaque de elementos
    "slider_groove": "#bdc3c7",  # Cinza para o trilho do slider
    "slider_handle": "#3498db",  # Azul para o manipulador do slider
}

# Notas musicais
notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# Classe para comunicar eventos das threads para a UI
class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    status = pyqtSignal(str)


class AudioWorker(threading.Thread):
    """Worker thread para processamento de áudio"""

    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.daemon = True  # Thread será encerrada quando o programa principal terminar

    def run(self):
        try:
            result = self.task(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


class StylishButton(QPushButton):
    """Botão estilizado com cores personalizadas"""

    def __init__(self, text, parent=None, color=COLORS["primary"], icon=None):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 10))
        self.setMinimumHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color = color

        # Estilo usando CSS
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color, 10)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 20)};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #888888;
            }}
        """)

        # Configurar ícone, se fornecido
        if icon:
            self.setIcon(QIcon(icon))
            self.setIconSize(QSize(18, 18))

    def _darken_color(self, hex_color, percent):
        """Escurece uma cor em uma determinada porcentagem"""
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb = tuple(max(0, int(c * (1 - percent / 100))) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class CustomSlider(QSlider):
    """Slider personalizado com estilo moderno"""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {COLORS["slider_groove"]};
                border-radius: 3px;
            }}

            QSlider::handle:horizontal {{
                background: {COLORS["slider_handle"]};
                border: none;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}

            QSlider::sub-page:horizontal {{
                background: {COLORS["primary"]};
                border-radius: 3px;
            }}
        """)


class InfoLabel(QLabel):
    """Rótulo estilizado para informações"""

    def __init__(self, text="", parent=None, is_bold=False, size=10):
        super().__init__(text, parent)
        font = QFont("Segoe UI", size)
        if is_bold:
            font.setBold(True)
        self.setFont(font)
        self.setStyleSheet(f"color: {COLORS['dark']};")


class Section(QFrame):
    """Seção separada para organizar os elementos"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS["light"]};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        self.setContentsMargins(10, 10, 10, 10)


class Transpositor(QWidget):
    def __init__(self):
        super().__init__()

        # Configurações da janela
        self.setWindowTitle("Music Transposer")
        self.setFixedSize(500, 500)
        self.setStyleSheet(f"background-color: {COLORS['background']};")

        # Variáveis de estado
        self.nota_base = 0
        self.tipo_escala = "Maior"
        self.audio_transposto = {}
        self.y_original = None
        self.sr = 22050
        self.ton_atual = 0
        self.stream = None
        self.posicao_amostral = 0
        self.reproduzindo = False
        self.slider_sendo_arrastado = False
        self.buffer_audio = None
        self.duracao_total = 0
        self.tonalidade_detectada = ""
        self.stft_base = None

        # Fila para processar tons em ordem
        self.ton_queue = queue.Queue()
        self.worker_thread = None
        self.processando_tons = False

        # Configuração do layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Cabeçalho
        header = QHBoxLayout()
        title_label = InfoLabel("Music Transposer", is_bold=True, size=16)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title_label)
        main_layout.addLayout(header)

        # Seção de Status e Carregamento
        status_section = Section()
        status_layout = QVBoxLayout(status_section)

        self.label_status = InfoLabel("Nenhum arquivo carregado")
        self.label_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.label_status)

        load_button_layout = QHBoxLayout()
        load_button_layout.addStretch()
        self.botao_carregar = StylishButton("Carregar Música", color=COLORS["secondary"])
        self.botao_carregar.clicked.connect(self.carregar_musica)
        self.botao_carregar.setFixedWidth(200)
        load_button_layout.addWidget(self.botao_carregar)
        load_button_layout.addStretch()
        status_layout.addLayout(load_button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 3)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS["light"]};
                border-radius: 3px;
                text-align: center;
                height: 12px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS["primary"]};
                border-radius: 3px;
            }}
        """)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        main_layout.addWidget(status_section)

        # Seção de informações do tom
        info_section = Section()
        info_layout = QVBoxLayout(info_section)

        self.label_tom = InfoLabel("", is_bold=True, size=12)
        self.label_tom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.label_tom)

        # Botões de alteração de tom
        tom_buttons_layout = QHBoxLayout()
        tom_buttons_layout.setSpacing(10)

        self.botao_descer = StylishButton("↓ Baixar Tom", color=COLORS["danger"])
        self.botao_descer.clicked.connect(self.descer_tom)
        self.botao_descer.setEnabled(False)
        tom_buttons_layout.addWidget(self.botao_descer)

        self.botao_subir = StylishButton("↑ Subir Tom", color=COLORS["accent"])
        self.botao_subir.clicked.connect(self.subir_tom)
        self.botao_subir.setEnabled(False)
        tom_buttons_layout.addWidget(self.botao_subir)

        info_layout.addLayout(tom_buttons_layout)
        main_layout.addWidget(info_section)

        # Seção de reprodução
        player_section = Section()
        player_layout = QVBoxLayout(player_section)

        # Slider e tempo
        self.slider = CustomSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.atualizar_tempo_slider)
        self.slider.sliderPressed.connect(self.slider_pressionado)
        self.slider.sliderReleased.connect(self.slider_solto)
        player_layout.addWidget(self.slider)

        time_layout = QHBoxLayout()
        self.label_tempo = InfoLabel("0:00")
        self.label_duracao = InfoLabel("0:00")
        time_layout.addWidget(self.label_tempo)
        time_layout.addStretch()
        time_layout.addWidget(self.label_duracao)
        player_layout.addLayout(time_layout)

        # Botão de reprodução
        play_layout = QHBoxLayout()
        play_layout.addStretch()
        self.botao_play = StylishButton("▶ Tocar", color=COLORS["primary"])
        self.botao_play.clicked.connect(self.tocar_ou_pausar)
        self.botao_play.setEnabled(False)
        self.botao_play.setMinimumWidth(120)
        play_layout.addWidget(self.botao_play)
        play_layout.addStretch()
        player_layout.addLayout(play_layout)

        main_layout.addWidget(player_section)

        self.botao_exportar = StylishButton("💾 Exportar Música", color=COLORS["secondary"])
        self.botao_exportar.clicked.connect(self.exportar_tom_atual)
        self.botao_exportar.setEnabled(False)
        play_layout.addWidget(self.botao_exportar)

        # Seção de status de processamento
        status_processing = QHBoxLayout()
        self.loading_label = InfoLabel("")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_processing.addWidget(self.loading_label)
        main_layout.addLayout(status_processing)

        # Footer com informações
        footer = QHBoxLayout()
        footer_label = InfoLabel("Desenvolvido por Jefferson De Sousa Amorim", size=8)
        footer_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        footer.addWidget(footer_label)
        main_layout.addLayout(footer)

        # Timer para atualizar o slider
        self.timer = QTimer()
        self.timer.setInterval(100)  # Atualizar a cada 100ms para melhor precisão
        self.timer.timeout.connect(self.atualizar_slider)

        # Worker thread para processamento
        self.iniciar_worker_thread()

    def iniciar_worker_thread(self):
        """Inicia uma thread de trabalho para processar a fila de tons"""

        def worker():
            while True:
                try:
                    # Obtém a próxima tarefa da fila
                    semitons, inicial, progresso_idx = self.ton_queue.get()
                    self.processando_tons = True

                    # Sinaliza que está começando a processar
                    self.loading_label.setText(f"Processando tom {semitons:+}...")

                    # Processa o tom
                    if semitons not in self.audio_transposto:
                        y_mod = librosa.effects.pitch_shift(self.y_original, sr=self.sr, n_steps=semitons)
                        self.audio_transposto[semitons] = y_mod

                    # Atualiza a UI após o processamento
                    if inicial:
                        self.progress_bar.setValue(progresso_idx)
                        if progresso_idx == 3:
                            self.progress_bar.setVisible(False)
                            self.label_status.setText("Música pronta para reprodução!")
                            self.botao_play.setEnabled(True)
                            self.botao_subir.setEnabled(True)
                            self.botao_descer.setEnabled(True)
                            self.botao_exportar.setEnabled(True)

                    # Limpa o status de processamento
                    self.loading_label.setText("")
                    self.processando_tons = False
                    self.botao_play.setEnabled(True)  # 🔓 reativa ao terminar

                    # Marca a tarefa como concluída
                    self.ton_queue.task_done()
                except Exception as e:
                    from PyQt6.QtWidgets import QMessageBox
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
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Usar uma thread para carregar o arquivo de áudio
        def carregar_arquivo():
            try:
                y, sr = librosa.load(caminho, sr=None, mono=True)
                return y, sr
            except Exception as e:
                return None, str(e)

        def on_load_finished(result):
            if result is None or isinstance(result[1], str):
                # Erro ao carregar o arquivo
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Erro ao carregar arquivo", result[1])
                self.botao_carregar.setEnabled(True)
                self.progress_bar.setVisible(False)
                self.label_status.setText("Erro ao carregar música")
                return

            # Arquivo carregado com sucesso
            y, sr = result
            self.y_original = y
            self.sr = sr
            self.ton_atual = 0
            self.reproduzindo = False
            self.posicao_amostral = 0

            # Detectar a tonalidade em uma thread separada
            self.loading_label.setText("Detectando tonalidade...")
            worker = AudioWorker(self.detectar_tonalidade_thread, y, sr)
            worker.signals.finished.connect(self.on_tonalidade_detectada)
            worker.signals.error.connect(self.on_error)
            worker.start()

        # Criar e iniciar o worker
        worker = AudioWorker(carregar_arquivo)
        worker.signals.finished.connect(on_load_finished)
        worker.signals.error.connect(self.on_error)
        worker.start()

    def detectar_tonalidade_thread(self, y, sr):
        """Detecção otimizada de tonalidade usando perfil de correlação"""

        # Usa apenas a média da matriz de cromas para representação geral
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Perfis baseados em Krumhansl-Schmuckler
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Normaliza os vetores
        chroma_norm = (chroma_mean - np.mean(chroma_mean)) / np.std(chroma_mean)
        major_profile = (major_profile - np.mean(major_profile)) / np.std(major_profile)
        minor_profile = (minor_profile - np.mean(minor_profile)) / np.std(minor_profile)

        # Usa correlação vetorial simples (mais rápida que np.corrcoef)
        def correlacao_rotacionada(perfil):
            return [np.dot(np.roll(perfil, i), chroma_norm) for i in range(12)]

        corr_maj = correlacao_rotacionada(major_profile)
        corr_min = correlacao_rotacionada(minor_profile)

        max_maj = int(np.argmax(corr_maj))
        max_min = int(np.argmax(corr_min))

        if max(corr_maj) > max(corr_min):
            nota_base = max_maj
            tipo_escala = "Maior"
            tonalidade = f"{notas[max_maj]} Maior"
        else:
            nota_base = max_min
            tipo_escala = "Menor"
            tonalidade = f"{notas[max_min]} Menor"

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
                rate = 2.0 ** (semitons / 12.0)

                stft_mod = librosa.phase_vocoder(D=stft, rate=rate, hop_length=512)

                y_mod = librosa.istft(stft_mod, hop_length=512)

                if y_mod is None or len(y_mod) == 0:
                    raise ValueError("Resultado de transposição está vazio.")

                self.audio_transposto[semitons] = y_mod


            except Exception as e:

                import traceback

                erro_str = f"Erro ao processar tom {semitons:+}:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"

                print(erro_str)  # <-- mostra no console mesmo se UI falhar

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
                    self.botao_exportar.setEnabled(True)  # <-- ADICIONE ISSO
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
        """Callback quando ocorre um erro em uma thread"""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Erro", erro)
        self.loading_label.setText("")
        self.botao_carregar.setEnabled(True)
        self.progress_bar.setVisible(False)

    def gerar_tom(self, semitons, inicial=False, progresso_idx=0):
        if semitons in self.audio_transposto:
            return

        def processar():
            try:
                self.loading_label.setText(f"Processando tom {semitons:+}...")
                self.botao_play.setEnabled(False)
                self.botao_exportar.setEnabled(False)

                y_mod = librosa.effects.pitch_shift(self.y_original, sr=self.sr, n_steps=semitons)

                if y_mod is None or len(y_mod) == 0:
                    raise ValueError("Resultado de transposição está vazio.")

                self.audio_transposto[semitons] = y_mod

            except Exception as e:
                erro_str = (
                    f"Erro ao processar tom {semitons:+}:\n{str(e)}\n\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                print(erro_str)
                if self.isVisible():
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "Erro ao processar tom", erro_str)
            finally:
                if self.isVisible():
                    self.loading_label.setText("")
                    self.botao_play.setEnabled(True)
                    self.botao_exportar.setEnabled(True)

        threading.Thread(target=processar, daemon=True).start()

    def tocar_ou_pausar(self):
        if self.reproduzindo:
            self.pausar_audio()
        else:
            if self.ton_atual not in self.audio_transposto:
                self.loading_label.setText("Gerando tom sob demanda...")
                self.botao_play.setEnabled(False)  # 🔒 trava o botão

                self.gerar_tom(self.ton_atual)

                # Verifica periodicamente se o tom está pronto
                def verificar_tom():
                    audio = self.audio_transposto.get(self.ton_atual)
                    if isinstance(audio, np.ndarray) and len(audio) > 0:
                        self.loading_label.setText("")
                        self.botao_play.setEnabled(True)  # 🔓 reativa o botão
                        self.reproduzir_audio(self.ton_atual)
                    else:
                        QTimer.singleShot(500, verificar_tom)

                verificar_tom()
            else:
                self.reproduzir_audio(self.ton_atual)

    def pausar_audio(self):
        self.reproduzindo = False
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
            audio = self.audio_transposto.get(semitons)

            if not isinstance(audio, np.ndarray):
                raise ValueError("Áudio não é um array numpy.")

            if audio.ndim != 1:
                raise ValueError("Áudio deve ser mono (1D array).")

            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                raise ValueError("Áudio contém valores inválidos (NaN ou inf).")

            if len(audio) == 0:
                raise ValueError("Áudio transposto está vazio.")

            print(f"[DEBUG] Reproduzindo tom {semitons}, {len(audio)} amostras.")

            # Verifica a taxa de amostragem do dispositivo
            device_info = sd.query_devices(kind='output')
            default_samplerate = int(device_info['default_samplerate'])

            print(f"[DEBUG] Criando OutputStream: sample_rate={self.sr}, default_device_sr={default_samplerate}")

            # Se incompatível, faz resample automático
            sr_to_use = self.sr if self.sr <= default_samplerate else default_samplerate
            if self.sr != sr_to_use:
                audio = librosa.resample(audio, orig_sr=self.sr, target_sr=sr_to_use)
                self.sr = sr_to_use

            self.buffer_audio = audio

            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass

            self.stream = sd.OutputStream(
                samplerate=self.sr,
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

        except Exception as e:
            print("[ERRO CALLBACK]", e)
            outdata[:] = np.zeros((frames, 1))
            self.reproduzindo = False

    def audio_finished(self):
        if not self.reproduzindo:
            # Reinicia o player quando a música acaba
            self.pausar_audio()
            self.posicao_amostral = 0
            self.slider.setValue(0)
            self.label_tempo.setText("0:00")

    def atualizar_slider(self):
        if not self.reproduzindo or self.slider_sendo_arrastado:
            return

        segundos = int(self.posicao_amostral / self.sr)

        # Limita o valor máximo do slider
        if segundos > self.duracao_total:
            segundos = self.duracao_total

        self.slider.setValue(segundos)
        self.label_tempo.setText(self.formatar_tempo(segundos))

        # Verifica se chegou ao final da música
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

        # Se estava reproduzindo, continua a reprodução a partir da nova posição
        if self.reproduzindo:
            # Reinicia o stream para a nova posição
            if self.stream:
                self.stream.stop()
                self.stream.close()

            self.reproduzir_audio(self.ton_atual)

    def atualizar_tempo_slider(self, valor):
        # Atualiza o rótulo de tempo enquanto o slider é arrastado
        self.label_tempo.setText(self.formatar_tempo(valor))

    def formatar_tempo(self, s):
        return f"{s // 60}:{s % 60:02}"

    def atualizar_tom_atual_na_interface(self):
        nota_atual = (self.nota_base + self.ton_atual) % 12
        nome_tom = f"{notas[nota_atual]} {self.tipo_escala}"
        self.label_tom.setText(f"Tom atual: {nome_tom}")

    def subir_tom(self):
        if self.ton_atual < 12:
            self.ton_atual += 1
            self.atualizar_tom_atual_na_interface()

            if self.reproduzindo:
                self.pausar_audio()

            self.botao_play.setEnabled(False)
            self.botao_exportar.setEnabled(False)  # também desativa exportar
            self.loading_label.setText(f"Gerando tom {self.ton_atual:+}...")

            def verificar():
                audio = self.audio_transposto.get(self.ton_atual)
                if isinstance(audio, np.ndarray) and len(audio) > 0:
                    self.loading_label.setText("")
                    self.botao_play.setEnabled(True)
                    self.botao_exportar.setEnabled(True)
                    self.reproduzir_audio(self.ton_atual)
                else:
                    QTimer.singleShot(500, verificar)

            self.gerar_tom(self.ton_atual)
            verificar()

    def descer_tom(self):
        if self.ton_atual > -12:
            self.ton_atual -= 1
            self.atualizar_tom_atual_na_interface()

            if self.reproduzindo:
                self.pausar_audio()

            self.botao_play.setEnabled(False)
            self.botao_exportar.setEnabled(False)
            self.loading_label.setText(f"Gerando tom {self.ton_atual:+}...")

            def verificar():
                audio = self.audio_transposto.get(self.ton_atual)
                if isinstance(audio, np.ndarray) and len(audio) > 0:
                    self.loading_label.setText("")
                    self.botao_play.setEnabled(True)
                    self.botao_exportar.setEnabled(True)
                    self.reproduzir_audio(self.ton_atual)
                else:
                    QTimer.singleShot(500, verificar)

            self.gerar_tom(self.ton_atual)
            verificar()

    def exportar_tom_atual(self):
        try:
            audio = self.audio_transposto.get(self.ton_atual)
            if not isinstance(audio, np.ndarray) or len(audio) == 0:
                raise ValueError("O áudio do tom atual ainda não foi processado.")

            # Caixa de diálogo para escolher onde salvar
            caminho, _ = QFileDialog.getSaveFileName(
                self, "Salvar áudio", f"musica_{self.ton_atual:+}.wav", "WAV files (*.wav)"
            )
            if not caminho:
                return  # Usuário cancelou

            # Salvar arquivo
            sf.write(caminho, audio, self.sr)
            self.loading_label.setText("Arquivo exportado com sucesso!")

        except Exception as e:
            import traceback
            erro_str = f"Erro ao exportar áudio:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(erro_str)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Erro ao exportar", erro_str)

    def closeEvent(self, event):
        # Garantir que o stream de áudio seja encerrado corretamente
        if self.stream:
            self.stream.stop()
            self.stream.close()
        event.accept()

def excecao_global(tipo, valor, tb):
    import traceback
    print("Exceção não capturada:", ''.join(traceback.format_exception(tipo, valor, tb)))

sys.excepthook = excecao_global
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Aplica estilo global
    app.setStyle("Fusion")

    janela = Transpositor()
    janela.show()
    sys.exit(app.exec())