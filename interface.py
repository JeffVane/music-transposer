import sys
import threading
import queue
import numpy as np
import librosa
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QSlider, QHBoxLayout,
    QFrame, QSizePolicy, QMainWindow, QStackedWidget, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, QObject, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette, QPixmap, QLinearGradient, QGradient, QPainter, QPainterPath
import soundfile as sf
import traceback
from backend_audio import TranspositorBase
from components import RoundedButton, COLORS  # se também usar aqui

from PyQt6.QtWidgets import QListWidget, QListWidgetItem
import os


# Paleta de cores moderna com tons musicais
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


# Classe para comunicar eventos das threads para a UI
class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    progresso_textual = pyqtSignal(str)  # <-- novo


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


class RoundedButton(QPushButton):
    """Botão moderno com cantos arredondados e efeitos de hover"""

    def __init__(self, text, parent=None, color=COLORS["primary"], icon=None, size=(36, 120)):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 10))
        self.setMinimumHeight(size[0])
        self.setMinimumWidth(size[1])
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color = color

        # Adicionando sombra
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        # Estilo usando CSS
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {COLORS["text_primary"]};
                border-radius: 18px;
                padding: 10px 20px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self._lighten_color(color, 10)};
                transform: scale(1.05);
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 10)};
            }}
            QPushButton:disabled {{
                background-color: #555555;
                color: #888888;
            }}
        """)

        # Configurar ícone, se fornecido
        if icon:
            self.setIcon(QIcon(icon))
            self.setIconSize(QSize(18, 18))

    def _lighten_color(self, hex_color, percent):
        """Clareia uma cor em uma determinada porcentagem"""
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c * (1 + percent / 100))) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def _darken_color(self, hex_color, percent):
        """Escurece uma cor em uma determinada porcentagem"""
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb = tuple(max(0, int(c * (1 - percent / 100))) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class ModernSlider(QSlider):
    """Slider personalizado com estilo moderno e animações"""

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

            QSlider::handle:horizontal:hover {{
                background: {COLORS["accent"]};
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }}
        """)


class StyledLabel(QLabel):
    """Rótulo estilizado para informações"""

    def __init__(self, text="", parent=None, is_bold=False, size=10, color=COLORS["text_primary"]):
        super().__init__(text, parent)
        font = QFont("Segoe UI", size)
        if is_bold:
            font.setBold(True)
        self.setFont(font)
        self.setStyleSheet(f"color: {color};")
        self.setWordWrap(True)


class CardFrame(QFrame):
    """Card moderno com efeito de sombra e cantos arredondados"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Aplicar sombra
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS["card_bg"]};
                border-radius: 12px;
                padding: 15px;
            }}
        """)
        self.setContentsMargins(15, 15, 15, 15)


class WaveformPlaceholder(QFrame):
    """Visualização para forma de onda (placeholder)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setStyleSheet(f"""
            background-color: {COLORS["card_bg"]};
            border-radius: 8px;
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Desenhar fundo
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 8, 8)
        painter.fillPath(path, QColor(COLORS["card_bg"]))

        # Desenhar forma de onda placeholder
        painter.setPen(QColor(COLORS["primary"]))

        center_y = self.height() / 2
        step = self.width() / 40

        for i in range(40):
            x = i * step
            height = np.sin(i * 0.5) * 20
            painter.drawLine(int(x), int(center_y - height), int(x), int(center_y + height))


class ModernProgressBar(QProgressBar):
    """Barra de progresso com estilo moderno"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(False)
        self.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS["progress_bg"]};
                border-radius: 6px;
                height: 8px;
            }}

            QProgressBar::chunk {{
                background-color: {COLORS["primary"]};
                border-radius: 6px;
            }}
        """)


class Transpositor(TranspositorBase):
    def __init__(self):
        super().__init__()

        # Configurações da janela
        self.setWindowTitle("Music Transposer Pro")
        self.setMinimumSize(650, 650)
        self.setStyleSheet(f"background-color: {COLORS['background']};")

        # Variáveis de estado (mantidas do código original)
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

        # Fila para processar tons
        self.ton_queue = queue.Queue()
        self.worker_thread = None
        self.processando_tons = False

        # Configuração do layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # Cabeçalho com logo
        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_container = QVBoxLayout()
        title_label = StyledLabel("MUSIC TRANSPOSER PRO", is_bold=True, size=10)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label = StyledLabel("Ajuste o tom de suas músicas com facilidade",
                                     color=COLORS["text_secondary"], size=11)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)
        header.addLayout(title_container)

        main_layout.addLayout(header)

        # Seção de Status e Carregamento
        status_card = CardFrame()
        status_layout = QVBoxLayout(status_card)

        self.label_status = StyledLabel("Nenhum arquivo carregado")
        self.label_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.label_status)

        load_button_layout = QHBoxLayout()
        load_button_layout.addStretch()
        self.botao_carregar = RoundedButton("Carregar Música", color=COLORS["secondary"])
        self.botao_carregar.clicked.connect(self.carregar_musica)
        self.botao_carregar.setFixedWidth(200)
        load_button_layout.addWidget(self.botao_carregar)
        load_button_layout.addStretch()
        status_layout.addLayout(load_button_layout)



        self.botao_youtube = RoundedButton("🎵 YouTube", color=COLORS["accent"])
        self.botao_youtube.clicked.connect(self.baixar_youtube)
        load_button_layout.addWidget(self.botao_youtube)

        self.progress_bar = ModernProgressBar()
        self.progress_animation_timer = QTimer()
        self.progress_animation_timer.setInterval(30)
        self.progress_animation_timer.timeout.connect(self.animate_progress_bar)
        self._progress_anim_value = 0
        self._progress_anim_direction = 1

        self.progress_bar.setRange(0, 3)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        main_layout.addWidget(status_card)

        # Visualização da forma de onda (placeholder)
        #self.waveform = WaveformPlaceholder()
        #main_layout.addWidget(self.waveform)

        # Seção de informações do tom
        info_card = CardFrame()
        info_layout = QVBoxLayout(info_card)

        self.label_tom = StyledLabel("", is_bold=True, size=14)
        self.label_tom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.label_tom)

        # Botões de alteração de tom com layout melhorado
        tom_buttons_layout = QHBoxLayout()
        tom_buttons_layout.setSpacing(20)
        tom_buttons_layout.setContentsMargins(20, 10, 20, 10)

        self.botao_descer = RoundedButton("↓ Baixar Tom", color=COLORS["danger"])
        self.botao_descer.clicked.connect(self.descer_tom)
        self.botao_descer.setEnabled(False)
        tom_buttons_layout.addWidget(self.botao_descer)

        self.botao_subir = RoundedButton("↑ Subir Tom", color=COLORS["accent"])
        self.botao_subir.clicked.connect(self.subir_tom)
        self.botao_subir.setEnabled(False)
        tom_buttons_layout.addWidget(self.botao_subir)



        info_layout.addLayout(tom_buttons_layout)
        main_layout.addWidget(info_card)

        self.label_lista = StyledLabel("Músicas disponíveis:", is_bold=True, size=12, color=COLORS["text_secondary"])
        main_layout.addWidget(self.label_lista)

        self.lista_musicas = QListWidget()
        self.lista_musicas.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS["card_bg"]};
                color: {COLORS["text_primary"]};
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-size: 12pt;
            }}
            QListWidget::item {{
                padding: 10px;
                margin-bottom: 5px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS["hover"]};
                color: {COLORS["primary"]};
            }}
        """)
        self.lista_musicas.setMinimumHeight(180)
        self.lista_musicas.setMaximumHeight(300)
        self.lista_musicas.itemClicked.connect(self.abrir_musica_da_lista)
        main_layout.addWidget(self.lista_musicas)

        # Seção de reprodução
        player_card = CardFrame()
        player_layout = QVBoxLayout(player_card)

        # Slider e tempo
        self.slider = ModernSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.atualizar_tempo_slider)
        self.slider.sliderPressed.connect(self.slider_pressionado)
        self.slider.sliderReleased.connect(self.slider_solto)
        player_layout.addWidget(self.slider)

        time_layout = QHBoxLayout()
        self.label_tempo = StyledLabel("0:00", color=COLORS["text_secondary"])
        self.label_duracao = StyledLabel("0:00", color=COLORS["text_secondary"])
        time_layout.addWidget(self.label_tempo)
        time_layout.addStretch()
        time_layout.addWidget(self.label_duracao)
        player_layout.addLayout(time_layout)

        # Botões de controle de reprodução
        play_layout = QHBoxLayout()
        play_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        play_layout.setSpacing(20)

        self.botao_play = RoundedButton("▶ Tocar", color=COLORS["primary"])
        self.botao_play.clicked.connect(self.tocar_ou_pausar)
        self.botao_play.setEnabled(False)
        self.botao_play.setMinimumWidth(120)
        play_layout.addWidget(self.botao_play)

        self.botao_exportar = RoundedButton("💾 Exportar", color=COLORS["secondary"])
        self.botao_exportar.clicked.connect(self.exportar_tom_atual)
        self.botao_exportar.setEnabled(False)
        play_layout.addWidget(self.botao_exportar)

        player_layout.addLayout(play_layout)

        main_layout.addWidget(player_card)

        # Seção de status de processamento
        status_processing = QHBoxLayout()
        self.loading_label = StyledLabel("", color=COLORS["primary"])
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_processing.addWidget(self.loading_label)
        main_layout.addLayout(status_processing)

        # Footer com informações
        footer_card = CardFrame()
        footer_layout = QVBoxLayout(footer_card)
        footer_text = StyledLabel(
            "Desenvolvido por Jefferson De Sousa Amorim",
            size=9,
            color=COLORS["text_secondary"]
        )
        footer_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        footer_layout.addWidget(footer_text)
        main_layout.addWidget(footer_card)

        # Timer para atualizar o slider
        self.timer = QTimer()
        self.timer.setInterval(100)  # Atualizar a cada 100ms para melhor precisão
        self.timer.timeout.connect(self.atualizar_slider)

        # Worker thread para processamento
        self.iniciar_worker_thread()

        # Animação de inicialização
        self.fadeInInterface()
        self.listar_musicas_local()

    def fadeInInterface(self):
        """Cria uma animação de fade-in para a interface"""
        self.setWindowOpacity(0)
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(800)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.start()

    def resizeEvent(self, event):
        """Assegura que todos os elementos se ajustem corretamente ao redimensionar"""
        super().resizeEvent(event)

    def animate_progress_bar(self):
        """Anima a barra de progresso com efeito fluido"""
        if self.progress_bar.maximum() == 0:  # modo indeterminado
            self._progress_anim_value += self._progress_anim_direction * 5
            if self._progress_anim_value >= 100 or self._progress_anim_value <= 0:
                self._progress_anim_direction *= -1
            self.progress_bar.setValue(self._progress_anim_value)

    def listar_musicas_local(self):
        """Atualiza a lista de músicas no diretório atual"""
        self.lista_musicas.clear()
        for arquivo in os.listdir():
            if arquivo.endswith((".mp3", ".wav")):
                self.lista_musicas.addItem(arquivo)

    def abrir_musica_da_lista(self, item):
        """Carrega o arquivo selecionado na lista"""
        caminho = item.text()
        self.carregar_arquivo_local(caminho)







def excecao_global(tipo, valor, tb):
    import traceback
    print("Exceção não capturada:", ''.join(traceback.format_exception(tipo, valor, tb)))
if __name__ == "__main__":
    import traceback

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    janela = Transpositor()
    janela.show()

    try:
        sys.exit(app.exec())
    except Exception:
        erro = traceback.format_exc()
        print("Exceção durante execução da aplicação:\n", erro)
        QMessageBox.critical(None, "Erro fatal", erro)
