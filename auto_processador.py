import os
import json
import librosa
import numpy as np
import soundfile as sf
import time
from collections import Counter

# Caminhos
CACHE_PATH = "cache_tonalidades.json"
PASTA_SAIDA = "tons_processados"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Tons a gerar: de -12 a +12
TONS = list(range(-12, 13))
NOTAS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Configurações personalizáveis para detecção de tonalidade
CONFIG = {
    # Pesos dos perfis tonais (Krumhansl & Kessler)
    "PERFIL_MAIOR": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "PERFIL_MENOR": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    # Configurações de análise
    "JANELA_ANALISE": 8,  # Tamanho da janela de análise em segundos
    "SOBREPOSICAO": 0.5,  # Sobreposição entre janelas (50%)
    "PESO_INICIO_FIM": 0.7,  # Maior peso para início e fim da música (teoria musical)
    "USAR_HARMONIC_CQT": True,  # Usar harmonic CQT para melhor precisão
    "LIMIAR_CONFIANCA": 0.15,  # Diferença mínima entre correlações para considerar confiável
}

# Garante que o cache seja criado mesmo vazio
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def detectar_tonalidade_aprimorado(y, sr, mostrar_debug=False):
    """Algoritmo aprimorado de detecção de tonalidade com análise segmentada, pesos e heurísticas"""

    # Extrai chromagrama com CQT harmônico (mais robusto a percussão)
    if CONFIG["USAR_HARMONIC_CQT"]:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512,
                                            bins_per_octave=36, n_octaves=5,
                                            fmin=librosa.note_to_hz('C2'))
    else:
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)

    hop_samples = 512
    amostras_por_janela = int(CONFIG["JANELA_ANALISE"] * sr / hop_samples)
    passo = int(amostras_por_janela * (1 - CONFIG["SOBREPOSICAO"]))

    janelas = []
    for i in range(0, chroma.shape[1], passo):
        fim = min(i + amostras_por_janela, chroma.shape[1])
        if fim - i >= amostras_por_janela // 2:
            janelas.append(chroma[:, i:fim])

    perfil_maior = (CONFIG["PERFIL_MAIOR"] - np.mean(CONFIG["PERFIL_MAIOR"])) / np.std(CONFIG["PERFIL_MAIOR"])
    perfil_menor = (CONFIG["PERFIL_MENOR"] - np.mean(CONFIG["PERFIL_MENOR"])) / np.std(CONFIG["PERFIL_MENOR"])

    resultados = []

    for idx, janela in enumerate(janelas):
        chroma_mean = np.mean(janela, axis=1)
        chroma_norm = (chroma_mean - np.mean(chroma_mean)) / (np.std(chroma_mean) + 1e-8)

        corr_maior = np.array([np.corrcoef(np.roll(perfil_maior, i), chroma_norm)[0, 1] for i in range(12)])
        corr_menor = np.array([np.corrcoef(np.roll(perfil_menor, i), chroma_norm)[0, 1] for i in range(12)])

        idx_maj = np.argmax(corr_maior)
        idx_min = np.argmax(corr_menor)
        val_maj = corr_maior[idx_maj]
        val_min = corr_menor[idx_min]

        peso = 2.0 if idx in [0, len(janelas) - 1] else 1.0

        if val_maj > val_min:
            tonalidade = f"{NOTAS[idx_maj]} Maior"
            confianca = (val_maj - val_min) * peso
        else:
            tonalidade = f"{NOTAS[idx_min]} Menor"
            confianca = (val_min - val_maj) * peso

        resultados.append({
            "tonalidade": tonalidade,
            "confianca": confianca,
            "posicao": f"{idx + 1}/{len(janelas)}"
        })

    resultados_validos = [r for r in resultados if r["confianca"] > CONFIG["LIMIAR_CONFIANCA"]]
    if not resultados_validos:
        resultados_validos = resultados  # usa todos como fallback

    votos = {}
    for r in resultados_validos:
        votos[r["tonalidade"]] = votos.get(r["tonalidade"], 0) + r["confianca"]

    # Heurística por padrão harmônico comum
    contagem_c = sum(1 for r in resultados_validos if r["tonalidade"].startswith("C") or "Am" in r["tonalidade"])
    contagem_g = sum(1 for r in resultados_validos if r["tonalidade"].startswith("G") or "Em" in r["tonalidade"])
    reforco_c_ou_g = None
    if abs(contagem_c - contagem_g) >= 3:
        reforco_c_ou_g = "C Maior" if contagem_c > contagem_g else "G Maior"
        votos[reforco_c_ou_g] = votos.get(reforco_c_ou_g, 0) + 2.5  # reforça com peso artificial

    tonalidade_final = max(votos.items(), key=lambda x: x[1])[0]

    if mostrar_debug:
        print("---- ANÁLISE DE TONALIDADE ----")
        for r in resultados:
            print(f"{r['posicao']}: {r['tonalidade']} (confiança: {r['confianca']:.2f})")
        print("\nDistribuição ponderada:")
        for tom, peso in votos.items():
            print(f"  {tom}: {peso:.2f}")
        print(f"→ Tonalidade final: {tonalidade_final}")
        print("------------------------------")

    return tonalidade_final



def detectar_tonalidade_krumhansl(y, sr):
    """Método original para compatibilidade"""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    chroma_norm = (chroma_mean - np.mean(chroma_mean)) / np.std(chroma_mean)
    major_profile = (major_profile - np.mean(major_profile)) / np.std(major_profile)
    minor_profile = (minor_profile - np.mean(minor_profile)) / np.std(minor_profile)

    def correlacao_rotacionada(perfil):
        return [np.dot(np.roll(perfil, i), chroma_norm) for i in range(12)]

    corr_maj = correlacao_rotacionada(major_profile)
    corr_min = correlacao_rotacionada(minor_profile)

    max_maj = int(np.argmax(corr_maj))
    max_min = int(np.argmax(corr_min))

    if max(corr_maj) > max(corr_min):
        return f"{NOTAS[max_maj]} Maior"
    else:
        return f"{NOTAS[max_min]} Menor"


def tons_estao_completos(nome_arquivo):
    base_nome = os.path.splitext(os.path.basename(nome_arquivo))[0]
    pasta_destino = os.path.join(PASTA_SAIDA, base_nome)
    if not os.path.exists(pasta_destino):
        return False
    for semitons in TONS:
        nome_esperado = os.path.join(pasta_destino, f"{base_nome}_{semitons:+}.wav")
        if not os.path.exists(nome_esperado):
            return False
    return True


def processar_arquivo(nome_arquivo, modo="avancado"):
    try:
        print(f"\n🎧 Processando: {nome_arquivo}")
        y, sr = librosa.load(nome_arquivo, sr=None, mono=True)

        # Usa o método original para comparação
        tonalidade_original = detectar_tonalidade_krumhansl(y, sr)

        # Usa o método avançado para detecção mais precisa
        tonalidade_avancada = detectar_tonalidade_aprimorado(y, sr, mostrar_debug=CONFIG.get("DEBUG", False))

        print(f"→ Tonalidade (método original): {tonalidade_original}")
        print(f"→ Tonalidade (método avançado): {tonalidade_avancada}")

        # Escolhe qual tonalidade usar com base no modo selecionado
        tonalidade = tonalidade_avancada if modo == "avancado" else tonalidade_original

        base_nome = os.path.splitext(os.path.basename(nome_arquivo))[0]
        pasta_destino = os.path.join(PASTA_SAIDA, base_nome)
        os.makedirs(pasta_destino, exist_ok=True)

        for semitons in TONS:
            try:
                print(f"  [INFO] Gerando tom {semitons:+}...", end=" ")
                y_mod = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitons)
                if y_mod is not None and isinstance(y_mod, np.ndarray) and len(y_mod) > 0 and not np.any(np.isnan(y_mod)) and not np.any(np.isinf(y_mod)):
                    saida = os.path.join(pasta_destino, f"{base_nome}_{semitons:+}.wav")
                    sf.write(saida, y_mod, sr)
                    print("✓")
                else:
                    print("⚠ Inválido")
            except Exception as e:
                print(f"[X] Erro no tom {semitons:+}: {e}")

        # Salva tonalidade detectada no cache com nome do arquivo
        cache[os.path.basename(nome_arquivo)] = {
            "tonalidade_original": tonalidade_original,
            "tonalidade_avancada": tonalidade_avancada,
            "tonalidade_escolhida": tonalidade
        }

        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"[OK] Cache atualizado para {nome_arquivo}")

    except Exception as e:
        print(f"[ERRO] {nome_arquivo}: {e}")


def iniciar_processamento_sequencial(modo="avancado"):
    arquivos = [f for f in os.listdir() if f.lower().endswith(('.mp3', '.wav'))]
    for f in arquivos:
        if f not in cache or not tons_estao_completos(f):
            processar_arquivo(f, modo)
            time.sleep(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detector e processador de tonalidades musicais")
    parser.add_argument("--modo", choices=["original", "avancado"], default="avancado",
                        help="Modo de detecção: 'original' (padrão antigo) ou 'avancado' (novo algoritmo)")
    parser.add_argument("--arquivo", help="Especificar um único arquivo para processamento")
    parser.add_argument("--debug", action="store_true", help="Mostrar informações detalhadas de debug")

    args = parser.parse_args()

    # Ajusta configurações de debug
    CONFIG["DEBUG"] = args.debug

    if args.arquivo:
        if os.path.exists(args.arquivo):
            processar_arquivo(args.arquivo, args.modo)
        else:
            print(f"Arquivo não encontrado: {args.arquivo}")
    else:
        iniciar_processamento_sequencial(args.modo)
        print("\n✅ Finalizado. Todos os arquivos foram processados.")