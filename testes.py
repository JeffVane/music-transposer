import re
import unicodedata

def normalizar_nome_musica(titulo):
    # 1. Normaliza usando NFC para preservar acentos e símbolos visuais
    titulo = unicodedata.normalize("NFC", titulo)

    # 2. Remove apenas os caracteres realmente proibidos
    titulo = re.sub(r'[<>:"/\\|?*\n\r\t]', '', titulo)

    # 3. Remove espaços duplos
    titulo = re.sub(r'\s+', ' ', titulo).strip()

    return titulo


# Testes
titulos = [
    "Felipe Rodrigues - Toca Em Mim De Novo  Ministração (Ao Vivo)",
    "Alexandre Aposan - Moisés ft. Coral Resgate",
    "YESHUA • BRASAS ｜ THEO RUBIA ｜ FELIPE RODRIGUES (Ao Vivo)",
    "Canção nova: Adoração | Louvor <Oficial>",
    "Exemplo com /barra, \\contra barra e *asterisco?",
]

for t in titulos:
    print("Original :", t)
    print("Normalizado:", normalizar_nome_musica(t))
    print("-" * 50)
