import re
import unicodedata
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# =========================
# CONFIGURAÇÕES
# =========================
USUARIO = "postgres"
SENHA = "Fl@qu1nh@"
HOST = "localhost"
PORTA = "5433"
BANCO = "chatbotGI2026"
SCHEMA = "Schemabot"
TABELA = "dados"

MODEL_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_VETOR = 5
TOP_K_FINAL = 5

# pesos do score híbrido
PESO_TEXTO = 0.65
PESO_VETOR = 0.35

# limiares
LIMIAR_RESPOSTA_DIRETA = 45.0
LIMIAR_MINIMO = 25.0

# =========================
# CARREGAR MODELO
# =========================
print("Carregando modelo de embeddings...")
modelo = SentenceTransformer(MODEL_EMBEDDING)
print("✅ Modelo carregado.")

# =========================
# CONEXÃO COM O BANCO
# =========================
print("Conectando ao PostgreSQL...")
conn = psycopg2.connect(
    dbname=BANCO,
    user=USUARIO,
    password=SENHA,
    host=HOST,
    port=PORTA
)
register_vector(conn)
cur = conn.cursor()
print("✅ Conexão com PostgreSQL OK.")

# =========================
# NORMALIZAÇÃO
# =========================
def remover_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

def limpar_texto(texto: str) -> str:
    texto = texto.lower().strip()
    texto = texto.replace("?", " ")
    texto = texto.replace("!", " ")
    texto = texto.replace(".", " ")
    texto = texto.replace(",", " ")
    texto = texto.replace(";", " ")
    texto = texto.replace(":", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def expandir_siglas(texto: str) -> str:
    texto = limpar_texto(texto)

    substituicoes = {
        "oque": "o que",
        "oq": "o que",
        "g i": "gestão da informação",
        "gi": "gestão da informação",
        "gestao da informacao": "gestão da informação",
        "curso gi": "curso de gestão da informação",
        "gestor da informacao": "gestor da informação",
        "ciencia da informacao": "ciência da informação",
    }

    texto = f" {texto} "

    for origem, destino in substituicoes.items():
        texto = re.sub(rf"\b{re.escape(origem)}\b", destino, texto)

    return re.sub(r"\s+", " ", texto).strip()

def normalizar_para_comparacao(texto: str) -> str:
    texto = expandir_siglas(texto)
    texto = remover_acentos(texto)
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def tokenizar(texto: str):
    texto = normalizar_para_comparacao(texto)
    return [t for t in texto.split() if len(t) > 1]

# =========================
# BUSCA TEXTUAL EM MEMÓRIA
# =========================
def carregar_base():
    cur.execute(
        f'''
        SELECT id, pergunta, resposta, eixo
        FROM "{SCHEMA}"."{TABELA}"
        '''
    )
    return cur.fetchall()

BASE = carregar_base()

def score_textual(pergunta_usuario: str, pergunta_bd: str) -> float:
    usuario_norm = normalizar_para_comparacao(pergunta_usuario)
    bd_norm = normalizar_para_comparacao(pergunta_bd)

    tokens_usuario = set(tokenizar(pergunta_usuario))
    tokens_bd = set(tokenizar(pergunta_bd))

    score = 0.0

    # igualdade exata
    if usuario_norm == bd_norm:
        score += 100.0

    # containment
    if usuario_norm and usuario_norm in bd_norm:
        score += 50.0

    if bd_norm and bd_norm in usuario_norm:
        score += 35.0

    # sobreposição de tokens
    if tokens_usuario and tokens_bd:
        inter = len(tokens_usuario & tokens_bd)
        uniao_base = max(len(tokens_usuario), 1)
        score += (inter / uniao_base) * 50.0

    # bônus semântico/manual para perguntas básicas
    if "gestao da informacao" in usuario_norm and "gestao da informacao" in bd_norm:
        score += 35.0

    if "gestor da informacao" in usuario_norm and "gestor da informacao" in bd_norm:
        score += 25.0

    if "materias" in usuario_norm and ("materias" in bd_norm or "estuda" in bd_norm):
        score += 20.0

    if "curso" in usuario_norm and "curso" in bd_norm:
        score += 10.0

    return min(score, 100.0)

def buscar_textual(pergunta_usuario: str):
    resultados = []

    for id_reg, pergunta_bd, resposta_bd, eixo_bd in BASE:
        st = score_textual(pergunta_usuario, pergunta_bd)

        if st > 0:
            resultados.append({
                "id": id_reg,
                "pergunta": pergunta_bd,
                "resposta": resposta_bd,
                "eixo": eixo_bd,
                "score_texto": st,
                "score_vetor": 0.0,
                "score_final": 0.0
            })

    resultados.sort(key=lambda x: x["score_texto"], reverse=True)
    return resultados[:TOP_K_FINAL]

# =========================
# BUSCA VETORIAL
# =========================
def buscar_vetorial(pergunta_usuario: str):
    pergunta_expandida = expandir_siglas(pergunta_usuario)

    vetor = modelo.encode(
        pergunta_expandida,
        normalize_embeddings=True
    ).astype(np.float32)

    cur.execute(
        f"""
        SELECT
            id,
            pergunta,
            resposta,
            eixo,
            embedding <=> %s::vector AS distancia
        FROM "{SCHEMA}"."{TABELA}"
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vetor, vetor, TOP_K_VETOR)
    )

    rows = cur.fetchall()

    resultados = []
    for id_reg, pergunta_bd, resposta_bd, eixo_bd, distancia in rows:
        similaridade = (1 - float(distancia)) * 100

        resultados.append({
            "id": id_reg,
            "pergunta": pergunta_bd,
            "resposta": resposta_bd,
            "eixo": eixo_bd,
            "score_texto": 0.0,
            "score_vetor": similaridade,
            "score_final": 0.0
        })

    return resultados

# =========================
# FUSÃO HÍBRIDA
# =========================
def fundir_resultados(resultados_texto, resultados_vetor):
    mapa = {}

    for item in resultados_texto:
        mapa[item["id"]] = item.copy()

    for item in resultados_vetor:
        if item["id"] not in mapa:
            mapa[item["id"]] = item.copy()
        else:
            mapa[item["id"]]["score_vetor"] = item["score_vetor"]

    for item in mapa.values():
        item["score_final"] = (
            (item["score_texto"] * PESO_TEXTO) +
            (item["score_vetor"] * PESO_VETOR)
        )

    resultados = list(mapa.values())
    resultados.sort(key=lambda x: x["score_final"], reverse=True)
    return resultados[:TOP_K_FINAL]

# =========================
# RESPOSTA
# =========================
def responder(pergunta_usuario: str):
    pergunta_limpa = pergunta_usuario.strip()

    if not pergunta_limpa:
        print("🤖 Assistente: Digite uma pergunta válida.")
        return

    resultados_texto = buscar_textual(pergunta_limpa)
    resultados_vetor = buscar_vetorial(pergunta_limpa)
    resultados = fundir_resultados(resultados_texto, resultados_vetor)

    if not resultados:
        print("\n🤖 Assistente: Desculpe, não encontrei nenhuma informação relacionada no banco de dados.")
        return

    melhor = resultados[0]
    confianca = melhor["score_final"]

    # regra principal:
    # se texto bateu forte OU score final razoável -> responde direto
    if melhor["score_texto"] >= 55 or confianca >= LIMIAR_RESPOSTA_DIRETA:
        print(f"\nPergunta: {pergunta_usuario}")
        print(f"\n🤖 Assistente: {melhor['resposta']}")
        print(f"✅ [Confiança: {confianca:.1f}% | Fonte: {melhor['pergunta']}]")
        return

    # se ainda não foi muito bem, mas há algum sinal
    if confianca >= LIMIAR_MINIMO:
        print(f"\nPergunta: {pergunta_usuario}")
        print(f"\n🤖 Assistente: {melhor['resposta']}")
        print(f"⚠️ [Confiança: {confianca:.1f}% | Fonte: {melhor['pergunta']}]")
        return

    print(f"\nPergunta: {pergunta_usuario}")
    print("\n🤖 Assistente: Desculpe, não tenho essa informação específica no meu banco de dados atual sobre o curso.")
    print(f"⚠️ [Confiança: {confianca:.1f}% | Fonte mais próxima: {melhor['pergunta']}]")

# =========================
# LOOP PRINCIPAL
# =========================
print("\n=== Chatbot GI UFPR iniciado ===")
print("Digite sua pergunta.")
print("Para encerrar, digite: sair\n")

try:
    while True:
        pergunta_usuario = input("Você: ").strip()

        if pergunta_usuario.lower() in ["sair", "exit", "quit"]:
            print("🤖 Assistente: Encerrando. Até mais!")
            break

        responder(pergunta_usuario)
        print()

except KeyboardInterrupt:
    print("\n🤖 Assistente: Encerrado pelo usuário.")

finally:
    cur.close()
    conn.close()