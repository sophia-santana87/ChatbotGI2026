import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# =========================
# CONFIGURAÇÕES DO BANCO
# =========================
USUARIO = "postgres"
SENHA = quote_plus("Fl@qu1nh@")
HOST = "localhost"
PORTA = "5433"
BANCO = "chatbotGI2026"
SCHEMA = "Schemabot"
TABELA = "dados"

# =========================
# CAMINHO DA PLANILHA
# =========================
caminho = r"C:\Users\sophi\OneDrive\Desktop\Estágio_Obrigatório\FONTES\PERGUNTAS_RESPOSTAS.xlsx"

if not os.path.exists(caminho):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

# =========================
# LEITURA DA PLANILHA
# =========================
df = pd.read_excel(caminho)

# Padroniza nomes
df.columns = [col.strip().upper() for col in df.columns]

# Confere colunas esperadas
colunas_esperadas = {"TIPO", "PERGUNTA", "RESPOSTA"}
if not colunas_esperadas.issubset(df.columns):
    raise ValueError(f"Colunas encontradas: {list(df.columns)}. Esperado: {colunas_esperadas}")

# Renomeia para o padrão do banco
df = df.rename(columns={
    "TIPO": "eixo",
    "PERGUNTA": "pergunta",
    "RESPOSTA": "resposta"
})

# Remove linhas vazias
df = df.dropna(subset=["pergunta", "resposta"]).copy()

# Limpeza básica
df["pergunta"] = df["pergunta"].astype(str).str.strip()
df["resposta"] = df["resposta"].astype(str).str.strip()
df["eixo"] = df["eixo"].fillna("").astype(str).str.strip()

# Remove duplicatas
df = df.drop_duplicates(subset=["pergunta"]).reset_index(drop=True)

print("Prévia dos dados:")
print(df.head())
print(f"\nTotal de registros para importar: {len(df)}")

# =========================
# CONEXÃO COM O BANCO
# =========================
url = f"postgresql+psycopg2://{USUARIO}:{SENHA}@{HOST}:{PORTA}/{BANCO}"
engine = create_engine(url)

# =========================
# ENVIO PARA O BANCO
# =========================
df.to_sql(
    TABELA,
    engine,
    schema=SCHEMA,
    if_exists="append",
    index=False
)

print('\n✅ Dados importados com sucesso para "Schemabot"."dados"!')