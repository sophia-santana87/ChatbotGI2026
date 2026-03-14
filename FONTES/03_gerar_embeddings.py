import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from urllib.parse import quote_plus

# =========================
# CONFIGURAÇÕES DO BANCO
# =========================
USUARIO = "postgres"
SENHA = "Fl@qu1nh@"
HOST = "localhost"
PORTA = "5433"
BANCO = "chatbotGI2026"
SCHEMA = "Schemabot"
TABELA = "dados"

# =========================
# MODELO DE EMBEDDING
# =========================
# Modelo multilíngue bom para português
modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# =========================
# CONEXÃO COM O POSTGRESQL
# =========================
conn = psycopg2.connect(
    dbname=BANCO,
    user=USUARIO,
    password=SENHA,
    host=HOST,
    port=PORTA
)

register_vector(conn)

cur = conn.cursor()

# Busca registros sem embedding
cur.execute(f'''
    SELECT id, pergunta, eixo
    FROM "{SCHEMA}"."{TABELA}"
    WHERE embedding IS NULL
    ORDER BY id
''')

registros = cur.fetchall()

if not registros:
    print("✅ Todos os registros já possuem embedding.")
    cur.close()
    conn.close()
    raise SystemExit

print(f"Total de registros sem embedding: {len(registros)}")

for registro in registros:
    id_registro, pergunta, eixo = registro

    texto_para_embedding = f"eixo: {eixo}\npergunta: {pergunta}"

    vetor = modelo.encode(texto_para_embedding, normalize_embeddings=True).tolist()

    cur.execute(
        f'''
        UPDATE "{SCHEMA}"."{TABELA}"
        SET embedding = %s
        WHERE id = %s
        ''',
        (vetor, id_registro)
    )

    print(f"Embedding gerado para id={id_registro}")

conn.commit()
cur.close()
conn.close()

print("\n✅ Embeddings gerados e salvos com sucesso!")