import os
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# =========================
# CONFIGURAÇÕES DO BANCO
# =========================
USUARIO = 'postgres'
SENHA = quote_plus('Fl@qu1nh@')
HOST = 'localhost'
PORTA = '5433'
BANCO = 'chatbotGI2026'
SCHEMA = 'Schemabot'
TABELA = 'dados'

# =========================
# CAMINHO DO ARQUIVO EXCEL
# =========================
caminho = r'C:\Users\sophi\OneDrive\Desktop\Estágio_Obrigatório\FONTES\PERGUNTAS_RESPOSTAS.xlsx'

# =========================
# VERIFICA SE O ARQUIVO EXISTE
# =========================
print('Caminho informado:')
print(caminho)

if not os.path.exists(caminho):
    print(f'❌ Arquivo não encontrado: {caminho}')
    raise SystemExit

# =========================
# LÊ O ARQUIVO EXCEL
# =========================
df = pd.read_excel(caminho)

# Padroniza nomes das colunas
df.columns = [c.strip().lower() for c in df.columns]

print('\n✅ Arquivo Excel lido com sucesso!')
print('Colunas encontradas:', list(df.columns))
print(f'Total de registros: {len(df)}')

# =========================
# CRIA A CONEXÃO COM O POSTGRESQL
# =========================
url_conexao = f'postgresql+psycopg2://{USUARIO}:{SENHA}@{HOST}:{PORTA}/{BANCO}'
engine = create_engine(url_conexao)

# =========================
# TESTA A CONEXÃO E CRIA O SCHEMA
# =========================
try:
    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
        conn.commit()
        print('\n✅ Conexão com PostgreSQL realizada com sucesso!')
        print(f'✅ Schema "{SCHEMA}" verificado/criado com sucesso!')
except Exception as e:
    print('\n❌ Erro ao conectar no PostgreSQL:')
    print(e)
    raise SystemExit

# =========================
# ENVIA OS DADOS PARA A TABELA
# =========================
try:
    df.to_sql(
        TABELA,
        engine,
        schema=SCHEMA,
        if_exists='append',
        index=False
    )
    print(f'\n✅ Dados carregados com sucesso na tabela "{SCHEMA}"."{TABELA}"!')
except Exception as e:
    print('\n❌ Erro ao enviar os dados para o PostgreSQL:')
    print(e)