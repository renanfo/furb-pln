#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho 2 - Comparação Simples de Modelos
Comparação direta entre BERT, TF-IDF e Word2Vec
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# PLN
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import string

# Modelos
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch

# Configuração
DATABASE_DIR = "./database/trabalho2"

class ProcessadorSimples:
    """Processador básico para preparar textos."""
    
    def __init__(self):
        # Setup NLTK
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('portuguese'))
        print("✅ ProcessadorSimples inicializado")
    
    def processar_texto(self, texto, url="", titulo=""):
        """Processa um texto: tokenização + normalização."""
        try:
            # Tokenização
            tokens = word_tokenize(texto.lower(), language='portuguese')
            
            # Normalização
            tokens_limpos = []
            for token in tokens:
                if (token not in string.punctuation and 
                    not token.isdigit() and 
                    len(token) > 2 and
                    token not in self.stop_words and
                    any(c.isalpha() for c in token)):
                    tokens_limpos.append(token)
            
            return {
                'id': url,
                'titulo': titulo,
                'texto_bruto': texto,
                'tokens_normalizados': tokens_limpos,
                'num_tokens': len(tokens_limpos)
            }
        except Exception as e:
            print(f"❌ Erro ao processar: {e}")
            return {
                'id': url, 'titulo': titulo, 'texto_bruto': texto,
                'tokens_normalizados': [], 'num_tokens': 0
            }

class ComparadorSimples:
    """Comparação simples entre BERT, TF-IDF e Word2Vec."""
    
    def __init__(self):
        print("🚀 Inicializando ComparadorSimples...")
        
        # BERT
        self.bert_model_name = "neuralmind/bert-base-portuguese-cased"
        self.bert_tokenizer = None
        self.bert_model = None
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        print("✅ ComparadorSimples inicializado")
    
    def _carregar_bert(self):
        """Carrega BERT se necessário."""
        if self.bert_tokenizer is None:
            print("📥 Carregando BERT...")
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
                self.bert_model.eval()
                print("✅ BERT português carregado")
            except Exception:
                print("⚠️  Usando BERT multilingual...")
                self.bert_model_name = "bert-base-multilingual-cased"
                self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
                self.bert_model.eval()
                print("✅ BERT multilingual carregado")
    
    def processar_tfidf(self, textos):
        """Processa com TF-IDF."""
        print(f"🔄 TF-IDF: processando {len(textos)} textos...")
        try:
            matriz_tfidf = self.tfidf_vectorizer.fit_transform(textos)
            embeddings = matriz_tfidf.toarray()
            print(f"✅ TF-IDF concluído: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"❌ Erro TF-IDF: {e}")
            return np.zeros((len(textos), 1000))
    
    def processar_word2vec(self, tokens_lista):
        """Processa com Word2Vec."""
        print(f"🔄 Word2Vec: treinando com {len(tokens_lista)} documentos...")
        try:
            model = Word2Vec(
                sentences=tokens_lista,
                vector_size=300,
                window=5,
                min_count=2,
                workers=4,
                epochs=10
            )
            
            # Embeddings médios por documento
            embeddings = []
            for tokens in tokens_lista:
                if not tokens:
                    embeddings.append(np.zeros(300))
                    continue
                
                vetores = []
                for token in tokens:
                    if token in model.wv:
                        vetores.append(model.wv[token])
                
                if vetores:
                    embeddings.append(np.mean(vetores, axis=0))
                else:
                    embeddings.append(np.zeros(300))
            
            embeddings = np.array(embeddings)
            print(f"✅ Word2Vec concluído: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Erro Word2Vec: {e}")
            return np.zeros((len(tokens_lista), 300))
    
    def processar_bert(self, textos):
        """Processa com BERT."""
        self._carregar_bert()
        
        print(f"🔄 BERT: processando {len(textos)} textos...")
        embeddings = []
        
        for i, texto in enumerate(textos):
            try:
                inputs = self.bert_tokenizer(
                    texto, return_tensors="pt", truncation=True, 
                    padding=True, max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.append(embedding.flatten())
                
                if (i + 1) % 3 == 0:
                    print(f"  📈 {i + 1}/{len(textos)} processados")
                    
            except Exception as e:
                print(f"⚠️  Erro no texto {i}: {e}")
                embeddings.append(np.zeros(768))
        
        embeddings = np.array(embeddings)
        print(f"✅ BERT concluído: {embeddings.shape}")
        return embeddings
    
    def calcular_similaridades(self, embeddings, nome_modelo):
        """Calcula estatísticas de similaridade."""
        try:
            matriz_sim = cosine_similarity(embeddings)
            
            # Similaridades (excluindo diagonal)
            mask = ~np.eye(matriz_sim.shape[0], dtype=bool)
            similaridades = matriz_sim[mask]
            
            return {
                'modelo': nome_modelo,
                'dimensoes': embeddings.shape[1],
                'num_documentos': embeddings.shape[0],
                'similaridade_media': np.mean(similaridades),
                'similaridade_std': np.std(similaridades),
                'similaridade_min': np.min(similaridades),
                'similaridade_max': np.max(similaridades)
            }
        except Exception as e:
            return {'modelo': nome_modelo, 'erro': str(e)}
    
    def comparar_modelos(self, dados_processados):
        """Executa comparação completa entre os três modelos."""
        print("\n" + "="*60)
        print("🎯 COMPARAÇÃO SIMPLES DE MODELOS")
        print("="*60)
        
        if not dados_processados:
            print("❌ Nenhum dado para processar")
            return {}
        
        # Extrair dados
        textos_brutos = [item.get('texto_bruto', '') for item in dados_processados]
        tokens_normalizados = [item.get('tokens_normalizados', []) for item in dados_processados]
        
        # Filtrar dados válidos
        indices_validos = [i for i, texto in enumerate(textos_brutos) if texto.strip()]
        textos_validos = [textos_brutos[i] for i in indices_validos]
        tokens_validos = [tokens_normalizados[i] for i in indices_validos]
        
        print(f"📋 Processando {len(textos_validos)} documentos válidos")
        
        resultados = {}
        
        # 1. TF-IDF
        print("\n🔹 TF-IDF")
        embeddings_tfidf = self.processar_tfidf(textos_validos)
        resultados['tfidf'] = self.calcular_similaridades(embeddings_tfidf, "TF-IDF")
        
        # 2. Word2Vec
        print("\n🔹 Word2Vec")
        embeddings_w2v = self.processar_word2vec(tokens_validos)
        resultados['word2vec'] = self.calcular_similaridades(embeddings_w2v, "Word2Vec")
        
        # 3. BERT
        print("\n🔹 BERT")
        embeddings_bert = self.processar_bert(textos_validos)
        resultados['bert'] = self.calcular_similaridades(embeddings_bert, "BERT")
        
        print("\n✅ Comparação concluída!")
        return resultados

def carregar_dados():
    """Carrega dados já processados."""
    print("📂 Carregando dados existentes...")
    
    # Tentar CSV primeiro
    csv_path = os.path.join(DATABASE_DIR, "dados_processados.csv")
    if os.path.exists(csv_path):
        print(f"📄 Carregando CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        dados = []
        for _, row in df.iterrows():
            tokens = []
            if pd.notna(row.get('Tokens_Normalizados')):
                tokens = str(row['Tokens_Normalizados']).split()
            
            dados.append({
                'id': row.get('ID', ''),
                'titulo': f"Post {len(dados) + 1}",
                'texto_bruto': str(row.get('Texto_Bruto', '')),
                'tokens_normalizados': tokens
            })
        
        print(f"✅ {len(dados)} documentos carregados do CSV")
        return dados
    
    # Tentar arquivos .txt
    if os.path.exists(DATABASE_DIR):
        txt_files = [f for f in os.listdir(DATABASE_DIR) if f.endswith('.txt')]
        
        if txt_files:
            print(f"📁 Processando {len(txt_files)} arquivos .txt...")
            
            processador = ProcessadorSimples()
            dados = []
            
            for filename in txt_files[:10]:  # Limitar para teste
                filepath = os.path.join(DATABASE_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        conteudo = f.read()
                    
                    if len(conteudo.strip()) < 100:
                        continue
                    
                    resultado = processador.processar_texto(
                        conteudo, filename, filename.replace('.txt', '')
                    )
                    dados.append(resultado)
                    
                except Exception as e:
                    print(f"⚠️  Erro em {filename}: {e}")
                    continue
            
            print(f"✅ {len(dados)} documentos processados")
            return dados
    
    print("❌ Nenhum dado encontrado!")
    return None

def gerar_relatorio(resultados):
    """Gera relatório simples."""
    if not resultados:
        return "❌ Sem resultados"
    
    relatorio = []
    relatorio.append("="*60)
    relatorio.append("📊 RELATÓRIO COMPARATIVO")
    relatorio.append("="*60)
    
    # Tabela resumo
    relatorio.append("\n📈 RESULTADOS:")
    relatorio.append("-" * 60)
    
    header = f"{'Modelo':<12} {'Dimensões':<10} {'Sim.Média':<10} {'Desvio':<8} {'Min':<8} {'Max':<8}"
    relatorio.append(header)
    relatorio.append("-" * 60)
    
    for dados in resultados.values():
        if 'erro' not in dados:
            linha = f"{dados['modelo']:<12} {dados['dimensoes']:<10} {dados['similaridade_media']:<10.3f} {dados['similaridade_std']:<8.3f} {dados['similaridade_min']:<8.3f} {dados['similaridade_max']:<8.3f}"
            relatorio.append(linha)
    
    relatorio.append("\n🔍 INTERPRETAÇÃO:")
    relatorio.append("-" * 30)
    
    for dados in resultados.values():
        if 'erro' not in dados:
            sim = dados['similaridade_media']
            relatorio.append(f"\n{dados['modelo']}:")
            relatorio.append(f"  📏 Dimensões: {dados['dimensoes']}")
            relatorio.append(f"  🎯 Similaridade: {sim:.3f}")
            
            if sim > 0.5:
                interp = "🔴 Alta - Documentos muito parecidos"
            elif sim > 0.3:
                interp = "🟡 Moderada - Alguns padrões comuns"
            else:
                interp = "🟢 Baixa - Documentos diversos"
            
            relatorio.append(f"  💡 {interp}")
    
    relatorio.append("\n" + "="*60)
    return "\n".join(relatorio)

def plotar_comparacao(resultados):
    """Plota gráfico simples."""
    if not resultados:
        print("❌ Sem dados para plotar")
        return
    
    modelos = []
    similaridades = []
    dimensoes = []
    
    for dados in resultados.values():
        if 'erro' not in dados:
            modelos.append(dados['modelo'])
            similaridades.append(dados['similaridade_media'])
            dimensoes.append(dados['dimensoes'])
    
    if not modelos:
        print("❌ Nenhum resultado válido")
        return
    
    # Criar gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Similaridade
    bars1 = ax1.bar(modelos, similaridades, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('📊 Similaridade Média por Modelo', fontweight='bold')
    ax1.set_ylabel('Similaridade Média')
    ax1.set_ylim(0, 1)
    
    for bar, value in zip(bars1, similaridades):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', fontweight='bold')
    
    # Dimensões
    bars2 = ax2.bar(modelos, dimensoes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('📏 Dimensões por Modelo', fontweight='bold')
    ax2.set_ylabel('Número de Dimensões')
    
    for bar, value in zip(bars2, dimensoes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dimensoes)*0.02,
                f'{value}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    return fig

def main():
    """Função principal - executa comparação completa."""
    print("🚀 EXECUTANDO COMPARAÇÃO SIMPLES")
    print("="*50)
    
    # 1. Carregar dados
    dados = carregar_dados()
    if not dados:
        return None
    
    print(f"\n📋 Dados: {len(dados)} documentos")
    
    # 2. Comparar modelos
    comparador = ComparadorSimples()
    resultados = comparador.comparar_modelos(dados)
    
    if not resultados:
        print("❌ Falha na comparação")
        return None
    
    # 3. Gerar relatório
    relatorio = gerar_relatorio(resultados)
    print("\n" + relatorio)
    
    # 4. Plotar gráficos
    print("\n📈 Gerando gráficos...")
    fig = plotar_comparacao(resultados)
    
    # 5. Salvar relatório
    try:
        caminho = os.path.join(DATABASE_DIR, "relatorio_comparacao_simples.txt")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(relatorio)
        print(f"\n💾 Relatório salvo: {caminho}")
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")
    
    print("\n✅ Comparação concluída!")
    return resultados, relatorio, fig

if __name__ == "__main__":
    main()
