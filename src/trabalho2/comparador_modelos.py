import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
import time
import warnings
warnings.filterwarnings('ignore')

class ComparadorModelos:
    """
    Classe para comparar diferentes técnicas de representação textual:
    - BERT (Transformers)
    - TF-IDF (Sklearn)
    - Word2Vec (Gensim)
    
    Baseado na estrutura da apresentação para fazer análise comparativa.
    """
    
    def __init__(self):
        """Inicializa o comparador com os modelos."""
        print("Inicializando ComparadorModelos...")
        
        self.bert_model_name = "neuralmind/bert-base-portuguese-cased"
        self.bert_tokenizer = None
        self.bert_model = None
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.word2vec_model = None
        
        self.resultados_comparacao = {}
        
        print("ComparadorModelos inicializado")
    
    def _carregar_bert(self):
        """Carrega o modelo BERT português."""
        if self.bert_tokenizer is None or self.bert_model is None:
            print("Carregando modelo BERT...")
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
                self.bert_model.eval()
                print("BERT carregado com sucesso")
            except Exception as e:
                print(f"Erro ao carregar BERT: {e}")
                print("Tentando modelo alternativo...")
                self.bert_model_name = "bert-base-multilingual-cased"
                self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
                self.bert_model.eval()
                print("BERT multilingual carregado")
    
    def processar_textos_bert(self, textos: list) -> np.ndarray:
        """
        Processa textos usando BERT e retorna embeddings.
        """
        self._carregar_bert()
        
        print(f"Processando {len(textos)} textos com BERT...")
        embeddings = []
        
        inicio = time.time()
        
        for i, texto in enumerate(textos):
            try:
                inputs = self.bert_tokenizer(
                    texto, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.append(embedding.flatten())
                
                if (i + 1) % 10 == 0:
                    print(f"BERT: {i + 1}/{len(textos)} processados")
                    
            except Exception as e:
                print(f"Erro no texto {i}: {e}")
                embeddings.append(np.zeros(768))
        
        tempo_total = time.time() - inicio
        print(f"BERT concluído em {tempo_total:.2f}s")
        
        return np.array(embeddings)
    
    def processar_textos_tfidf(self, textos: list) -> np.ndarray:
        """
        Processa textos usando TF-IDF.
        """
        print(f"Processando {len(textos)} textos com TF-IDF...")
        
        inicio = time.time()
        
        try:
            matriz_tfidf = self.tfidf_vectorizer.fit_transform(textos)
            embeddings = matriz_tfidf.toarray()
            
            tempo_total = time.time() - inicio
            print(f"TF-IDF concluído em {tempo_total:.2f}s")
            print(f"Dimensões: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            print(f"Erro no TF-IDF: {e}")
            return np.zeros((len(textos), 1000))
    
    def processar_textos_word2vec(self, tokens_normalizados: list) -> np.ndarray:
        """
        Treina Word2Vec e calcula embeddings médios dos documentos.
        """
        print(f"Treinando Word2Vec com {len(tokens_normalizados)} documentos...")
        
        inicio = time.time()
        
        try:
            self.word2vec_model = Word2Vec(
                sentences=tokens_normalizados,
                vector_size=300,
                window=5,
                min_count=2,
                workers=4,
                epochs=10
            )
            
            print(f"Vocabulário Word2Vec: {len(self.word2vec_model.wv.key_to_index)} palavras")
            
            embeddings = []
            for tokens in tokens_normalizados:
                if not tokens:
                    embeddings.append(np.zeros(300))
                    continue
                
                vetores_palavras = []
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        vetores_palavras.append(self.word2vec_model.wv[token])
                
                if vetores_palavras:
                    embedding_medio = np.mean(vetores_palavras, axis=0)
                    embeddings.append(embedding_medio)
                else:
                    embeddings.append(np.zeros(300))
            
            tempo_total = time.time() - inicio
            print(f"Word2Vec concluído em {tempo_total:.2f}s")
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"Erro no Word2Vec: {e}")
            return np.zeros((len(tokens_normalizados), 300))
    
    def calcular_similaridades(self, embeddings: np.ndarray, nome_modelo: str) -> dict:
        """
        Calcula matriz de similaridade coseno e estatísticas.
        """
        print(f"Calculando similaridades para {nome_modelo}...")
        
        try:
            matriz_sim = cosine_similarity(embeddings)
            
            mask = ~np.eye(matriz_sim.shape[0], dtype=bool)
            similaridades = matriz_sim[mask]
            
            stats = {
                'modelo': nome_modelo,
                'dimensoes': embeddings.shape[1],
                'num_documentos': embeddings.shape[0],
                'similaridade_media': np.mean(similaridades),
                'similaridade_std': np.std(similaridades),
                'similaridade_min': np.min(similaridades),
                'similaridade_max': np.max(similaridades),
                'matriz_similaridade': matriz_sim
            }
            
            return stats
            
        except Exception as e:
            print(f"Erro no cálculo de similaridades: {e}")
            return {'modelo': nome_modelo, 'erro': str(e)}
    
    def comparar_modelos(self, dados_processados: list) -> dict:
        """
        Executa comparação completa entre os três modelos.
        """
        print("\n" + "="*50)
        print("INICIANDO COMPARAÇÃO DE MODELOS")
        print("="*50)
        
        if not dados_processados:
            print("Nenhum dado para processar")
            return {}
        
        textos_brutos = [item.get('texto_bruto', '') for item in dados_processados]
        tokens_normalizados = [item.get('tokens_normalizados', []) for item in dados_processados]
        
        print(f"Processando {len(textos_brutos)} documentos")
        
        indices_validos = [i for i, texto in enumerate(textos_brutos) if texto.strip()]
        textos_validos = [textos_brutos[i] for i in indices_validos]
        tokens_validos = [tokens_normalizados[i] for i in indices_validos]
        
        print(f"Documentos válidos: {len(textos_validos)}")
        
        resultados = {}
        
        try:
            print("\n--- TF-IDF ---")
            embeddings_tfidf = self.processar_textos_tfidf(textos_validos)
            resultados['tfidf'] = self.calcular_similaridades(embeddings_tfidf, "TF-IDF")
        except Exception as e:
            print(f"Erro no TF-IDF: {e}")
            resultados['tfidf'] = {'modelo': 'TF-IDF', 'erro': str(e)}
        
        try:
            print("\n--- Word2Vec ---")
            embeddings_w2v = self.processar_textos_word2vec(tokens_validos)
            resultados['word2vec'] = self.calcular_similaridades(embeddings_w2v, "Word2Vec")
        except Exception as e:
            print(f"Erro no Word2Vec: {e}")
            resultados['word2vec'] = {'modelo': 'Word2Vec', 'erro': str(e)}
        
        try:
            print("\n--- BERT ---")
            embeddings_bert = self.processar_textos_bert(textos_validos)
            resultados['bert'] = self.calcular_similaridades(embeddings_bert, "BERT")
        except Exception as e:
            print(f"Erro no BERT: {e}")
            resultados['bert'] = {'modelo': 'BERT', 'erro': str(e)}
        
        self.resultados_comparacao = resultados
        return resultados
    
    def gerar_relatorio_comparativo(self) -> str:
        """
        Gera relatório comparativo detalhado.
        """
        if not self.resultados_comparacao:
            return "Não há resultados para gerar relatório"
        
        relatorio = []
        relatorio.append("="*60)
        relatorio.append("RELATÓRIO COMPARATIVO - MODELOS DE REPRESENTAÇÃO TEXTUAL")
        relatorio.append("="*60)
        relatorio.append("")
        
        relatorio.append("COMPARAÇÃO GERAL:")
        relatorio.append("-" * 60)
        
        linha_header = f"{'Modelo':<12} {'Dimensões':<10} {'Sim.Média':<10} {'Std':<8} {'Min':<8} {'Max':<8}"
        relatorio.append(linha_header)
        relatorio.append("-" * 60)
        
        for modelo_key, dados in self.resultados_comparacao.items():
            if 'erro' not in dados:
                linha = f"{dados['modelo']:<12} {dados['dimensoes']:<10} {dados['similaridade_media']:<10.3f} {dados['similaridade_std']:<8.3f} {dados['similaridade_min']:<8.3f} {dados['similaridade_max']:<8.3f}"
                relatorio.append(linha)
            else:
                linha = f"{dados['modelo']:<12} {'ERRO':<10} {dados['erro']}"
                relatorio.append(linha)
        
        relatorio.append("")
        
        relatorio.append("ANÁLISE DETALHADA:")
        relatorio.append("-" * 40)
        
        for modelo_key, dados in self.resultados_comparacao.items():
            if 'erro' not in dados:
                relatorio.append(f"\n{dados['modelo']}:")
                relatorio.append(f"  • Dimensionalidade: {dados['dimensoes']}")
                relatorio.append(f"  • Documentos processados: {dados['num_documentos']}")
                relatorio.append(f"  • Similaridade média: {dados['similaridade_media']:.4f}")
                relatorio.append(f"  • Desvio padrão: {dados['similaridade_std']:.4f}")
                
                if dados['similaridade_media'] > 0.5:
                    interpretacao = "Alta similaridade entre documentos"
                elif dados['similaridade_media'] > 0.3:
                    interpretacao = "Similaridade moderada"
                else:
                    interpretacao = "Baixa similaridade (mais diversidade)"
                
                relatorio.append(f"  • Interpretação: {interpretacao}")
        
        relatorio.append("\n" + "="*60)
        relatorio.append("CARACTERÍSTICAS DOS MODELOS:")
        relatorio.append("="*60)
        
        relatorio.append("\nTF-IDF:")
        relatorio.append("  v Rápido e eficiente")
        relatorio.append("  v Boa para recuperação de informação")
        relatorio.append("  x Não captura semântica profunda")
        relatorio.append("  x Complexo, complexidade computacional alta")
        
        relatorio.append("\nWord2Vec:")
        relatorio.append("  v Captura relações semânticas")
        relatorio.append("  v Representação densa")
        relatorio.append("  x Não considera contexto global")
        relatorio.append("  x Requer treinamento específico")
        
        relatorio.append("\nBERT:")
        relatorio.append("  v Contexto bidirecional completo")
        relatorio.append("  v Modelo mais moderno e avançado")
        relatorio.append("  v Modelo mais robusto e preciso")
        relatorio.append("  v Modelo com mais base de treinamento")
        relatorio.append("  x Computacionalmente caro")
        relatorio.append("  x Maior tempo de processamento")
        
        return "\n".join(relatorio)
    
    def salvar_resultados(self, caminho_base: str):
        """
        Salva resultados da comparação em arquivos.
        """
        try:
            relatorio = self.gerar_relatorio_comparativo()
            with open(f"{caminho_base}_relatorio_comparativo.txt", 'w', encoding='utf-8') as f:
                f.write(relatorio)
            
            dados_csv = []
            for modelo_key, dados in self.resultados_comparacao.items():
                if 'erro' not in dados:
                    dados_csv.append({
                        'Modelo': dados['modelo'],
                        'Dimensoes': dados['dimensoes'],
                        'Num_Documentos': dados['num_documentos'],
                        'Similaridade_Media': dados['similaridade_media'],
                        'Similaridade_Std': dados['similaridade_std'],
                        'Similaridade_Min': dados['similaridade_min'],
                        'Similaridade_Max': dados['similaridade_max']
                    })
            
            if dados_csv:
                df = pd.DataFrame(dados_csv)
                df.to_csv(f"{caminho_base}_comparacao_modelos.csv", index=False, encoding='utf-8-sig')
            
            print(f"Resultados salvos em: {caminho_base}_*")
            
        except Exception as e:
            print(f"Erro ao salvar resultados: {e}")
