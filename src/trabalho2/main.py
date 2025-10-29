# Setup Selenium e bibliotecas
import time
import hashlib
import os
from functools import wraps
from urllib.parse import urlparse

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    WebDriverException, TimeoutException, StaleElementReferenceException
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, WordNetLemmatizer
import string
import re

# Import do comparador de modelos
from comparador_modelos import ComparadorModelos

# Configuração
MAX_POSTS = 2
DATABASE_DIR = "./database/trabalho2"

# Configuração do NLTK
def setup_nltk_resources():
    """Configura e baixa os recursos necessários do NLTK."""
    required_resources = [
        'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'rslp'
    ]
    
    for resource in required_resources:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource in ['stopwords', 'wordnet']:
                nltk.data.find(f'corpora/{resource}')
            elif resource == 'rslp':
                nltk.data.find('stemmers/rslp')
            else:
                nltk.data.find(f'taggers/{resource}')
        except LookupError:
            print(f"Baixando recurso NLTK: {resource}")
            nltk.download(resource, quiet=True)

# Inicialização NLTK
setup_nltk_resources()

# Configuração e utilitários
def build_driver():
    """Cria uma instância do Chrome usando o Selenium Manager automático."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")
    options.add_argument("--remote-allow-origins=*") # Boa prática para compatibilidade

    # Apenas instanciar webdriver.Chrome com as opções.
    # O Selenium Manager fará o download do browser e do driver corretos automaticamente.
    driver = webdriver.Chrome(options=options)

    driver.set_page_load_timeout(60)
    return driver

# Exceções que indicam perda de sessão/conexão com o Chromedriver
RECONECT_EXC = (WebDriverException,)

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _extract_filename_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    path = parsed_url.path.strip('/')
    parts = path.split('/')
    
    if len(parts) >= 3:
        year, month, title = parts[0], parts[1], parts[2]
        if title.endswith('.html'):
            title = title[:-5]
        return f"{year}-{month}-{title}.txt"
    else:
        filename = path.replace('/', '-')
        if filename.endswith('.html'):
            filename = filename[:-5]
        return f"{filename}.txt"

# ==============================
# Decorator de retry com reinit
# ==============================
def with_reconnect(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        last_exc = None
        for attempt in range(3):
            try:
                return method(self, *args, **kwargs)
            except RECONECT_EXC as e:
                last_exc = e
                print(f"[WARN] Sessão perdida ({e.__class__.__name__}). Reiniciando driver... (tentativa {attempt+1}/3)")
                try: self.driver.quit()
                except Exception: pass
                self.driver = build_driver()
                self.wait = WebDriverWait(self.driver, 20)
                # Retorna para a URL principal
                if hasattr(self, 'base_url'):
                    try:
                        self.driver.get(self.base_url)
                        time.sleep(2)
                    except Exception as e2:
                        print(f"[WARN] Falha ao retornar para a URL de recuperação: {e2}")
                time.sleep(0.5 + attempt * 0.5)
            except TimeoutException as e:
                last_exc = e
                print(f"[WARN] Timeout. Repetindo operação... (tentativa {attempt+1}/3)")
                time.sleep(1.2 + attempt * 0.5)
            except StaleElementReferenceException as e:
                last_exc = e
                print(f"[WARN] Elemento stale. Recarregando página atual... (tentativa {attempt+1}/3)")
                try:
                    self.driver.refresh()
                    time.sleep(2)
                except Exception: pass
        raise last_exc
    return wrapper

# Classe base
class ProcessadorPLN:
    """
    Nessa classe, deixei implementado o processamento de limpeza e pre-processamento de texto
    conforme apresentado na aula 3 e 4 do professor, ai o meu scraper é baseado nessa classe e posso usar no trabalho 3 da unidade 5 e 6.
    Tentei usar aquela tabelinha de exemplo da apresentação da aula 4.
    Implementado tokenização, normalização, stemming, lemmatização e extração de metadados.
    """
    
    def __init__(self):
        """Init processador PLN."""
        self.stemmer = RSLPStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('portuguese'))
        print("ProcessadorPLN inicializado")
    
    def tokenizar(self, texto: str) -> dict:
        """
        Realiza tokenização, usando o nltk para tokenizar o texto em sentenças e palavras.
        Não sei se ja poderia usar o nltk, mas achei num vídeo que usava o nltk para tokenizar
        o texto e por coincidencia o professor mostrou na aula 5.
        """
        try:
            sentencas = sent_tokenize(texto, language='portuguese')
            tokens = word_tokenize(texto.lower(), language='portuguese')
            
            return {
                'sentencas': sentencas,
                'tokens': tokens,
                'num_sentencas': len(sentencas),
                'num_tokens': len(tokens)
            }
        except Exception as e:
            print(f"Erro na tokenização: {e}")
            return {'sentencas': [], 'tokens': [], 'num_sentencas': 0, 'num_tokens': 0}
    
    def normalizar_texto(self, tokens: list) -> list:
        """
        Normaliza tokens: remove pontuação, stopwords, converte para minúsculas.
        Tive alguns problemas com a normalização, pois tem postagem que tem muitas 
        quebras de linha, texto customizado, e o texto fica muito difícil de ler.
        Acho que comentei algo contigo professor.
        Então fui incrementando as regras de normalização.
        """
        try:
            # Remove pontuação e números isolados
            tokens_limpos = []
            for token in tokens:
                token_lower = token.lower()
                if (token_lower not in string.punctuation and 
                    not token_lower.isdigit() and 
                    len(token_lower) > 2):
                    tokens_limpos.append(token_lower)
            
            # Remove stopwords (português e inglês)
            tokens_sem_stopwords = [
                token for token in tokens_limpos 
                if token not in self.stop_words
            ]
            
            # Remove tokens muito curtos ou que são apenas símbolos
            tokens_filtrados = [
                token for token in tokens_sem_stopwords 
                if len(token) >= 3 and any(c.isalpha() for c in token)
            ]
            
            return tokens_filtrados
            
        except Exception as e:
            print(f"Erro na normalização: {e}")
            return []
    
    def aplicar_stemming(self, tokens: list) -> list:
        """Aplica stemming usando RSLP Stemmer."""
        try:
            return [self.stemmer.stem(token) for token in tokens]
        except Exception as e:
            print(f"Erro no stemming: {e}")
            return tokens
    
    def aplicar_lemmatizacao(self, tokens: list) -> list:
        """Aplica lemmatização usando WordNet Lemmatizer (TODO: Verificar com o professor se seria essa lemmatização)."""
        try:
            lemmas = []
            for token in tokens:
                lemma = self.lemmatizer.lemmatize(token, pos='n')  # Substantivo
                if lemma == token:
                    lemma = self.lemmatizer.lemmatize(token, pos='v')  # Verbo
                if lemma == token:
                    lemma = self.lemmatizer.lemmatize(token, pos='a')  # Adjetivo
                lemmas.append(lemma)
            return lemmas
        except Exception as e:
            print(f"Erro na lemmatização: {e}")
            return tokens
    
    def extrair_metadados(self, texto: str, url: str) -> dict:
        """Extrai metadados relevantes: datas, números, nomes próprios."""
        try:
            metadados = {
                'url': url,
                'tamanho_caracteres': len(texto),
                'datas_encontradas': [],
                'numeros_relevantes': [],
                'nomes_proprios': []
            }
            
            # Extração de data
            padrao_data = r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'
            datas = []
            datas.extend(re.findall(padrao_data, texto, re.IGNORECASE))
            metadados['datas_encontradas'] = list(set(datas))[:5]
            
            # Extração de números relevantes (anos, quantidades)
            numeros = re.findall(r'\b(?:19|20)\d{2}\b|\b\d{1,3}(?:[.,]\d{3})*\b', texto)
            metadados['numeros_relevantes'] = list(set(numeros))[:10]
            
            # Extração de nomes próprios (podendo ser nome de pessoa, local, rua, etc)
            entidades = re.findall(r'\b[A-ZÁÉÍÓÚÂÊÔÀÇ][a-záéíóúâêôàçãõ]{2,}\b', texto)
            palavras_comuns = {'O', 'A', 'Os', 'As', 'E', 'De', 'Da', 'Do', 'Em', 'Na', 'No'}
            entidades_filtradas = [e for e in entidades if e not in palavras_comuns]
            metadados['nomes_proprios'] = list(set(entidades_filtradas))[:10]
            
            return metadados
            
        except Exception as e:
            print(f"Erro na extração de metadados: {e}")
            return {'url': url, 'erro': str(e)}
    
    def processar_texto_completo(self, texto: str, url: str, titulo: str = "") -> dict:
        """Executa por completo: tokenização, normalização, stemming, lemmatização e extração de metadados."""
        try:
            # 1. Tokenização
            resultado_tokenizacao = self.tokenizar(texto)
            tokens_originais = resultado_tokenizacao['tokens']
            
            # 2. Normalização
            tokens_normalizados = self.normalizar_texto(tokens_originais)
            
            # 3. Stemming
            tokens_stemming = self.aplicar_stemming(tokens_normalizados)
            
            # 4. Lemmatização
            tokens_lemmatizacao = self.aplicar_lemmatizacao(tokens_normalizados)
            
            # 5. Extração de metadados
            metadados = self.extrair_metadados(texto, url)
            
            # 6. Montagem do resultado estruturado conforme tabela do professor
            resultado = {
                'id': url,
                'titulo': titulo,
                'texto_bruto': texto,
                'tokens_originais': tokens_originais,
                'tokens_normalizados': tokens_normalizados,
                'tokens_stemming': tokens_stemming,
                'tokens_lemmatizacao': tokens_lemmatizacao,
                'estatisticas': {
                    'num_sentencas': resultado_tokenizacao['num_sentencas'],
                    'num_tokens_originais': resultado_tokenizacao['num_tokens'],
                    'num_tokens_normalizados': len(tokens_normalizados),
                    'num_caracteres': len(texto)
                },
                'metadados': metadados,
                'hash_conteudo': _hash_text(texto)
            }
            
            return resultado
            
        except Exception as e:
            print(f"Erro no processamento completo: {e}")
            return {
                'id': url,
                'titulo': titulo,
                'texto_bruto': texto,
                'erro_processamento': str(e),
                'metadados': {'url_origem': url, 'erro': str(e)},
                'hash_conteudo': _hash_text(texto)
            }

"""
A classe **`AdalbertodayScraper`** tem tudo implementado conforme critérios de avaliação.
Tokenização, normalização, stemming, lemmatização e extração de metadados são implementados na classe ProcessadorPLN.
"""

# Scraper principal (baseado na classe FURBScraper do professor)
class AdalbertoScraper:
    
    def __init__(self):
        """Inicializa o scraper com configurações de controle."""
        self.driver = build_driver()
        self.wait = WebDriverWait(self.driver, 20)
        self.base_url = "https://adalbertoday.blogspot.com/"
        self.processador_pln = ProcessadorPLN()
        self.urls_processadas = set()
        self.hashes_conteudo = set()
        self.resultados = []
        
        print(f"AdalbertoScraper inicializado - máximo {MAX_POSTS} posts")
    
    def _get(self, url):
        self.driver.get(url)
    
    def _extrair_texto_limpo(self, elemento) -> str:
        try:
            texto = elemento.text
            
            if not texto:
                texto = self.driver.execute_script("return arguments[0].textContent;", elemento)
            
            texto = re.sub(r'\s+', ' ', texto)
            texto = re.sub(r'\n\s*\n', '\n\n', texto)
            
            return texto.strip()
            
        except Exception as e:
            print(f"Erro na extração de texto: {e}")
            return ""
    
    @with_reconnect
    def abrir_blog(self):
        """Abre a página principal do blog Adalbertoday."""
        print(f"Abrindo blog: {self.base_url}")
        self._get(self.base_url)
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        print("Acessado blog")
    
    @with_reconnect
    def encontrar_links_posts(self) -> list:
        print("Procurando links dos posts")
        
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "a")))
            elementos_link = self.driver.find_elements(By.TAG_NAME, "a")
            
            links_encontrados = []
            for elemento in elementos_link:
                try:
                    href = elemento.get_attribute("href")
                    if href and 'adalbertoday.blogspot.com' in href:
                        if re.search(r'/\d{4}/\d{2}/', href):
                            if href not in links_encontrados:
                                links_encontrados.append(href)
                                print(f"Post encontrado: {href}")
                except Exception as e:
                    continue
            
            print(f"{len(links_encontrados)} posts encontrados")
            return links_encontrados
            
        except Exception as e:
            print(f"Erro ao buscar links: {e}")
            return []
    
    @with_reconnect
    def processar_post(self, url: str) -> dict:
        print(f"Processando post: {url}")
        
        try:
            if url in self.urls_processadas:
                return None
            
            self._get(url)
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            try:
                titulo_elem = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h3.post-title, .entry-title, .post-title"))
                )
                titulo = titulo_elem.text.strip() if titulo_elem else "Sem título"
            except:
                titulo = "Sem título"
            
            try:
                conteudo_elem = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".post-body, .entry-content, .post-content"))
                )
                texto_limpo = self._extrair_texto_limpo(conteudo_elem)
            except:
                print("Texto do post não encontrado")
                return None
            
            hash_conteudo = _hash_text(texto_limpo)
            if hash_conteudo in self.hashes_conteudo: # Verifica duplicação por hash, pego link duplicado
                return None

            resultado = self.processador_pln.processar_texto_completo(
                texto_limpo, url, titulo
            )
            
            self.urls_processadas.add(url)
            self.hashes_conteudo.add(hash_conteudo)
            
            print(f"Post processado: {titulo[:50]}...")
            print(f"Caracteres: {len(texto_limpo)}")
            print(f"Tokens normalizados: {len(resultado.get('tokens_normalizados', []))}")
            
            return resultado
            
        except Exception as e:
            print(f"Erro ao processar post {url}: {e}")
            return None
    
    @with_reconnect
    def processar_lista_posts(self):
        """Processa lista completa de posts encontrados. Na homepage do blog contém todos os posts do blog."""
        try:
            self.abrir_blog()
            links_posts = self.encontrar_links_posts()
            
            if not links_posts:
                print("Nenhum post encontrado")
                return
            
            if len(links_posts) > MAX_POSTS:
                print(f"Limitando processamento a {MAX_POSTS} posts")
                links_posts = links_posts[:MAX_POSTS]
            
            print(f"\nProcessando {len(links_posts)} posts")
            print("--------------------------------")
            
            for i, url in enumerate(links_posts, 1):
                print(f"\n[{i}/{len(links_posts)}] Processando")
                
                resultado = self.processar_post(url)
                if resultado:
                    self.resultados.append(resultado)
                    self._salvar_arquivo_individual(resultado, url) # Ideia do professor para não perder a base de dados, qualquer coisa tenho os arquivos do post
                
                if i < len(links_posts):
                    time.sleep(2)
            
            print(f"\nProcessamento concluído: {len(self.resultados)} posts")
            
        except Exception as e:
            print(f"Erro no scraping: {e}")
    
    def _salvar_arquivo_individual(self, resultado: dict, url: str):
        """Salva texto bruto em arquivo individual para não perder a base de dados, qualquer coisa tenho os arquivos do post."""
        try:
            diretorio = DATABASE_DIR
            os.makedirs(diretorio, exist_ok=True)
            
            nome_arquivo = _extract_filename_from_url(url)
            caminho = os.path.join(diretorio, nome_arquivo)
            
            with open(caminho, 'w', encoding='utf-8') as f:
                f.write(resultado.get('texto_bruto', ''))
                
        except Exception as e:
            print(f"Erro ao salvar arquivo: {e}")
    
    def close(self):
        """Fecha o navegador."""
        try:
            self.driver.quit()
            print("Navegador fechado")
        except Exception:
            pass

def exportar_resultados(resultados: list) -> str:
    """
    Exporta resultados da tabela da aula 4 para CSV.
    """
    if not resultados:
        print("\nNenhum resultado foi extraído")
        return None
    
    try:
        rows = []
        for item in resultados:
            row = {
                "ID": item.get('id', ''),
                "Texto_Bruto": item.get('texto_bruto', '')[:200] + '...' if len(item.get('texto_bruto', '')) > 200 else item.get('texto_bruto', ''),
                "Tokens_Normalizados": ' '.join(item.get('tokens_normalizados', [])[:20]),
                "Stemming": ' '.join(item.get('tokens_stemming', [])[:20]),
                "Lemma": ' '.join(item.get('tokens_lemmatizacao', [])[:20]),
                "Outros_Dados_Relevantes": _formatar_metadados(item.get('metadados', {})),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        diretorio = DATABASE_DIR
        os.makedirs(diretorio, exist_ok=True)
        caminho_csv = os.path.join(diretorio, "dados_processados.csv")
        df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
        
        print(f"\nArquivo CSV salvo .")
        return caminho_csv
        
    except Exception as e:
        print(f"Erro na exportação: {e}")
        return None

def executar_comparacao_modelos(resultados: list):
    """
    Executa comparação entre BERT, TF-IDF e Word2Vec usando os dados processados.
    """
    if not resultados:
        print("\nNenhum resultado disponível para comparação")
        return
    
    print("\n" + "="*60)
    print("INICIANDO ANÁLISE COMPARATIVA DE MODELOS")
    print("="*60)
    
    try:
        # Inicializar comparador
        comparador = ComparadorModelos()
        
        # Executar comparação
        resultados_comparacao = comparador.comparar_modelos(resultados)
        
        # Gerar e exibir relatório
        relatorio = comparador.gerar_relatorio_comparativo()
        print("\n" + relatorio)
        
        # Salvar resultados
        caminho_base = os.path.join(DATABASE_DIR, "analise_modelos")
        comparador.salvar_resultados(caminho_base)
        
        print(f"\n✓ Análise comparativa concluída!")
        print(f"✓ Relatórios salvos em: {DATABASE_DIR}/")
        
    except Exception as e:
        print(f"\nErro na comparação de modelos: {e}")
        print("Certifique-se de que as dependências estão instaladas:")
        print("pip install transformers torch gensim scikit-learn matplotlib seaborn")

def _formatar_metadados(metadados: dict) -> str:
    """Formata metadados."""
    try:
        elementos = []
        if metadados.get('datas_encontradas'):
            elementos.append(f"Datas: {', '.join(metadados['datas_encontradas'][:3])}")
        if metadados.get('nomes_proprios'):
            elementos.append(f"Nomes: {', '.join(metadados['nomes_proprios'][:3])}")
        return " | ".join(elementos) if elementos else "N/A"
    except Exception:
        return "N/A"

"""
Aqui gera um relatório final com as estatísticas do processamento conforme a tabela da aula 4.
"""
# Colocado o número máximo de posts a processar para testar


def main():
    # Inicialização
    scraper = AdalbertoScraper()
    
    try:
        scraper.processar_lista_posts()
        exportar_resultados(scraper.resultados)
        
        # Executar comparação de modelos
        if scraper.resultados:
            print("\n" + "="*60)
            print("INICIANDO ANÁLISE COMPARATIVA DE MODELOS")
            print("="*60)
            
            resposta = input("\nDeseja executar a comparação BERT vs TF-IDF vs Word2Vec? (s/n): ").lower().strip()
            if resposta in ['s', 'sim', 'y', 'yes']:
                executar_comparacao_modelos(scraper.resultados)
            else:
                print("Análise comparativa pulada.")
        
    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário")
    except Exception as e:
        print(f"\nErro crítico: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()