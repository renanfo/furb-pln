import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin, urlparse
import time

# URL base do Adalbertoday
URL_BASE = "https://adalbertoday.blogspot.com/"
MAX_POSTS = 10

def extrair_texto_limpo(elemento):
    print("Extraindo texto limpo")

    tags_paragrafo = ['p', 'div']
    for tag in tags_paragrafo:
        for elem in elemento.find_all(tag):
            if elem.get_text().strip():
                elem.append("\n\n")
    
    texto = elemento.get_text(separator=' ')
    
    texto = re.sub(r' +', ' ', texto)
    
    texto = re.sub(r'\n\s*\n\s*\n+', '\n\n', texto)
    
    linhas = [linha.strip() for linha in texto.split('\n')]
    
    texto_limpo = []
    linha_anterior_vazia = False
    
    # Tratar quebra de linha, pois no site são usadas muitas quebras de linha
    # e o texto fica muito difícil de ler
    for linha in linhas:
        if linha:
            texto_limpo.append(linha)
            linha_anterior_vazia = False
        elif not linha_anterior_vazia:
            texto_limpo.append('')
            linha_anterior_vazia = True
    
    resultado = '\n'.join(texto_limpo).strip()
    
    resultado = re.sub(r'\s+([.,:;!?])', r'\1', resultado)
    
    resultado = re.sub(r'([.!?])([A-Z])', r'\1 \2', resultado)
    
    return resultado

def extrair_nome_arquivo_da_url(url):
    print(f"Extraindo nome do arquivo da URL: {url}")
    # https://adalbertoday.blogspot.com/2020/08/o-menino-caiu-na-privada.html
    # 2020-08-o-menino-caiu-na-privada.txt
    
    parsed_url = urlparse(url)
    caminho = parsed_url.path
    
    caminho = caminho.strip('/')
    
    partes = caminho.split('/')
    
    if len(partes) >= 3:
        ano = partes[0]
        mes = partes[1]
        nome_arquivo = partes[2]
        
        if nome_arquivo.endswith('.html'):
            nome_arquivo = nome_arquivo[:-5]
        
        nome_final = f"{ano}-{mes}-{nome_arquivo}.txt"
        return nome_final
    else:
        nome_arquivo = caminho.replace('/', '-')
        if nome_arquivo.endswith('.html'):
            nome_arquivo = nome_arquivo[:-5]
        return f"{nome_arquivo}.txt"

def encontrar_links_posts(soup, url_base):
    print("Encontrando links dos posts")
    links_posts = []
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        url_completa = urljoin(url_base, href)
        
        if 'adalbertoday.blogspot.com' in url_completa:
            if re.search(r'/\d{4}/\d{2}/', url_completa):
                if url_completa not in links_posts:
                    print(f"Encontrado link: {url_completa}")
                    links_posts.append(url_completa)
    
    return links_posts

def processar_post(url_post):
    print(f"Processando: {url_post}")
    try:
        response = requests.get(url_post)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        post_title = soup.find('h3', class_='post-title entry-title')
        titulo = None
        if post_title:
            titulo = post_title.get_text().strip()
        
        post_content_div = soup.find('div', class_='post-body entry-content')
        texto_completo = ""
        if post_content_div:
            texto_completo = extrair_texto_limpo(post_content_div)
        
        return titulo, texto_completo
        
    except Exception as e:
        print(f"Erro ao processar {url_post}: {e}")
        return None, None

def main():
    print("Acessando página principal do blog")
    try:
        response = requests.get(URL_BASE)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("Procurando links dos posts")
        links_posts = encontrar_links_posts(soup, URL_BASE)
        quantidade_posts = len(links_posts)
        
        print(f"Encontrados {quantidade_posts} posts para processar")
        if quantidade_posts > MAX_POSTS:
            print(f"Verificado que a quantidade de posts é maior que o limite, processando apenas os primeiros {MAX_POSTS} posts")
            links_posts = links_posts[:MAX_POSTS]
        
        diretorio = "arquivos/trabalho2"
        os.makedirs(diretorio, exist_ok=True)
        
        total_processados = 0
        total_salvos = 0
        
        for i, url_post in enumerate(links_posts, 1):
            print(f"\n[{i}/{len(links_posts)}] Processando post")
            
            titulo, conteudo = processar_post(url_post)
            
            if titulo and conteudo:
                print("Gera nome do arquivo")
                nome_arquivo = extrair_nome_arquivo_da_url(url_post)
                caminho_arquivo = os.path.join(diretorio, nome_arquivo)
                
                try:
                    print(f"Salvando o arquivo {nome_arquivo}")
                    with open(caminho_arquivo, 'w', encoding='utf-8') as arquivo:
                        arquivo.write(conteudo)
                    print(f"Salvo: {nome_arquivo}")
                    total_salvos += 1
                    
                except Exception as e:
                    print(f"Erro ao salvar {nome_arquivo}: {e}")
            else:
                print("Erro: Não foi possível extrair conteúdo")
            
            total_processados += 1
            
            time.sleep(1)
        
        print(f"\n{'='*60}")
        print(f"Posts encontrados: {quantidade_posts}")
        print(f"Máximo para processar: {MAX_POSTS}")
        print(f"Posts processados: {total_processados}")
        print(f"Arquivos salvos: {total_salvos}")
        print(f"Diretório de saída: {diretorio}")
        
    except Exception as e:
        print(f"Erro ao acessar a página principal: {e}")

if __name__ == "__main__":
    main()