import requests
from bs4 import BeautifulSoup
import re
import os

url = "https://adalbertoday.blogspot.com/2011/02/encontro-com-onca-pintada-do-spitzkopf.html"

response = requests.get(url)
content = response.content

soup = BeautifulSoup(content, 'html.parser')

post_title = soup.find('h3', class_='post-title entry-title')
titulo_principal = None
if post_title:
    titulo_principal = post_title.get_text().strip()

def extrair_texto_limpo(elemento):
    for script in elemento(["script", "style", "noscript"]):
        script.decompose()
    
    for img in elemento.find_all("img"):
        img.decompose()
    
    for br in elemento.find_all("br"):
        br.replace_with("\n")
    
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

post_content_div = soup.find('div', class_='post-body entry-content')
texto_completo = ""
if post_content_div:
    texto_completo = extrair_texto_limpo(post_content_div)

print("=" * 60)
print("TÍTULO PRINCIPAL:")
print("=" * 60)
if titulo_principal:
    print(titulo_principal)
else:
    print("Título não encontrado")

print("\n" + "=" * 60)
print("CONTEÚDO DO POST:")
print("=" * 60)
if texto_completo:
    print(texto_completo)
else:
    print("Conteúdo não encontrado")

diretorio = "database/trabalho1"
os.makedirs(diretorio, exist_ok=True)

nome_arquivo = os.path.join(diretorio, "texto_completo.txt")
try:
    with open(nome_arquivo, 'w', encoding='utf-8') as arquivo:
        arquivo.write("TÍTULO: " + (titulo_principal if titulo_principal else "Não encontrado") + "\n\n")
        arquivo.write("CONTEÚDO:\n")
        arquivo.write("=" * 80 + "\n")
        arquivo.write(texto_completo)
        arquivo.write("\n\n" + "=" * 80 + "\n")
        arquivo.write(f"Estatísticas: {len(texto_completo)} caracteres, {len(texto_completo.split())} palavras")
    
    print(f"\nConteúdo salvo no arquivo: {nome_arquivo}")
except Exception as e:
    print(f"\nErro ao salvar arquivo: {e}")
