#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste para comparação de modelos usando dados já coletados.
Útil para testar apenas a funcionalidade de comparação sem fazer scraping.
"""

import os
import sys
import pandas as pd
from comparador_modelos import ComparadorModelos

def carregar_dados_existentes():
    """
    Carrega dados já processados do CSV ou arquivos de texto.
    """
    database_dir = "./database/trabalho2"
    
    # Tentar carregar CSV primeiro
    csv_path = os.path.join(database_dir, "dados_processados.csv")
    if os.path.exists(csv_path):
        print(f"Carregando dados de: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Simular estrutura de dados processados
        dados_processados = []
        for _, row in df.iterrows():
            item = {
                'id': row.get('ID', ''),
                'texto_bruto': row.get('Texto_Bruto', ''),
                'tokens_normalizados': row.get('Tokens_Normalizados', '').split() if row.get('Tokens_Normalizados') else []
            }
            dados_processados.append(item)
        
        return dados_processados
    
    # Tentar carregar arquivos de texto individuais
    txt_files = []
    if os.path.exists(database_dir):
        txt_files = [f for f in os.listdir(database_dir) if f.endswith('.txt')]
    
    if txt_files:
        print(f"Carregando {len(txt_files)} arquivos de texto...")
        dados_processados = []
        
        for filename in txt_files[:20]:  # Limitar a 20 para teste
            filepath = os.path.join(database_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    conteudo = f.read()
                    
                    # Processamento básico para simular dados
                    tokens = conteudo.lower().split()
                    # Filtrar tokens básicos
                    tokens_filtrados = [t for t in tokens if len(t) > 3 and t.isalpha()]
                    
                    item = {
                        'id': filename,
                        'texto_bruto': conteudo,
                        'tokens_normalizados': tokens_filtrados
                    }
                    dados_processados.append(item)
                    
            except Exception as e:
                print(f"Erro ao carregar {filename}: {e}")
        
        return dados_processados
    
    return None

def teste_comparacao():
    """
    Executa teste da comparação de modelos.
    """
    print("="*60)
    print("TESTE DE COMPARAÇÃO DE MODELOS")
    print("="*60)
    
    # Carregar dados
    dados = carregar_dados_existentes()
    
    if not dados:
        print("❌ Nenhum dado encontrado!")
        print("\nPara usar este teste:")
        print("1. Execute primeiro o scraper principal (main.py)")
        print("2. Ou coloque arquivos .txt na pasta ./database/trabalho2/")
        return False
    
    print(f"✓ {len(dados)} documentos carregados")
    
    # Executar comparação
    try:
        comparador = ComparadorModelos()
        resultados = comparador.comparar_modelos(dados)
        
        # Exibir relatório
        relatorio = comparador.gerar_relatorio_comparativo()
        print("\n" + relatorio)
        
        # Salvar resultados
        caminho_base = "./database/trabalho2/teste_comparacao"
        comparador.salvar_resultados(caminho_base)
        
        print(f"\n✓ Teste concluído com sucesso!")
        print(f"✓ Relatórios salvos em: ./database/trabalho2/")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def main():
    """Função principal do teste."""
    if not teste_comparacao():
        print("\n" + "="*60)
        print("INSTALAÇÃO DE DEPENDÊNCIAS")
        print("="*60)
        print("Se houver erros, instale as dependências:")
        print("pip install -r requirements.txt")
        print("\nOu individualmente:")
        print("pip install transformers torch gensim scikit-learn")

if __name__ == "__main__":
    main()
