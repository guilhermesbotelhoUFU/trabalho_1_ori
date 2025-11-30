#Gabriel Silva Tassara - 12311BSI218
#Guilherme Siqueira Botelho - 12311BSI217
#João Lucas Goncalves Teixeira - 12311BSI201
#Marcos Paulo Oliveira Gomes - 12311BSI231
#Kauan Felipe Desterro Carvalho 12311BSI226

import json
import math
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('stemmers/rslp')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('rslp')

class SistemaRecuperacaoInformacao:
    def __init__(self):
        self.documentos = {} 
        self.vocabulario = set()
        self.indice_invertido = {}
        self.matriz_tf_idf = {}
        self.radicalizador = RSLPStemmer()
        self.stop_words = set(stopwords.words('portuguese'))

    def _processar_texto(self, texto):
        texto = texto.lower()
        tokens = re.findall(r'\b[a-zà-ú]+\b', texto)
        tokens_processados = []
        
        for token in tokens:
            if token not in self.stop_words:
                radical = self.radicalizador.stem(token)
                tokens_processados.append(radical)
        
        return tokens_processados

    def _atualizar_estruturas(self):
        self.vocabulario = set()
        self.indice_invertido = {}
        self.matriz_tf_idf = {}

        for id_doc, texto in self.documentos.items():
            tokens = self._processar_texto(texto)
            
            for posicao, token in enumerate(tokens):
                self.vocabulario.add(token)
                
                if token not in self.indice_invertido:
                    self.indice_invertido[token] = {}
                
                if id_doc not in self.indice_invertido[token]:
                    self.indice_invertido[token][id_doc] = []
                
                self.indice_invertido[token][id_doc].append(posicao)

        total_docs = len(self.documentos)
        if total_docs == 0:
            return

        for id_doc, texto in self.documentos.items():
            tokens = self._processar_texto(texto)
            contagem_termos = Counter(tokens)
            total_termos_doc = len(tokens)
            
            self.matriz_tf_idf[id_doc] = {}

            for termo in self.vocabulario:
                tf = contagem_termos[termo] / total_termos_doc if total_termos_doc > 0 else 0
                
                contagem_doc_com_termo = len(self.indice_invertido.get(termo, {}))
                
                if contagem_doc_com_termo > 0:
                    idf = math.log10(total_docs / contagem_doc_com_termo)
                else:
                    idf = 0
                
                peso = tf * idf
                
                if contagem_doc_com_termo > 0 and termo in contagem_termos:
                     self.matriz_tf_idf[id_doc][termo] = peso

    def adicionar_documento(self, id_doc, texto):
        self.documentos[id_doc] = texto
        self._atualizar_estruturas()

    def adicionar_lote_documentos(self, lista_docs):
        for doc in lista_docs:
            id_doc = str(doc.get('name'))
            texto = doc.get('content', '')
            self.documentos[id_doc] = texto
        self._atualizar_estruturas()

    def remover_documento(self, id_doc):
        if id_doc in self.documentos:
            del self.documentos[id_doc]
            self._atualizar_estruturas()
            return True
        return False

    def obter_vocabulario(self):
        return sorted(list(self.vocabulario))

    def obter_tf_idf(self):
        return self.matriz_tf_idf

    def obter_indice_invertido(self):
        return self.indice_invertido

    def busca_booleana(self, consulta):
        tokens_brutos = consulta.split()
        docs_resultado = None
        operador_atual = 'OR'
        
        i = 0
        while i < len(tokens_brutos):
            token = tokens_brutos[i]
            token_upper = token.upper()

            if token_upper in ['AND', 'OR', 'NOT']:
                operador_atual = token_upper
                i += 1
                continue
            
            termo_radical = self.radicalizador.stem(token.lower())
            
            docs_com_termo = set()
            for id_doc, pesos_termos in self.matriz_tf_idf.items():
                if termo_radical in pesos_termos:
                    docs_com_termo.add(id_doc)
            
            if docs_resultado is None:
                if operador_atual == 'NOT':
                    todos_docs = set(self.documentos.keys())
                    docs_resultado = todos_docs - docs_com_termo
                else:
                    docs_resultado = docs_com_termo
            else:
                if operador_atual == 'AND':
                    docs_resultado = docs_resultado.intersection(docs_com_termo)
                elif operador_atual == 'OR':
                    docs_resultado = docs_resultado.union(docs_com_termo)
                elif operador_atual == 'NOT':
                    docs_resultado = docs_resultado.difference(docs_com_termo)
            
            i += 1

        if docs_resultado is None:
            return []
            
        return list(docs_resultado)

    def busca_similaridade_cosseno(self, consulta):
        tokens_consulta = self._processar_texto(consulta)
        vetor_consulta = Counter(tokens_consulta)
        
        pontuacoes = {}
        
        norma_consulta = math.sqrt(sum(c**2 for c in vetor_consulta.values()))
        if norma_consulta == 0:
            return []

        for termo, peso_consulta in vetor_consulta.items():
            if termo in self.indice_invertido:
                postagens = self.indice_invertido[termo]
                for id_doc in postagens:
                    peso_doc = self.matriz_tf_idf[id_doc].get(termo, 0)
                    pontuacoes[id_doc] = pontuacoes.get(id_doc, 0) + (peso_consulta * peso_doc)

        ranking_final = []
        for id_doc, produto_escalar in pontuacoes.items():
            pesos_doc = self.matriz_tf_idf[id_doc].values()
            norma_doc = math.sqrt(sum(w**2 for w in pesos_doc))
            
            if norma_doc > 0:
                similaridade = produto_escalar / (norma_consulta * norma_doc)
                ranking_final.append((id_doc, similaridade))

        ranking_final.sort(key=lambda x: x[1], reverse=True)
        return ranking_final

    def busca_frase(self, frase):
        tokens = self._processar_texto(frase)
        if not tokens:
            return []
        
        primeiro_termo = tokens[0]
        if primeiro_termo not in self.indice_invertido:
            return []
            
        docs_candidatos = set(self.indice_invertido[primeiro_termo].keys())
        
        for token in tokens[1:]:
            if token not in self.indice_invertido:
                return []
            docs_candidatos.intersection_update(self.indice_invertido[token].keys())
            
        resultados = []
        
        for id_doc in docs_candidatos:
            listas_posicoes = [self.indice_invertido[token][id_doc] for token in tokens]
            
            for pos in listas_posicoes[0]:
                if self._verificar_sequencia(listas_posicoes, pos):
                    resultados.append(id_doc)
                    break 
        
        return resultados

    def _verificar_sequencia(self, listas_posicoes, posicao_inicial_termo1):
        posicao_esperada = posicao_inicial_termo1 + 1
        
        for i in range(1, len(listas_posicoes)):
            if posicao_esperada not in listas_posicoes[i]:
                return False
            
            posicao_esperada += 1
            
        return True

def carregar_arquivo_json(nome_arquivo):
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            return dados
    except FileNotFoundError:
        print(f"Erro: Arquivo '{nome_arquivo}' não encontrado.")
        return []
    except json.JSONDecodeError:
        print("Erro: Arquivo JSON inválido.")
        return []

def imprimir_separador():
    print("-" * 50)

def main():
    sistema_ri = SistemaRecuperacaoInformacao()
    dados_json = carregar_arquivo_json('colecao - trabalho 01.json')
    indice_json = 0

    while True:
        imprimir_separador()
        print("SISTEMA DE RECUPERAÇÃO DA INFORMAÇÃO")
        print("1. Adicionar próximo documento do JSON")
        print("2. Adicionar TODOS os documentos do JSON")
        print("3. Remover documento por ID")
        print("4. Exibir Vocabulário")
        print("5. Exibir Matriz TF-IDF")
        print("6. Exibir Índice Invertido")
        print("7. Consulta Booleana (AND/OR/NOT)")
        print("8. Consulta por Similaridade (Cosseno)")
        print("9. Consulta por Frase")
        print("10. Sair")
        
        try:
            escolha = input("Escolha uma opção: ")
            imprimir_separador()
            
            if escolha == '1':
                if indice_json < len(dados_json):
                    doc = dados_json[indice_json]
                    sistema_ri.adicionar_documento(str(doc['name']), doc['content'])
                    print(f"Documento {doc['name']} adicionado.")
                    indice_json += 1
                else:
                    print("Todos os documentos do arquivo já foram adicionados.")

            elif escolha == '2':
                docs_restantes = dados_json[indice_json:]
                sistema_ri.adicionar_lote_documentos(docs_restantes)
                indice_json = len(dados_json)
                print(f"{len(docs_restantes)} documentos adicionados.")

            elif escolha == '3':
                id_doc = input("Digite o ID (name) do documento a remover: ")
                if sistema_ri.remover_documento(id_doc):
                    print("Documento removido com sucesso.")
                else:
                    print("ID não encontrado.")

            elif escolha == '4':
                vocab = sistema_ri.obter_vocabulario()
                print(f"Tamanho do vocabulário: {len(vocab)}")
                print(vocab)

            elif escolha == '5':
                matriz = sistema_ri.obter_tf_idf()
                for id_doc, termos in matriz.items():
                    print(f"Doc {id_doc}:")
                    for termo, valor in termos.items():
                        print(f"  {termo}: {valor:.4f}")

            elif escolha == '6':
                idx = sistema_ri.obter_indice_invertido()
                for termo, postagens in idx.items():
                    print(f"Termo: '{termo}'")
                    for id_doc, posicoes in postagens.items():
                        print(f"  -> Doc {id_doc}: posições {posicoes}")

            elif escolha == '7':
                consulta = input("Digite a consulta booleana (ex: termo1 AND termo2): ")
                resultados = sistema_ri.busca_booleana(consulta)
                print("Documentos encontrados:", resultados)

            elif escolha == '8':
                consulta = input("Digite a consulta para similaridade: ")
                resultados = sistema_ri.busca_similaridade_cosseno(consulta)
                print("Ranking de similaridade:")
                for id_doc, pontuacao in resultados:
                    print(f"Doc: {id_doc} | Pontuação: {pontuacao:.4f}")

            elif escolha == '9':
                frase = input("Digite a frase exata: ")
                resultados = sistema_ri.busca_frase(frase)
                print("Documentos contendo a frase:", resultados)

            elif escolha == '10':
                print("Encerrando...")
                break
            
            else:
                print("Opção inválida.")
        
        except Exception as e:
            print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()