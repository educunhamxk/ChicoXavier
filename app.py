import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
from pycaret.regression import *
from sklearn.pipeline import Pipeline


#configuração da página
st.set_page_config(page_title="Chico Xavier e Inteligência Artificial", page_icon="📚", layout="centered", initial_sidebar_state="collapsed")

#definição do tema
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #4F4F4F;
}
</style>
    """, unsafe_allow_html=True)

#título
st.title("O que a Inteligência Artificia diz sobre a obra de Chico Xavier?")

#exibir imagem tema do lula
st.image("chico.png")

#texto
st.markdown("Não sou espírita, mas sempre tive curiosidade a respeito da figura do Chico Xavier. Ele escreveu mais de 450 livros psicografados, vendeu mais de 50 milhões de exemplares, mas viveu de forma simples, cedendo todos os direitos de seus livros para instituições sociais e organizações espíritas, dedicando a maior parte do seu tempo para ajudar ao próximo, sem aceitar nada em troca. E mesmo assim, foi alvo de alegações de fraude e charlatanismo. Resolvi fazer um estudo para tentar entender se existem diferentes padrões nos textos psicografados de Chico que poderiam sugerir diferentes autorias, padrões que apenas modelos embasados em inteligência artificial poderiam captar.")

#1º Bloco************************************************************************************************************************
st.subheader("Conceitos utilizados na análise")

st.markdown("Utilizando conceitos de NLP, similaridade de cosseno e técnicas de clusterização, foi realizado um estudo para investigar se há diferenças significativas no estilo de escrita dos livros de Chico Xavier, de modo que seja possível separá-los pelos espíritos, que segundo o espiritismo, ditaram as obras.")

#2º Bloco************************************************************************************************************************
st.subheader("Preparação dos dados")

st.markdown("""
- Importação de mais de 100 livros escritos por Chico Xavier para o projeto;
- Remoção de caracteres especiais;
- Remoção de stopwords, palavras sem muito valor para análise, como preposições, por exemplo;
- Conversão para minúsculo para padronização;
- Remoção do nome do espíritos em todas as obras;
- Criação da matriz de vetores com base em TF-IDF, no qual cada termo ou conjunto de termos recebe uma nota de acordo com a importância que possui para o texto. Abaixo temos um exemplo, palavras que aparecem muito ao longo do texto recebem um menor peso do que palavras que aparecem poucas vezes, estas tendem a ser mais específicas e dizem mais sobre o conteúdo:
""")
            
st.image("tf-idf.png")

st.markdown("""
- Em seguida utilizamos a similaridade de cosseno, que é uma medida que avalia quão similar podem ser dois ou mais conteúdos. Utilizamos essa medida em cima do vetor criado na etapa anterior; 
- A partir disso, criamos os clusters utilizando também PCA para reduzir a dimensinalidade e facilitar a visualização dos grupos. Optamos por esse modelo de aprendizado não supervisionado, justamente porque a ideia é não passar nenhum viés para o modelo. Então o algoritmo identifica quais dados ou livros são mais similares e agrupa em clusters, ou grupos específicos;
""")
st.image("clusters.JPG")

#2º Bloco************************************************************************************************************************
st.subheader("1º Teste - Geração de Clusters de livros de Emmanuel e André Luiz")

st.markdown("""
Neste primeiro teste trabalhamos com 39 livros de Emmanuel e 15 livros de André Luiz. Os únicos filtros utilizados para determinar quais livros destes autores entrariam no teste é a quantidade de tokens, neste caso podemos dizer que é o mesmo que palavras. Estipulamos um limite mínimo de 50.000 tokens para que o livros pudesse ser considerado. Utilizamos este parâmetro porque livros que tendem a ser muito curtos poderiam dificultar o modelo a identificar padrões.
""")
            
st.image("teste1.JPG")

st.markdown("""
Como podemos ver no gráfico acima, apenas um livro de André Luiz foi classificado no cluster dos livros de Emmanuel, para todos os demais o modelo conseguiu identificar um padrão específico de forma que conseguiu separar de acordo com os espíritos, que segundo o espiritismo, ditaram as obras. Esse primeiro resultado já sugere existir padrões suficientes nos livros através de palavras ou grupos de palavras que permitem que o modelo encontre semelhanças de estilo nos livros de um mesmo espírito
""")
            

st.subheader("2º Teste - Geração de Clusters de livros de Emmanuel, André Luiz e Humberto de Campos")

st.markdown("""
Neste segundo teste incluímos um terceiro espírito autor, Humberto de Campos. A partir deste teste, além do filtro de 50.000 tokens, incluímos um filtro de quantidade máximo de 5 livros por autor. Isso porque a quantidade de livros de Chico Xavier atribuídas aos diferentes espíritos é muito desigual. No caso de Humberto de Campos, por exemplo, só temos 3 livros que possuem mais de 50.000 tokens, após os tratamentos. Para que o modelo consiga identificar padrões de uma forma mais justa é necessário termos uma quantidade próxima de livros por autor.
""")
            
st.image("teste2.JPG")

st.markdown("""
Novamente verificamos que apenas um livro de André Luiz ficou classificado no cluster indevido, todos os demais livros foram agrupados no cluster do seu respectivo espírito autor, segundo o espiritismo.""")
            

st.subheader("3º Teste - Geração de Clusters de livros de Emmanuel, André Luiz, Irmão X e de Humberto de Campos em vida e através do espírito")

st.markdown("""
Neste terceiro teste incluímos um quarto espírito autor, Irmão X. E além disso, também incluímos livros que Humberto de Campos escreveu em vida, além dos outros que havíamos incluido anteriormente que teriam sido escritos por ele através de Chico. 
""")
            
st.image("teste3.JPG")

st.markdown("""
Novamente verificamos que apenas um livro de André Luiz ficou classificado no cluster indevido, todos os demais livros foram agrupados no cluster do seu respectivo espírito autor, segundo o espiritismo. E para nossa surpresa, o modelo identificou semelhanças entre os livros escritos por Humberto de Campos em vida e os que foram ditado por ele e escritos através de Chico Xavier, segundo o espiritismo. De forma que considerou todos os livros atribuídos a Humberto no mesmo cluster, seja em vida ou através do seu espírito.""")            

st.subheader("Conclusão")

st.markdown("""
O estudo apresentado permite deduzir que existem padrões consistentes nas obras de Chico Xavier, padrões estes que possibilitam categorizar os livros de acordo com o suposto espírito que, de acordo com a doutrina espírita, teria orientado a criação da obra por meio de Chico Xavier. Esses padrões emergem até mesmo em modelos não supervisionados.

Contudo, é importante ressaltar que o estudo não pretende, nem acredito ser possível, afirmar definitivamente se foram ou não espíritos que inspiraram as obras. No entanto, as análises propiciam reflexões intrigantes.

Se existem padrões distintos nos livros, poderia Chico Xavier ter a capacidade de ouvir coisas que nós, meros mortais, não conseguimos? Ou, em contrapartida, poderia Chico Xavier ter sido hábil o suficiente para simular variados estilos literários, a ponto de replicar de maneira quase perfeita o estilo de um escritor já falecido? Mas qual seria o propósito disso, considerando que ele nunca obteve qualquer benefício material em troca?

A partir deste ponto, a ciência reconhece suas limitações, e a fé de cada indivíduo passa a direcionar as respostas.
""")

    

        
