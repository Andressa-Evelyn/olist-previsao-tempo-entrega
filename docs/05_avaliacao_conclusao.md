# 5. Avaliação do Modelo e Conclusão

Esta etapa avalia a capacidade dos modelos de prever o tempo de entrega dos pedidos da Olist, compara seus desempenhos em dados não utilizados durante o treinamento e justifica a escolha do modelo final. A avaliação utiliza os resultados registrados em `docs/figures/04_treinamento/tabelas/4_resultados_parciais_teste.csv`.

### 5.1. Protocolo de avaliação

A avaliação final foi realizada sobre o conjunto de teste, composto por **19.020 pedidos (20% da base)**. Esses registros foram separados antes do treinamento e não participaram do ajuste dos hiperparâmetros. Os modelos avaliados são as versões otimizadas de Ridge, Decision Tree, KNN e Random Forest. Também foram incluídos dois baselines: a mediana do tempo de entrega observada no conjunto de treino e o prazo estimado informado pela Olist.

A seleção e o ajuste dos modelos foram conduzidos anteriormente por validação cruzada de cinco folds, restrita ao conjunto de treino. Assim, o conjunto de teste funciona como uma avaliação final em dados não vistos, permitindo comparar as alternativas sem utilizá-lo na escolha dos hiperparâmetros.

### 5.2. Métricas utilizadas

Foram utilizadas três métricas principais de regressão:

- **MAE:** calcula a média dos valores absolutos das diferenças entre o tempo real e o previsto. É expresso em dias e representa o tamanho médio do erro, sem considerar sua direção. Quanto menor, melhor.
- **RMSE:** calcula a raiz da média dos erros ao quadrado. Também é expresso em dias, mas penaliza mais fortemente erros de grande magnitude. Quanto menor, melhor.
- **R²:** indica quanto da variabilidade do tempo real de entrega é explicada pelas previsões do modelo. Quanto mais próximo de 1, melhor. Um valor negativo indica desempenho inferior ao de um preditor constante baseado na média do alvo, considerando a definição usual dessa métrica.

Como métricas complementares, também foram registrados o **MedAE** (erro absoluto mediano), o viés médio das previsões e o percentual de pedidos em que a previsão ficou a até três ou cinco dias do valor real.

### 5.3. Benchmark dos modelos no conjunto de teste

A tabela apresenta os resultados dos seis métodos avaliados. Os valores de erro estão em dias. Para MAE e RMSE, quanto menor, melhor o desempenho.

| Modelo                      | MAE (dias) | RMSE (dias) |        R² | Erro mediano (dias) | Erro até 5 dias |
| --------------------------- | ---------: | ----------: | --------: | ------------------: | --------------: |
| **Random Forest otimizado** |   **4,00** |    **5,54** | **0,440** |            **3,00** |      **73,08%** |
| KNN otimizado               |       4,18 |        5,78 |     0,391 |                3,15 |          71,38% |
| Decision Tree otimizada     |       4,18 |        5,79 |     0,388 |                3,10 |          70,39% |
| Ridge otimizado             |       4,26 |        6,24 |     0,290 |                3,27 |          69,51% |
| Dummy (mediana)             |       5,49 |        7,63 |    -0,060 |                4,01 |          60,02% |
| Prazo estimado da Olist     |      12,57 |       14,63 |    -2,906 |               12,17 |          13,01% |

O **Random Forest otimizado apresentou o melhor desempenho geral**: obteve o menor MAE e RMSE e o maior R² entre todos os métodos avaliados. Seu erro absoluto mediano foi de três dias, e 73,08% dos pedidos tiveram erro absoluto de até cinco dias. Além disso, 50,02% das previsões ficaram a até três dias do tempo real.

Os resultados também mostram que todos os quatro modelos de aprendizado de máquina superaram os dois baselines no conjunto de teste. O Dummy, que prevê sempre a mediana do tempo de entrega do treino, apresentou MAE de 5,49 dias. Já o prazo estimado da Olist apresentou MAE de 12,57 dias e R² negativo (-2,906), indicando desempenho bastante inferior ao dos modelos treinados para este conjunto de dados.

### 5.4. Justificativa da escolha do modelo final

O **Random Forest otimizado foi selecionado como modelo final** por apresentar a melhor combinação das métricas consideradas. Em relação ao segundo colocado, o KNN, reduziu o MAE de 4,18 para 4,00 dias e o RMSE de 5,78 para 5,54 dias, além de apresentar R² superior (0,440 contra 0,391).

A escolha também é sustentada pela validação cruzada de cinco folds realizada no conjunto de treino. Nessa etapa, o Random Forest otimizado obteve RMSE médio de **5,6001 ± 0,0564 dias**, MAE médio de **4,02 dias** e R² médio de **0,4393**. No conjunto de teste, os resultados foram RMSE de **5,5425 dias**, MAE de **3,9968 dias** e R² de **0,4398**. A proximidade entre os resultados da validação cruzada e do teste é um indicativo de desempenho consistente nessa avaliação, sem evidência de uma queda relevante de generalização entre essas duas etapas.

Outro ponto favorável é a comparação com o prazo estimado da Olist. O MAE caiu de 12,5681 para 3,9968 dias, uma redução aproximada de **68,2%**. Essa comparação indica que, nos pedidos avaliados, o modelo desenvolvido produziu estimativas consideravelmente mais próximas dos tempos reais de entrega do que o prazo de referência disponível na base.

### 5.5. Análise dos resultados e limitações

Apesar do melhor desempenho relativo, o R² de **0,440** indica que o Random Forest explica aproximadamente 44% da variabilidade observada nos tempos de entrega. Portanto, ainda existe uma parcela expressiva da variação que não é explicada pelas variáveis utilizadas. O MAE de aproximadamente quatro dias também mostra que as previsões não devem ser interpretadas como datas garantidas de entrega.

A diferença entre RMSE (5,54 dias) e MAE (4,00 dias) indica que erros de maior magnitude influenciam o resultado e merecem investigação. A distribuição dos erros e os percentuais de tolerância complementam a média: embora 73,08% dos pedidos tenham erro de até cinco dias, os demais pedidos ainda podem apresentar desvios relevantes.

Também foi identificada uma anomalia no conjunto de teste que afeta especialmente o Ridge: um pedido apresenta `prazo_postagem_dias` igual a 1.052 dias. Nesse registro, o Ridge produziu uma previsão de aproximadamente 331,9 dias, elevando seu RMSE para 6,24 dias. Esse caso evidencia a sensibilidade de modelos lineares a valores extremos. Os modelos baseados em árvores e vizinhos não apresentaram o mesmo impacto nesse registro.

Os resultados devem ser interpretados considerando que a avaliação foi feita por uma divisão aleatória de treino e teste. Essa estratégia mede o desempenho em pedidos separados da mesma base histórica, mas não reproduz integralmente um cenário de implantação em que o modelo precisa prever pedidos de períodos futuros, sujeitos a mudanças operacionais e sazonais.

### 5.6. Possíveis melhorias e trabalhos futuros

Com base nas métricas e nas limitações observadas, são propostas as seguintes evoluções:

1. **Aprimorar a qualidade dos dados:** investigar e tratar registros extremos ou inconsistentes, como o prazo de postagem de 1.052 dias identificado na avaliação. O tratamento deve ser orientado por regras de negócio e não apenas pela remoção automática de valores altos, pois alguns atrasos podem ser reais.
2. **Adicionar variáveis logísticas e contextuais:** avaliar a inclusão de informações sobre transportadoras, modalidade de envio, feriados, condições climáticas e características das rotas, desde que estejam disponíveis no momento em que a previsão for realizada. Isso pode ajudar a explicar parte da variabilidade ainda não capturada pelo modelo.
3. **Investigar o desempenho por segmentos:** calcular métricas separadas por estado, região, vendedor, distância e faixa de tempo de entrega. A avaliação agregada pode esconder grupos nos quais o modelo apresenta erros significativamente maiores.
4. **Testar outros algoritmos de ensemble:** comparar o Random Forest com métodos de Gradient Boosting, como XGBoost, LightGBM ou HistGradientBoosting, mantendo o mesmo protocolo de validação e teste.
5. **Ampliar a busca de hiperparâmetros:** explorar espaços de busca adicionais e avaliar se os ganhos justificam o custo computacional. A comparação deve continuar sendo feita por validação cruzada, preservando o conjunto de teste para a avaliação final.
6. **Realizar validação temporal:** em uma evolução do projeto, treinar com pedidos de períodos anteriores e testar em períodos posteriores. Essa estratégia aproxima a avaliação das condições reais de implantação e permite observar efeitos de sazonalidade e mudanças na operação.
7. **Estimar incerteza das previsões:** investigar intervalos de previsão ou estimativas por quantis, para que a aplicação possa apresentar uma faixa provável de entrega em vez de comunicar apenas um valor pontual.

### 5.7. Conclusão

Com base nos experimentos realizados, conclui-se que o **Random Forest otimizado é o modelo mais adequado entre as alternativas avaliadas para prever o tempo de entrega dos pedidos da Olist**. Ele apresentou MAE de aproximadamente quatro dias, RMSE de 5,54 dias e R² de 0,440 no conjunto de teste, superando Ridge, Decision Tree, KNN e os dois baselines utilizados na comparação.

O resultado mais expressivo foi a redução de aproximadamente 68,2% no erro absoluto médio em relação ao prazo estimado da Olist presente na base. Além disso, cerca de 73% dos pedidos tiveram previsões com erro de até cinco dias. Esses resultados indicam que o modelo aprendeu relações úteis entre as características dos pedidos e seus tempos de entrega.

Assim, o modelo atende ao objetivo do projeto como **ferramenta de apoio à estimativa logística**, podendo auxiliar no planejamento e na comunicação de prazos mais próximos dos tempos observados. Entretanto, o desempenho ainda não permite tratar a previsão como garantia de entrega: o R² demonstra capacidade explicativa moderada e os erros podem ser relevantes em determinados pedidos ou segmentos. Antes de uma utilização operacional, é recomendado validar o modelo em períodos futuros, monitorar seus erros e incorporar novas informações logísticas. Portanto, a solução é promissora como apoio à decisão, mas ainda requer validação e evolução para uma implantação em produção.

### 5.8. Rastreabilidade dos resultados

Os resultados apresentados podem ser conferidos e reproduzidos por:

- `src/train.py`: cálculo das métricas (`metricas_teste`), avaliação dos baselines e modelos no conjunto de teste e geração dos resultados.
- `docs/figures/04_treinamento/tabelas/4_resultados_parciais_teste.csv`: métricas finais dos seis métodos avaliados.
- `docs/figures/04_treinamento/tabelas/1_baselines_e_modelos_cv.csv`: resultados da validação cruzada de cinco folds.
- `data/processed/predicoes_teste.csv`: valores reais e previsões individuais usados na análise dos erros, gerado durante a execução do treinamento.
- `docs/04_treinamento.md`: documentação do treinamento, dos hiperparâmetros e dos resultados intermediários.

Para regenerar as métricas, figuras e tabelas, com os dados processados disponíveis, execute:

```bash
python src/train.py
```
