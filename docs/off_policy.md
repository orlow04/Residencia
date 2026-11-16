# Avaliação off-policy (OPE) — branch `off-policy`

Iniciado em 14/11/2026. Objetivo: estimar o valor das políticas treinadas a partir do log real, sem o simulador e sem a
regra de saída, e verificar se a ordem dos métodos se mantém (objetivos O1/O2 do `docs/protocolo_avaliacao.md`).

## Viabilidade (verificado em 14/11)

- **Usuários.** Os 1.411 usuários de avaliação não têm nenhuma interação com os 3.327 itens do small_matrix no
  big_matrix (0 de 533.090 linhas). Os demais usuários têm 9,45 milhões de linhas com esses itens. O OPE usa esses
  outros usuários.
- **Estado da política.** O estado não usa embedding de usuário (`use_userEmbedding=False`, `run_Policy_Main.py:77`),
  só o histórico de itens. A política pode, portanto, ser aplicada às sequências de qualquer usuário.
- **Checkpoints.** As políticas não foram salvas (`--is_save` desligado; `util/utils.py:147`). Elas precisam ser
  retreinadas com checkpoint. O treino é determinístico (`DORL_pelog_s2` = `DORL_s2`), então os logs devem sair
  idênticos aos atuais.

## Desenho

- **Unidade:** um passo (bandit contextual). Contexto = histórico recente de itens; ação = item; recompensa =
  watch_ratio limitado a 5, a mesma escala da avaliação. Em nível de sessão, com 3.327 ações, o produto dos pesos
  explode.
- **Estimadores** (`ope.py`): IPS, SNIPS, IPS com corte, modelo direto (DM) e doubly robust (DR). Diagnóstico: ESS e
  maior peso.
- **Política de registro π_β:** desconhecida; é estimada pelo próximo item dado o último item. É o mesmo objeto
  cuja entropia o P_E de k = 1 do DORL mede, embora as tabelas não sejam idênticas (a do DORL usa todos os itens e
  não corta sessões).

## Validação semi-sintética (`ope.py --validacao` → `results/ope/ope_validacao.txt`)

O small_matrix é totalmente observado, então o valor verdadeiro de qualquer política é conhecido.
- **Contexto:** o usuário. **Política de registro:** 0,95·popularidade (big_matrix, usuários fora da avaliação) +
  0,05·uniforme.
- **Políticas-alvo:** variam a concentração. São elas: uniforme; k itens fixos ao acaso (k = 300, 100, 30, 10);
  os k mais populares; e o oráculo top-10 por usuário.
- **Repetições:** 50, com logs de 10⁵ e 10⁶ passos.

Resultado (fato):
1. **IPS, SNIPS e DR recuperam o valor verdadeiro** em todos os alvos, com π_β conhecida ou estimada pelas contagens.
   - Viés de no máximo 0,025 com 10⁶ passos (oráculo, IPS).
   - Ordem dos 8 alvos preservada em 100% dos pares.
   - O SNIPS tem RMSE menor que o IPS em todos os alvos concentrados (oráculo, 10⁵ passos: 0,056 contra 0,292).
2. **O DM só funciona para políticas não personalizadas.** Com q̂ = média do item no log, ele acerta os alvos iguais
   para todos os usuários, mas erra o oráculo por −2,98 (valor verdadeiro 4,33). O DR corrige esse erro.
3. **O corte dos pesos em 100 enviesa os alvos concentrados**: −0,64 em fixos_10, −3,0 no oráculo. Só usar corte
   escolhido pelo ESS, e reportar o viés.
4. **A concentração derruba o ESS.** Com 10⁵ passos: 76.661 (uniforme) contra 265 (10 itens fixos) e 217 (oráculo).
   Políticas que concentram em poucos itens, como o DORL (cobertura de 2–3%), exigem logs grandes; o log real tem
   9,45 milhões de passos.

Limites desta validação (o que o caso real tem e ela não):
- π_β não depende do contexto aqui, então estimá-la pelas contagens está corretamente especificado. No caso real,
  π_β depende do histórico, e o erro de estimação entra no viés.
- A recompensa é determinística dado (usuário, item).
- O contexto é o usuário, não uma sequência.

## Validação com π_β dependente do contexto (`ope_contexto.py` → `results/ope/ope_contexto.txt`)

- **Contexto:** (usuário, último item).
- **Política de registro verdadeira:** a transição real do big_matrix, 0,95·transição + 0,05·uniforme. Usa
  8.768.473 pares consecutivos na mesma sessão (usuários fora da avaliação, itens do small_matrix).
- **π_β nos pesos, três versões:**
  - verdadeira;
  - contagens (último item → item), estimadas no próprio log;
  - marginal, que ignora o contexto (mal especificada).
- **Políticas-alvo:** uniforme; 30 fixos; oráculo top-10; segue o registro (10 próximos mais frequentes);
  100 fixos evitando a categoria do último item (imita o RL).
- **Repetições:** 20, com logs de 10⁶ e 10⁷ passos.

Resultado (fato; viés com 10⁷ passos):

| π_β nos pesos | IPS | SNIPS | DR |
|---|---|---|---|
| Verdadeira | ≤ 0,006 em todos os alvos | ≤ 0,003 | ≤ 0,003 |
| Contagens (estimada) | −0,57 a −0,60; oráculo −2,88 | ≤ 0,005 | oráculo −1,98; demais ≤ 0,001 |
| Marginal (mal especificada) | segue o registro +18,95; cat. diferente −0,057 | ≤ 0,008 | ≤ 0,005 |

1. **Com π_β estimada, o IPS falha e o SNIPS não.**
   - A tabela de contagens tem 3.327² células e é esparsa. Todo par registrado tem contagem ≥ 1, o que superestima
     π_β nas ações observadas e encolhe os pesos.
   - O viés diminui com o tamanho do log (−0,80 com 10⁶; −0,57 com 10⁷), mas não some.
   - O SNIPS cancela o fator comum ao normalizar pelos pesos.
2. **O DR herda o problema dos pesos encolhidos** quando o modelo direto está errado (oráculo: −1,98).
3. **π_β mal especificada é catastrófica para o IPS** quando a política-alvo se correlaciona com o contexto do
   registro (+18,95 em "segue o registro"). O SNIPS fica a 0,008.
4. **Escolha para o caso real:** SNIPS com π_β por contagens (último item → item), reportando ESS. IPS e DR entram
   como diagnóstico, não como estimativa principal.

Limites: a recompensa depende só do usuário, não do último item; no caso real, ela também depende do histórico.

## A política no teste (verificado no código)

O `A2CPolicy_withEmbedding.forward` (`src/core/policy/a2c2.py:150–185`) segue estes passos:
- obtém a distribuição sobre os itens da saída do *actor* (`Categorical`, saída *softmax* do `Actor` do tianshou);
- remove os itens já recomendados quando o protocolo é NX;
- **sorteia** a ação: `deterministic_eval` falso e sem `exploration_noise` no teste.

π_e(a|s) é, portanto, a saída do *actor*, conhecida exatamente: é o que o OPE precisa.

O estado (`StateTrackerAvg2.forward`, `state_tracker2.py:138`; `which_tracker avg`, `reward_handle cat`,
`window_size 3`) é construído assim, no teste:
- **t = 0:** [embedding de preenchimento, 1]. É o ramo `reset`, e a recompensa inicial 1 não é normalizada.
- **t ≥ 1:** média das entradas [embedding do item, recompensa normalizada] dos min(t, 3) passos anteriores. Se t < 3,
  entra também a entrada de preenchimento, com recompensa 1 normalizada.
- **Normalização:** min((r − r_min) / (r_max − r_min), 1), com r_min e r_max da matriz de avaliação (0 e 5).

O `ope_politicas.py` reimplementa isso a partir do checkpoint, sem importar classes do DORL. O
`tests/test_ope_politicas.py` compara estado e π_e(·|s) com o `StateTrackerAvg2` e o `Actor` reais numa sessão de
7 passos: diferença 0. Duas mutações do teste (t errado; recompensa errada) falham, como esperado.

## Log real (`ope_dados.py` → `results/ope/ope_dados.txt`; `ope_log.npz` não versionado, 206 MB)

- **Sessões:** intervalo > 30 min na sequência completa do usuário; dentro da sessão ficam só os itens do
  small_matrix, e t é a posição nessa sequência filtrada.
- **Tamanho:** 9.454.659 passos, 623.916 sessões, 5.765 usuários; os 3.327 itens aparecem.
- **Sessão filtrada:** mediana 5 itens, média 15,2, p90 36. 37,2% dos passos estão em sessões de até 30 itens.
- **Posição:** 6,6% dos passos em t = 0; 16,1% em t < 3 (estado com preenchimento).
- **Valor da política de registro:** 0,881 (recompensa média registrada); a média do small_matrix é 0,872.
- **π_β por contagens** (β = 10): mediana 9,9·10⁻⁴; 23,3% dos passos com t ≥ 1 são pares vistos uma única vez. É o
  regime esparso em que o IPS fica enviesado e o SNIPS não (seção anterior).
- **ESS de uma política-alvo uniforme:** 3,9 milhões.

## Aplicação às políticas (`ope_politicas.py`)

Para cada checkpoint `CK_*`:
- **Estimativas:** SNIPS (principal) com IC de 95% por *bootstrap* de usuários; SNIPS só com t ≥ 3; SNIPS com π_β
  marginal; IPS, DR e DM (diagnóstico); ESS e maior peso.
- **Sem pesos**, a política nos contextos reais: nº efetivo de itens por passo (entropia), itens para 95% da
  exposição, Gini e fração da exposição nos 10% de itens mais frequentes. Mede a concentração fora do simulador.
- **Referências:** política de registro e uniforme.
- **Agregação:** média ± dp entre sementes.

Ensaio com checkpoints falsos (pesos aleatórios), só para testar o fluxo: 47 s por checkpoint na CPU do Mac. Os
checkpoints são pequenos, então a aplicação pode rodar no Mac se eles forem copiados da 4090.

Limites conhecidos:
- **Contexto filtrado.** O histórico só tem itens do small_matrix; os outros itens que o usuário viu entre eles
  são ignorados.
- **Estado fora da distribuição de treino.** A política foi treinada com históricos gerados por ela mesma no
  simulador; aqui recebe históricos reais.
- **Sem máscara.** A política é avaliada sem remover itens já vistos, como no FB.
- **Só um passo.** O valor é por passo; a duração da sessão não é estimada.

### Resultado (16/11/2026)

Arquivos: `results/ope/ope_saida.txt` e `ope_por_semente.csv`; resumo e cruzamento com o simulador em
`results/ope/ope_comparacao.txt` (`ope_compara.py`). São 20 checkpoints, todos da época 200. Antes de rodar, o
`ope_politicas.py` foi corrigido: a função `estado` criava dois tensores sem dispositivo e falhava em CUDA. Em CPU o
estado é idêntico ao da versão anterior, e o `tests/test_ope_politicas.py` continua com diferença 0.

Referências: política de registro 0,881; uniforme 0,898 (IC95 0,893–0,903), ESS 3,9 milhões.

| Método | SNIPS (média ± dp) | Mediana | Mín–máx | ESS | Itens efetivos por passo | Itens para 95% da exposição |
|---|---|---|---|---|---|---|
| MBPO | 0,367 ± 0,309 | 0,278 | 0,134–0,903 | 1.225–2.911 | 1,03–1,19 | 2–4 |
| MOPO | 0,350 ± 0,271 | 0,331 | 0,125–0,804 | 215–2.806 | 1,13–1,38 | 3–4 |
| DORL | 0,639 ± 0,528 | 0,389 | 0,201–1,459 | 1.106–1.554 | 1,16–1,47 | 2–4 |
| DORL-const | 0,630 ± 0,633 | 0,384 | 0,250–1,757 | 1.507–2.427 | 1,17–2,25 | 4–16 |

Fatos:

1. **As políticas são quase determinísticas e quase independentes do contexto real.** Em 9,45 milhões de contextos,
   cada política concentra 95% da exposição em 2 a 16 dos 3.327 itens; o Gini da exposição fica entre 0,997 e 1,000
   (política de registro: 0,244).
2. **O OPE tem pouco suporte.** O ESS vai de 215 a 2.911 passos (0,002% a 0,03% do log), contra 3,9 milhões da
   uniforme. O SNIPS quase coincide com o DM, que não usa pesos (Spearman 0,98; diferença mediana 0,022). Na prática,
   a estimativa é a recompensa média registrada dos poucos itens que cada semente prefere.
3. **A ordem dos métodos não é identificável.** A maior diferença entre médias (0,29) não supera o dp entre sementes
   (0,27 a 0,63). As medianas ficam entre 0,28 e 0,39; as médias de DORL e DORL-const são puxadas por uma semente
   cada (1,459 e 1,757).
4. **A maioria das políticas fica abaixo da política de registro.** O SNIPS é menor que 0,881 em 17 de 20; o IC95
   fica inteiro abaixo em 15 e inteiro acima em 2 (DORL s4, DORL-const s3). No simulador, o R_each dos quatro métodos
   (0,74 a 0,78 em NX_0) também fica abaixo do acaso (0,872).
5. **Concordância com o simulador, política a política:** Spearman +0,57 entre o SNIPS e o R_each em NX_0
   (p = 0,01, n = 20, sem correção para 8 comparações); +0,35 em NX_10 e +0,38 no FB (p ≥ 0,10).
6. **Popularidade dos itens preferidos:** DORL e DORL-const põem no máximo 15% da exposição nos 10% de itens mais
   frequentes do log, nas 10 sementes; MBPO e MOPO põem mais de 50% em 5 de 10. A diferença entre as médias fica
   dentro do dp (a distribuição entre sementes é bimodal). Política de registro: 18,5%.

Leitura:

- O OPE não confirma nem refuta a ordem dos métodos do simulador: o ruído entre sementes é maior que qualquer
  diferença entre métodos.
- O que ele mostra, sem simulador e sem regra de saída, é a concentração (fato 1) e o valor por passo abaixo da
  referência na maioria das sementes (fato 4). Os dois vão na mesma direção dos achados no simulador (cobertura
  baixa; R_each abaixo do acaso).
- O IC95 cobre só a amostragem de usuários. Não cobre o erro de π_β estimada nem o suporte pequeno (fato 2); os
  dois casos acima da referência não sustentam, sozinhos, que essas políticas são melhores que a de registro.
- O fato 5 é exploratório: a correlação junta diferenças entre métodos e entre sementes.

Hipótese, não testada: a variedade das políticas no simulador sob NX vem da máscara de itens já recomendados, e não
da dependência do estado. É coerente com o FB, em que elas repetem 6,7% a 74,3% dos passos (`docs/semana10.md`).
Um OPE com máscara dos itens já vistos na sessão não foi feito.

A confirmar pelo Gabriel: que o `roda_ckpt.sh` imprimiu "idêntico ao original" nas 20 execuções (`ck_fila*.out`, na
4090). Sem isso, a correspondência entre os checkpoints e as execuções originais não está verificada.

## Próximos passos

1. Feito em 14/11: validação com π_β dependente do contexto (seção acima).
2. Feito em 16/11: retreino com checkpoint (`roda_ckpt.sh`; 20 checkpoints da época 200). Conferência dos logs
   contra os originais: a confirmar (ver acima).
3. Feito em 16/11: `ope_politicas.py` com os 20 checkpoints (seção Resultado).
4. Feito em 16/11: comparação com o simulador (`ope_compara.py`). A ordem dos métodos não é identificável sob OPE.
5. Em aberto: decidir se este ramo entra no `main` e no relatório.

Leituras para entender o código: `docs/off_policy_leitura.md`.
