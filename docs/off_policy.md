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
- **Política de registro π_β:** desconhecida; será estimada. Candidata natural: próximo item dado o último item,
  a mesma tabela de k = 1 do P_E. O termo de entropia do DORL é a entropia dessa política de registro.

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

O estado (`which_tracker avg`, `reward_handle cat`) é a média dos embeddings dos últimos 3 itens, concatenada com as
recompensas. No log real, essas recompensas serão o watch_ratio registrado, normalizado como no ambiente de teste.
Conferir essa normalização ao escrever o carregador.

## Próximos passos

1. Feito em 14/11: validação com π_β dependente do contexto (seção acima).
2. **Retreino com checkpoint** (`roda_ckpt.sh`): MBPO, MOPO, DORL e DORL-const, `--is_save`, 5 sementes; mantém só
   a última época e confere que os logs são idênticos aos atuais. Na 4090, depois da varredura do *survival bonus*.
3. **Dados reais:** sequências do big_matrix (usuários fora da avaliação, itens do small_matrix, sem duplicatas,
   sessões de 30 min). Estimar π_β(a | último item) e aplicar os estimadores às políticas retreinadas.
4. **Comparação:** ordem dos métodos sob OPE contra a ordem no simulador (R_each em NX_0 e NX_10).

Referência: a avaliação off-policy (IPS, doubly robust) está na literatura âncora do TCC (`docs/contexto_tcc.md`,
seção 3). Os nomes de artigos específicos ainda não foram fixados.
