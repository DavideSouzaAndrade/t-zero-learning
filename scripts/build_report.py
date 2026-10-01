"""Builds report/relatorio_a2c.pdf (A4, <= 3 pages) with reportlab."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, KeepTogether

FORK = "https://github.com/DavideSouzaAndrade/t-zero-learning"

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.2, leading=10.2,
                      alignment=4, spaceAfter=2.5)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=15, alignment=0, spaceAfter=2)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=9.6, leading=12, alignment=0,
                    spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1a3d6d"))
small = ParagraphStyle("s", parent=body, fontSize=7.2, leading=9, alignment=0, textColor=colors.HexColor("#444444"))
cap = ParagraphStyle("c", parent=small, alignment=1, spaceAfter=3)

W = A4[0] - 3.0 * cm


def P(t, st=body):
    return Paragraph(t, st)


def fig(path, caption, ratio):
    return KeepTogether([Image(path, width=W, height=W * ratio), P(caption, cap)])


def table(rows, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.0),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.0),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


s = []
s.append(P("Atividade: Policy Gradient com Actor-Critic (A2C) — CartPole-v1", h1))
s.append(P(f"Aluno: Davi de Souza Andrade &nbsp;&nbsp;|&nbsp;&nbsp; Fork com <font face='Courier'>algorithms/a2c.py</font> e "
           f"<font face='Courier'>networks/discrete_actor_critic.py</font> completos: <link href='{FORK}' color='blue'>{FORK}</link>", small))
s.append(P("<b>Implementação.</b> Parte 1: retorno de n passos calculado de trás para frente, "
           "R<sub>t</sub> = r<sub>t</sub> + γ(1−d<sub>t</sub>)R<sub>t+1</sub>, R<sub>T</sub> = V(s<sub>T</sub>), por coluna. "
           "Parte 2: perda −média(log π(a|s)·A) com A = (R − V).detach() (ou R.detach() sem baseline) — o peso é constante para o "
           "gradiente do ator, senão a perda do ator puxaria o crítico. Parte 3: <font face='Courier'>Categorical(logits)</font>, "
           "amostra ou argmax, log_prob da ação dada, entropia e V(s). Os 14 testes de <font face='Courier'>tests/test_a2c.py</font> passam. "
           "<b>Protocolo.</b> 500 mil passos, config <font face='Courier'>a2c_cartpole</font> + <font face='Courier'>--override</font>, "
           "2 seeds (1 e 2; linha cheia = s1, tracejada = s2) em <i>todas</i> as configurações (20 runs). Os gráficos usam exatamente as "
           "métricas que o harness envia ao wandb (mesmos nomes), registradas localmente. Baseline (n_envs=8, n=5, ent=0,01): avaliação final "
           "500 ± 0 nas duas seeds, retorno médio ≥ 450 aos 148–200 mil passos.", body))

# ---------------- Q1 ----------------
s.append(P("Q1 — Número de atores em paralelo (num_envs)", h2))
s.append(P("<b>Previsão.</b> Com 1 ambiente o lote vem de um único trecho de trajetória, muito correlacionado: espero um "
           "<i>policy_loss</i> muito mais ruidoso e um aprendizado instável. Com 64 ambientes o gradiente fica suave, mas, como o lote "
           "cresce, cabem menos atualizações em 500 mil passos; espero SPS crescendo com num_envs.", body))
s.append(P("<b>Varredura.</b> num_envs ∈ {1, 8, 64} e um controle <b>num_envs=1, num_steps=40</b>, que tem o <i>mesmo</i> lote (40) e o mesmo "
           "número de atualizações (12.500) do baseline: só muda a origem das transições (1 trajetória × 8 independentes).", body))
s.append(table([
    ["config", "lote", "atualizações", "dp policy_loss", "retorno final (last100)", "eval final", "SPS"],
    ["num_envs=1", "5", "100.000", "8,4 / 8,8", "382 / 350", "459 / 457", "890"],
    ["num_envs=1, n=40", "40", "12.500", "10,5 / 8,8", "395 / 409", "370 / 286", "1.320"],
    ["num_envs=8 (base)", "40", "12.500", "1,6 / 1,4", "466 / 474", "500 / 500", "3.770"],
    ["num_envs=64", "320", "1.562", "0,5 / 0,5", "153 / 146", "172 / 158", "7.100"],
], [3.2 * cm, 1.2 * cm, 2.0 * cm, 2.4 * cm, 3.4 * cm, 2.2 * cm, 1.6 * cm]))
s.append(Spacer(1, 3))
s.append(fig("report/figs/q1.png", "Fig. 1 — Q1. O painel do desvio móvel do policy_loss quantifica o ruído do gradiente seguido.", 2.05 / 7.2))
s.append(P("<b>Explicação.</b> Há dois efeitos misturados em num_envs, e o controle os separa. <b>(i) Correlação.</b> Com o mesmo lote e o "
           "mesmo número de passos de gradiente do baseline, o controle de 1 ambiente tem um <i>policy_loss</i> 6× mais ruidoso (desvio de "
           "~9–10 contra ~1,5) e fica preso em ~400, com avaliação de 286–370. Num trecho de 40 passos de uma mesma trajetória os estados são "
           "quase iguais e as vantagens têm o mesmo sinal (se o poste está caindo, todas são negativas). Os 40 termos então não se "
           "cancelam: o lote é, na prática, uma única amostra do gradiente, enviesada para aquele trecho. Os picos de −10 a −100 no "
           "<i>policy_loss</i> aparecem quando um episódio termina dentro do lote. Com 8 atores em pontos diferentes de episódios "
           "diferentes, a média é de estimativas quase independentes, e a variância cai. É o papel que o replay buffer cumpre no DQN e que "
           "um método on-policy não pode usar: esse é o argumento central do A3C, e a previsão se confirmou. <b>(ii) Tamanho do lote × número "
           "de atualizações.</b> num_envs=64 tem o gradiente mais limpo (desvio de 0,5), mas recebe só 1.562 passos de Adam com lr=7e-4 e "
           "estaciona em ~150: o limite passa a ser o número de atualizações, não a qualidade de cada uma. Isso eu não previ: esperava apenas "
           "\"mais lento\", e o efeito é forte. Com mais timesteps ou um lr maior o resultado tende a mudar. Já num_envs=1 com n=5 faz "
           "100 mil atualizações minúsculas e ruidosas e ainda chega a eval ≈ 458, mas com quedas bruscas (até ~20 de retorno). "
           "<b>(iii) Tempo.</b> O SPS sobe de 890 (≈9,5 min) para 3.770 (≈2,2 min) e 7.100 (≈1,2 min): um forward da rede sobre o lote de N "
           "observações custa quase o mesmo que sobre 1, então o paralelismo dilui o custo fixo por passo. Ou seja, o paralelismo compra "
           "tempo de parede e decorrelação ao mesmo tempo, mas, para um orçamento fixo de passos, num_envs grande demais custa atualizações.", body))

# ---------------- Q2 ----------------
s.append(P("Q2 — Horizonte do retorno de n passos (a2c.num_steps)", h2))
s.append(P("<b>Previsão.</b> n=1 (quase só bootstrap) tem baixa variância e alto viés, porque herda o erro de V. n grande (quase Monte Carlo) "
           "tem pouco viés e alta variância, então espero <i>value_loss</i> maior e <i>explained_variance</i> menor; e o ator piora quando o "
           "crítico piora. <b>Varredura.</b> n ∈ {1, 5, 32, 128}, com lote de 8 / 40 / 256 / 1.024 e 62.500 / 12.500 / 1.953 / 488 atualizações.", body))
s.append(table([
    ["config", "value_loss médio", "explained_var. média", "advantage_mean médio", "retorno final", "eval final"],
    ["n=1", "22 / 26", "0,75 / 0,50", "−0,5", "488 / 297", "500 / 455"],
    ["n=5 (base)", "36 / 32", "0,14 / 0,49", "−0,5", "466 / 474", "500 / 500"],
    ["n=32", "73 / 88", "0,11 / 0,14", "+5", "493 / 265", "500 / 309"],
    ["n=128", "502 / 497", "0,007 / 0,007", "+26", "445 / 333", "431 / 422"],
], [2.4 * cm, 2.7 * cm, 3.0 * cm, 3.2 * cm, 2.4 * cm, 2.3 * cm]))
s.append(Spacer(1, 3))
s.append(fig("report/figs/q2.png", "Fig. 2 — Q2. Value_loss e explained_variance suavizados por média móvel (15 logs).", 2.05 / 7.2))
s.append(P("<b>Explicação.</b> O alvo de n=1, r + γV(s'), contém uma única recompensa amostrada, então tem pouca variância; mas seu erro é o "
           "erro de V(s'), ou seja, viés. Com n=128 o alvo soma até 128 recompensas que dependem de uma política estocástica e do instante em "
           "que o poste cai: viés quase nulo e variância grande. Os gráficos confirmam a previsão. Com n=128 o <i>value_loss</i> fica em ~500 e a "
           "<i>explained_variance</i> em ≈0 durante todo o treino: o crítico não explica nada. Para isso contribuem os alvos ruidosos e "
           "também o fato de ele só receber 488 passos de gradiente (o lote mudou). O <i>advantage_mean</i> de +26 mostra que V subestima "
           "sistematicamente os retornos e está sempre atrasado. Como o ator usa A = R − V, um crítico que não explica nada deixa A ≈ R − const: "
           "o baseline deixa de reduzir a variância (o mesmo mecanismo da Q4), e o aprendizado fica lento e quase linear. "
           "<b>A armadilha:</b> com n=1 a <i>explained_variance</i> é a maior da varredura (0,5–0,75), muitas vezes perto de 1. Mas o alvo é "
           "1 + γV(s') e, no CartPole, V(s') ≈ V(s) em estados consecutivos: a variância do alvo é quase toda a variância da própria saída do "
           "crítico. A EV alta mede autoconsistência (V concorda com V deslocado), não acerto em relação ao retorno verdadeiro; um V "
           "uniformemente errado teria EV ≈ 1 do mesmo jeito. Prova disso é a seed 1 de n=1, que despencou para retorno ~10 aos 160 mil "
           "passos com EV perto de 1 no mesmo período, e a seed 2, que caiu de 500 para ~300 no fim. Observação: depois da convergência (retorno 500) muitos lotes "
           "não têm nenhum término, os alvos ficam quase constantes e o <i>value_loss</i> cai a ~1e-4 enquanto a EV fica mal definida "
           "(var(y) ≈ 0); daí as oscilações bimodais em n=1 e n=5, que não indicam piora do crítico. O compromisso n=5 é o único estável nas duas seeds.", body))

# ---------------- Q3 ----------------
s.append(P("Q3 — Coeficiente de entropia (a2c.ent_coef)", h2))
s.append(P("<b>Previsão.</b> Com ent_coef=0,1 a entropia deve ficar perto de ln 2 ≈ 0,69 e o retorno deve estacionar, porque a política não pode "
           "ficar quase determinística. Com ent_coef=0 a entropia cai mais rápido, com risco de colapso prematuro, embora o CartPole talvez tolere. "
           "<b>Varredura.</b> ent_coef ∈ {0, 0,01, 0,1}, 2 seeds cada. "
           "<b>Resultado.</b> ent=0: entropia final ≈ 0,47, retorno ≥ 450 aos 122 e 141 mil passos, eval 500/500 (o mais rápido). "
           "ent=0,01: entropia 0,54, eval 500/500. ent=0,1: entropia ≈ 0,60, retorno preso em 346–377, eval 390/415.", body))
s.append(fig("report/figs/q3.png", "Fig. 3 — Q3. Retorno médio e entropia da política para os três coeficientes.", 2.05 / 7.2))
s.append(P("<b>Explicação.</b> O bônus de entropia impede que a política fique determinística cedo demais, antes que o crítico e as vantagens "
           "sejam confiáveis. Num método on-policy isso é grave, porque a política só aprende com as ações que ela mesma amostra: se uma ação "
           "deixa de ser amostrada, nunca mais recebe gradiente. Com ent_coef=0,1 o termo −c·H compete com o gradiente de política. "
           "Perto de uma boa política as vantagens ficam pequenas (|A| ~ 2–3), e então o gradiente da entropia domina e segura a política "
           "em ~60/40 em vez de ~95/5. As ações \"erradas\" amostradas de vez em quando derrubam o poste, e o retorno estaciona: a política "
           "é ótima para o objetivo regularizado, não para o retorno (previsão confirmada). Com ent_coef=0 o CartPole funcionou nas duas "
           "seeds, e até mais rápido. O ambiente é benevolente: são só 2 ações, a recompensa é densa (+1 a cada passo) e uma ação ruim se "
           "manifesta em poucos passos. A inicialização do ator (std 0,01) começa uniforme e a própria estocasticidade inicial explora "
           "tudo o que existe para explorar. Além disso, com baseline as vantagens são centradas em ~0 e empurram nos dois sentidos, por "
           "isso a entropia não colapsou (0,47; compare com a Q4). Num ambiente com muitas ações ou recompensa rara (p. ex. LunarLander, "
           "com o grande bônus terminal), eu esperaria que, sem bônus, a política se fixasse cedo na primeira ação medíocre que dá algum "
           "retorno (p. ex. ficar pairando) e nunca descobrisse a recompensa rara; o bônus passa a ser necessário.", body))

# ---------------- Q4 ----------------
s.append(P("Q4 — Ablação do baseline (a2c.use_baseline=false)", h2))
s.append(P("<b>Previsão.</b> Sem baseline o peso passa a ser R, que é positivo e da ordem de dezenas: <i>advantage_mean</i> muito acima de 0, "
           "<i>advantage_std</i> bem maior e treino mais ruidoso. Com pesos sempre positivos, toda ação amostrada é reforçada, e espero que "
           "a entropia colapse. <b>Varredura.</b> use_baseline ∈ {true, false}, 2 seeds. "
           "<b>Resultado.</b> Média sobre o treino: <i>advantage_mean</i> de 50–70 (contra −0,5), <i>advantage_std</i> de 8,6–14,8 (contra 2,6), "
           "entropia de 0,06–0,15 aos 100 mil passos e 0,004 no fim (contra 0,54). Retorno: a seed 1 chega a 330 e degrada para 86 (eval 105); "
           "a seed 2 chega a 500 aos 206 mil passos, despenca para ~150 aos 290 mil e termina em 354 (eval 480).", body))
s.append(fig("report/figs/q4.png", "Fig. 4 — Q4. advantage_* = pesos efetivamente usados no gradiente (R − V com baseline; R sem). Média móvel de 15 logs.", 2.05 / 7.2))
s.append(P("<b>Explicação.</b> Como Σ<sub>a</sub>∇π(a|s) = 0, temos E[∇log π · b(s)] = 0: subtrair b(s) = V(s) não muda o gradiente "
           "esperado, mas o estimador amostral passa de log π·R para log π·(R − V), e sua variância depende do segundo momento dos pesos. "
           "Os gráficos de <i>advantage_*</i> confirmam isso: com baseline os pesos são centrados (média ≈ 0) e têm desvio ~2,6; sem ele, "
           "têm média 50–70 e desvio 3–6× maior. A maior parte do peso é um termo comum a todas as ações, que não carrega informação e só "
           "acrescenta ruído. A entropia mostra a consequência. No CartPole toda recompensa é +1, logo R > 0 sempre, e cada atualização "
           "aumenta log π de <i>todas</i> as ações amostradas. A ação já mais provável é amostrada mais vezes e por isso é mais reforçada "
           "(\"rich get richer\"). Em esperança o termo comum se cancela, mas com lotes finitos ele é um empurrão aleatório e forte rumo ao "
           "determinismo, e com pesos ~50 o bônus de 0,01·H é irrelevante. A entropia cai a 0,004 (política determinística) e, como "
           "∇log π → 0 para a ação de probabilidade ≈ 1, o agente não consegue mais explorar nem se corrigir: a seed 1 trava numa política "
           "ruim e a seed 2, que tinha chegado a 500, desaba e não se recupera. Previsão confirmada: o baseline é o que torna o gradiente "
           "útil, e no CartPole a ausência dele afeta mais a estabilidade (colapso de entropia) do que a velocidade inicial.", body))
s.append(P("Reprodução: <font face='Courier'>python scripts/run_logged.py logs/X.jsonl --config a2c_cartpole --override seed=S ...</font> "
           "(lista em <font face='Courier'>scripts/jobs.txt</font>); gráficos: <font face='Courier'>python scripts/make_plots.py</font>.", small))

doc = SimpleDocTemplate("report/relatorio_a2c.pdf", pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                        topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="Relatório A2C", author="")
doc.build(s)
