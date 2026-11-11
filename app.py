#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App de Análise de Nado — Streamlit
==================================
UI web que empacota o analisador existente (swim_analyzer.py), sem reescrever
a lógica: o usuário seleciona o estilo com botões, envia o vídeo, e o app
identifica erros de movimento (com limiar, momento do vídeo e drill de
correção), além de gerar o relatório em PDF e JSON.

Executar:
    streamlit run app.py
E abrir o endereço indicado (padrão: http://localhost:8501) no navegador.
"""

import io
import json
import os
import sys
import tempfile
import traceback

import streamlit as st

# garante que o import do analisador funciona ao rodar `streamlit run`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import swim_analyzer as _sa
except Exception:
    # no cloud, um erro aqui aparece cortado nos logs — exibe o traceback
    # COMPLETO na própria tela para diagnóstico imediato
    st.set_page_config(page_title="Analisador de Nado — erro de importação")
    st.error("**Falha ao importar o pipeline `swim_analyzer`.**"
             "\n\nVerifique se TODOS os arquivos do pacote estão na raiz do "
             "repositório (app.py, swim_analyzer.py, requirements.txt, "
             "packages.txt) e se a versão do Python do app é 3.12."
             "\n\n**Traceback completo:**")
    st.code(traceback.format_exc())
    st.stop()

STYLE_CONFIG = getattr(_sa, "STYLE_CONFIG", None) or {
    "crawl": {"label": "Crawl (freestyle)"}, "peito": {"label": "Peito (breaststroke)"},
    "costas": {"label": "Costas (backstroke)"}, "borboleta": {"label": "Borboleta (butterfly)"},
}
SwimAnalyzer = getattr(_sa, "SwimAnalyzer", None)
gerar_pdf = getattr(_sa, "gerar_pdf", None)

# getattr blindado com FALLBACK inline: um swim_analyzer.py desatualizado no
# repositório não pode derrubar o app — a tabela de limiares v9 é recriada aqui
FALLBACK_THRESHOLDS = {
    "crawl": {
        "cotovelo_min": {"valor": 80.0, "unidade": "graus", "fonte": "Vanderbilt YSJ 2025",
                         "descricao": "Ângulo mínimo do cotovelo na recuperação (alto cotovelo)"},
        "cotovelo_max": {"valor": 100.0, "unidade": "graus", "fonte": "Vanderbilt YSJ 2025",
                         "descricao": "Ângulo máximo do cotovelo na recuperação"},
        "cadencia_min": {"valor": 55.0, "unidade": "cpm", "fonte": "Triathlete (elite 55–100 cpm)",
                         "descricao": "Cadência mínima de braçadas por lado"},
        "cadencia_max": {"valor": 75.0, "unidade": "cpm", "fonte": "Triathlete (elite 55–100 cpm)",
                         "descricao": "Cadência máxima de braçadas por lado (não-sprint)"},
        "tronco_max": {"valor": 20.0, "unidade": "graus", "fonte": "SwimSmart (cabeça neutra)",
                       "descricao": "Inclinação máxima do tronco (ombro–quadril vs. vertical)"},
        "simetria_min": {"valor": 0.80, "unidade": "razão", "fonte": "HEURÍSTICO",
                         "descricao": "Razão mínima de rotação entre os lados (1,00 = simétrico)"},
    },
    "peito": {
        "ciclo_min": {"valor": 1.6, "unidade": "s/ciclo", "fonte": "SwimNow (referência 200 m)",
                      "descricao": "Duração mínima do ciclo de braçada (deslize completo)"},
        "ciclo_max": {"valor": 1.8, "unidade": "s/ciclo", "fonte": "SwimNow (referência 200 m)",
                      "descricao": "Duração máxima do ciclo de braçada"},
        "deslize_max": {"valor": 22.0, "unidade": "% do ciclo", "fonte": "SwimSwam (elite ≤22%)",
                        "descricao": "Fração máxima do ciclo em deslize passivo"},
        "joelho_flex_max": {"valor": 165.0, "unidade": "graus", "fonte": "HEURÍSTICO",
                            "descricao": "Flexão máxima do joelho na recuperação da pernada"},
        "tronco_max": {"valor": 20.0, "unidade": "graus", "fonte": "HEURÍSTICO (mesma lógica do crawl)",
                       "descricao": "Inclinação máxima do tronco entre as respirações"},
    },
    "costas": {
        "rotacao_min": {"valor": 35.0, "unidade": "graus/lado", "fonte": "Swim Like A Fish / RocketSwim",
                        "descricao": "Rotação mínima do corpo por lado"},
        "rotacao_max": {"valor": 45.0, "unidade": "graus/lado", "fonte": "Swim Like A Fish / RocketSwim",
                        "descricao": "Rotação máxima do corpo por lado"},
        "cabeca_osc_max": {"valor": 8.0, "unidade": "% do torso", "fonte": "HEURÍSTICO",
                           "descricao": "Oscilação máxima da cabeça (desvio-padrão normalizado)"},
        "simetria_min": {"valor": 0.80, "unidade": "razão", "fonte": "HEURÍSTICO",
                         "descricao": "Razão mínima de rotação entre os lados"},
    },
    "borboleta": {
        "batidas_alvo": {"valor": 2.0, "unidade": "batidas/ciclo", "fonte": "USMS / Vasa",
                         "descricao": "Batidas de golfinho por ciclo de braços"},
        "batidas_tol": {"valor": 0.5, "unidade": "batidas/ciclo", "fonte": "HEURÍSTICO",
                        "descricao": "Tolerância aceita em torno do alvo de batidas"},
        "ondulacao_min": {"valor": 0.08, "unidade": "× torso", "fonte": "HEURÍSTICO",
                          "descricao": "Ondulação mínima do quadril (pico-a-pico / torso)"},
    },
}
DEFAULT_THRESHOLDS = getattr(_sa, "DEFAULT_THRESHOLDS", FALLBACK_THRESHOLDS)

# getattr blindado: um swim_analyzer.py desatualizado no repositório não
# pode derrubar o app por falta deste símbolo
VERSION = getattr(_sa, "VERSION", "versão desconhecida (swim_analyzer.py "
                                  "desatualizado — substitua-o pelo do pacote)")
if SwimAnalyzer is None or gerar_pdf is None:
    st.error("**`swim_analyzer.py` desatualizado no repositório.** O `app.py` "
             "atual precisa das funções `SwimAnalyzer`/`gerar_pdf` da v9+. "
             "Substitua **ambos** os arquivos (`app.py` e `swim_analyzer.py`) "
             "pelos do pacote mais recente e faça o Commit.")
    st.code(traceback.format_exc())
    st.stop()

st.set_page_config(page_title="Analisador de Nado", page_icon="🏊", layout="wide")

st.title("🏊 Analisador de Nado")
st.caption(
    "Selecione o estilo, envie um vídeo com vista lateral (câmera fixa, "
    "60 fps recomendado) e o app identifica erros de movimento na execução — "
    "cada erro vem com o limiar de referência, o momento aproximado do vídeo "
    "em que ocorre e um drill de correção. O relatório em PDF reúne tudo. "
    "Valores marcados como HEURÍSTICO não têm fonte externa citada."
)

# --- estado do resultado entre interações -----------------------------------
if "resultado" not in st.session_state:
    st.session_state.resultado = None
if "estilo" not in st.session_state:
    st.session_state.estilo = "crawl"


def analisar(video_bytes, filename, estilo_sel, limiares=None):
    """Salva o upload num arquivo temporário e roda o pipeline existente.
    `limiares` são os valores ajustados pelo usuário (None = padrão).
    Blindado: funciona com SwimAnalyzer novo (aceita thresholds=) e antigo
    (sem o parâmetro), e mostra o erro real na tela em vez de derrubar."""
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1] or ".mp4",
                                     delete=False) as tmp:
        tmp.write(video_bytes)
        caminho = tmp.name
    try:
        import inspect
        try:
            aceitos = inspect.signature(SwimAnalyzer).parameters
        except (TypeError, ValueError):
            aceitos = {}
        kwargs = {"style": estilo_sel}
        if "thresholds" in aceitos:
            kwargs["thresholds"] = limiares
        try:
            analisador = SwimAnalyzer(caminho, **kwargs)
        except TypeError as e:
            st.error("**Falha ao construir o analisador** — a assinatura de "
                     f"`SwimAnalyzer` no repositório não bate com o `app.py`. "
                     f"Detalhe: `{e}`. Assinatura detectada: "
                     f"`{inspect.signature(SwimAnalyzer)}`. "
                     "Substitua **ambos** os arquivos (`app.py` e "
                     "`swim_analyzer.py`) pelos do pacote mais recente e faça "
                     "o Commit.")
            st.code(traceback.format_exc())
            return None
        # construtor antigo (sem thresholds=): aplica os limiares como atributo
        if limiares and "thresholds" not in aceitos:
            analisador.limiares_usados = dict(limiares)
        try:
            n = analisador.process()
            if n == 0:
                st.error("Não foi possível ler nenhum frame desse vídeo. "
                         "Confira o formato (MP4/MOV/AVI) e se o arquivo não está corrompido.")
                return None
            rep = analisador.report()
            rep["video"] = filename
            rep["frames_analisados"] = n
            rep["fontes"] = {
                "cotovelo_crawl": "Vanderbilt YSJ 2025 (80–100°)",
                "cadencia": "Triathlete (elite 55–100 cpm)",
                "tronco": "SwimSmart (cabeça neutra)",
                "peito_cadencia": "SwimNow 1,6–1,8 s/ciclo (200 m)",
                "peito_deslize": "SwimSwam (elite ≤22% do ciclo)",
                "peito_pernada": "Human Kinetics (qualitativo)",
                "costas_rotacao": "Swim Like A Fish / RocketSwim (35–45°/lado)",
                "borboleta_batidas": "USMS / Vasa (2 por ciclo)",
                "heuristico": "valores sem fonte externa estão marcados",
            }
            ts = analisador.timeseries()
            keyframes = analisador.keyframes
        except Exception:
            # qualquer falha nos passos de análise mostra o traceback completo
            # na tela (no cloud, os logs ficam truncados)
            st.error("**Falha durante a análise do vídeo.** Traceback completo:")
            st.code(traceback.format_exc())
            return None

        pdf_buf = io.BytesIO()
        gerar_pdf(rep, keyframes, ts, pdf_buf)
        json_buf = io.StringIO()  # json.dump escreve str: BytesIO quebra com TypeError
        import json as _json
        _json.dump(rep, json_buf, ensure_ascii=False, indent=2)
        return rep, pdf_buf.getvalue(), json_buf.getvalue()
    finally:
        os.unlink(caminho)


# --- seleção do estilo (botões) ----------------------------------------------
st.subheader("1. Selecione o estilo do nado")
ROTULOS = {
    "crawl": "🏊‍♂️ Crawl (Livre)",
    "peito": "🐸 Peito",
    "costas": "🏊‍♀️ Costas",
    "borboleta": "🦋 Borboleta",
}
cols = st.columns(4)
for i, key in enumerate(("crawl", "peito", "costas", "borboleta")):
    if cols[i].button(ROTULOS[key],
                      use_container_width=True,
                      type="primary" if st.session_state.estilo == key else "secondary"):
        st.session_state.estilo = key
if st.session_state.estilo in STYLE_CONFIG:
    st.success(f"Estilo selecionado: **{STYLE_CONFIG[st.session_state.estilo]['label']}**"
               " — os erros de movimento serão avaliados com os limiares deste estilo.")

# --- barra lateral -----------------------------------------------------------
with st.sidebar:
    st.header("Configurações")
    st.caption(f"Pipeline {VERSION}")
    st.divider()
    st.markdown("**Protocolo de filmagem recomendado**\n"
                "- Vista lateral, câmera fixa no tripé\n"
                "- Altura da água, ~5 m de distância\n"
                "- 10–15 s de nado contínuo, sem virada\n"
                "- 60 fps; luz frontal (sol de costas para a câmera)")

# --- tabela de limiares ajustáveis por estilo --------------------------------
st.subheader("2. Ajuste os limiares (opcional)")
st.caption("A tabela abaixo define os limiares usados na identificação de erros "
           "para o estilo selecionado. Edite a coluna **valor** conforme o nível "
           "do nadador — os erros passam a ser avaliados com os valores ajustados.")

_PADRAO = DEFAULT_THRESHOLDS[st.session_state.estilo]
_KEY_ED = f"limiares_ed_{st.session_state.estilo}"
if _KEY_ED not in st.session_state:
    st.session_state[_KEY_ED] = {}

_linhas = []
for k, info in _PADRAO.items():
    _linhas.append({
        "parâmetro": k,
        "valor": float(st.session_state[_KEY_ED].get(k, info["valor"])),
        "unidade": info["unidade"],
        "fonte": info["fonte"],
        "descrição": info["descricao"],
    })

_editado = st.data_editor(
    _linhas,
    key=f"editor_{st.session_state.estilo}",
    disabled=("parâmetro", "unidade", "fonte", "descrição"),
    column_config={
        "valor": st.column_config.NumberColumn("valor", format="%.2f", step=0.5),
        "descrição": st.column_config.TextColumn("descrição", width="large"),
    },
    num_rows="fixed",
    hide_index=True,
    use_container_width=True,
)
# st.data_editor retorna DataFrame no app, mas lista simples fora dele
# (bare mode) — tratar os dois formatos
_registros = (_editado.to_dict("records")
              if hasattr(_editado, "to_dict") else list(_editado))
st.session_state[_KEY_ED] = {r["parâmetro"]: float(r["valor"])
                             for r in _registros}

_ajustados = {k: v for k, v in st.session_state[_KEY_ED].items()
              if abs(v - _PADRAO[k]["valor"]) > 1e-9}
_c1, _c2, _c3 = st.columns(3)
if _c1.button("↩️ Restaurar padrão deste estilo"):
    st.session_state[_KEY_ED] = {}
    st.rerun()
_exporte = json.dumps(st.session_state[_KEY_ED], ensure_ascii=False, indent=2)
_c2.download_button("⬇️ Exportar limiares (JSON)", _exporte,
                    f"limiares_{st.session_state.estilo}.json", "application/json")
_json_imp = _c3.file_uploader("⬆️ Importar limiares (JSON)", type=["json"],
                              key=f"imp_{st.session_state.estilo}")
if _json_imp is not None:
    try:
        _imp = json.loads(_json_imp.getvalue().decode("utf-8"))
        _validos = {k: float(v) for k, v in _imp.items() if k in _PADRAO}
        st.session_state[_KEY_ED] = _validos
        st.success(f"{len(_validos)} limiar(es) importado(s) para "
                   f"{STYLE_CONFIG[st.session_state.estilo]['label']}.")
        st.rerun()
    except Exception:
        st.error("JSON inválido — use o formato exportado pelo botão acima.")
if _ajustados:
    st.info("Ajustados (fora do padrão): " + ", ".join(
        f"**{k}** = {v:g}" for k, v in _ajustados.items()))
else:
    st.caption("Todos os limiares estão nos valores padrão.")

# --- upload -------------------------------------------------------------------
st.subheader("3. Envie o vídeo")
arquivo = st.file_uploader("Vídeo do nado", type=["mp4", "mov", "avi", "mkv"])

st.subheader("4. Analise")
if arquivo and st.button("Analisar nado", type="primary"):
    with st.spinner("Processando vídeo (pose por frame + métricas + PDF)..."):
        res = analisar(arquivo.getvalue(), arquivo.name, st.session_state.estilo,
                       limiares=st.session_state.get(_KEY_ED) or None)
    if res:
        st.session_state.resultado = res

res = st.session_state.resultado
if res:
    rep, pdf_bytes, json_bytes = res
    m = rep["metricas"]

    # --- cabeçalho do resultado
    st.subheader(f"Resultado — estilo: {STYLE_CONFIG[rep['estilo']]['label']}")
    st.metric("Frames analisados", rep["frames_analisados"])

    # --- limiares usados + erros de movimento identificados
    st.subheader("⚠️ Erros de movimento identificados")
    with st.expander("📋 Limiares usados nesta análise"):
        st.dataframe([{"parâmetro": k, "valor": v["valor"], "unidade": v["unidade"],
                       "fonte": v["fonte"], "descrição": v["descricao"]}
                      for k, v in rep.get("limiares", {}).items()],
                     hide_index=True, use_container_width=True)
    erros = rep.get("erros", [])
    if erros:
        st.error(f"{len(erros)} erro(s) de movimento encontrado(s) na execução "
                 "— corrija com os drills sugeridos:")
        for i, e in enumerate(erros, 1):
            quando = (f" — ocorre em ~{e['quando_s']:.1f} s do vídeo"
                      if e.get("quando_s") is not None else "")
            with st.expander(f"{i}. {e['titulo']}{quando}", expanded=(i == 1)):
                c1, c2 = st.columns(2)
                c1.metric("O que foi medido", e["medido"])
                c2.metric("Limiar de referência", e["limiar"])
                st.markdown(f"**Como corrigir (drill):** {e['drill']}")
                st.markdown(f"**Fonte do limiar:** {e['fonte']}")
    else:
        st.success("Nenhum erro de movimento identificado com os limiares atuais "
                   "— os movimentos medidos estão dentro das faixas de referência "
                   "(ou os frames válidos foram insuficientes para medir; veja as "
                   "métricas abaixo).")

    # --- métricas
    st.subheader("Métricas")
    cols = st.columns(4)
    destaques = [k for k, v in m.items()
                 if isinstance(v, (int, float)) and k not in ("duracao_s", "torso_px")]
    for i, k in enumerate(destaques):
        cols[i % 4].metric(k, m[k])
    with st.expander("Todas as métricas (incl. não numéricas)"):
        st.json(m)

    # --- feedback resumido
    st.subheader("Feedback resumido")
    for linha in rep["feedback"]:
        st.markdown(f"- {linha}")

    # --- downloads
    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Baixar relatório (PDF)", pdf_bytes,
                       "relatorio_nado.pdf", "application/pdf",
                       use_container_width=True)
    d2.download_button("⬇️ Baixar dados (JSON)", json_bytes,
                       "relatorio_nado.json", "application/json",
                       use_container_width=True)
elif arquivo:
    st.info("Clique em **Analisar nado** para processar o vídeo enviado.")
