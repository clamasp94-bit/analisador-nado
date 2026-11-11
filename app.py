#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App de Análise de Nado — Streamlit
==================================
UI web que empacota o analisador existente (swim_analyzer.py), sem reescrever
a lógica: upload de vídeo -> autodetecção de estilo -> métricas + feedback
-> download do relatório em PDF e JSON.

Executar:
    streamlit run app.py
E abrir o endereço indicado (padrão: http://localhost:8501) no navegador.
"""

import io
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

STYLE_CONFIG = _sa.STYLE_CONFIG
SwimAnalyzer = _sa.SwimAnalyzer
gerar_pdf = _sa.gerar_pdf
# getattr blindado: um swim_analyzer.py desatualizado no repositório não
# pode derrubar o app por falta deste símbolo
VERSION = getattr(_sa, "VERSION", "versão desconhecida (swim_analyzer.py "
                                  "desatualizado — substitua-o pelo do pacote)")

st.set_page_config(page_title="Analisador de Nado", page_icon="🏊", layout="wide")

st.title("🏊 Analisador de Nado")
st.caption(
    "Envie um vídeo com vista lateral (câmera fixa, 60 fps recomendado). "
    "O app detecta o estilo (crawl, peito, costas ou borboleta), calcula "
    "métricas e gera o relatório em PDF. Valores marcados como HEURÍSTICO "
    "não têm fonte externa citada."
)

# --- estado do resultado entre interações -----------------------------------
if "resultado" not in st.session_state:
    st.session_state.resultado = None


def analisar(video_bytes, filename, estilo_sel):
    """Salva o upload num arquivo temporário e roda o pipeline existente."""
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1] or ".mp4",
                                     delete=False) as tmp:
        tmp.write(video_bytes)
        caminho = tmp.name
    try:
        analisador = SwimAnalyzer(caminho, style=estilo_sel)
        n = analisador.process()
        if n == 0:
            st.error("Não foi possível ler nenhum frame desse vídeo. "
                     "Confira o formato (MP4/MOV/AVI) e se o arquivo não está corrompido.")
            return None
        det = None
        if estilo_sel == "auto":
            det = analisador.detect_style()
            analisador.set_style(det["estilo"])
        rep = analisador.report()
        if det:
            rep["autodeteccao"] = det
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

        pdf_buf = io.BytesIO()
        gerar_pdf(rep, keyframes, ts, pdf_buf)
        json_buf = io.StringIO()  # json.dump escreve str: BytesIO quebra com TypeError
        import json as _json
        _json.dump(rep, json_buf, ensure_ascii=False, indent=2)
        return rep, pdf_buf.getvalue(), json_buf.getvalue()
    finally:
        os.unlink(caminho)


# --- barra lateral -----------------------------------------------------------
with st.sidebar:
    st.header("Configurações")
    st.caption(f"Pipeline {VERSION}")
    estilo_sel = st.selectbox(
        "Estilo do nado",
        ["auto", *STYLE_CONFIG],
        format_func=lambda s: "Detectar automaticamente" if s == "auto" else STYLE_CONFIG[s]["label"],
        help="Em 'auto', o estilo é detectado por heurísticas explicáveis "
             "(sincronia dos braços, orientação do corpo e ondulação do quadril); "
             "confiança baixa (< 0,15) cai para crawl com aviso.",
    )
    st.divider()
    st.markdown("**Protocolo de filmagem recomendado**\n"
                "- Vista lateral, câmera fixa no tripé\n"
                "- Altura da água, ~5 m de distância\n"
                "- 10–15 s de nado contínuo, sem virada\n"
                "- 60 fps; luz frontal (sol de costas para a câmera)")

# --- upload -------------------------------------------------------------------
arquivo = st.file_uploader("Vídeo do nado", type=["mp4", "mov", "avi", "mkv"])

if arquivo and st.button("Analisar nado", type="primary"):
    with st.spinner("Processando vídeo (pose por frame + métricas + PDF)..."):
        res = analisar(arquivo.getvalue(), arquivo.name, estilo_sel)
    if res:
        st.session_state.resultado = res

res = st.session_state.resultado
if res:
    rep, pdf_bytes, json_bytes = res
    m = rep["metricas"]

    # --- cabeçalho do resultado
    det = rep.get("autodeteccao")
    if det:
        cor = "orange" if det["fallback"] else "green"
        icon = "⚠️" if det["fallback"] else "✅"
        st.subheader(f"{icon} Estilo detectado: {STYLE_CONFIG[det['estilo']]['label']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Confiança", f"{det['confianca']:.2f}")
        c2.metric("Fallback p/ crawl", "sim" if det["fallback"] else "não")
        c3.metric("Frames analisados", rep["frames_analisados"])
        with st.expander("Como a detecção funcionou (sinais e escores)"):
            st.write("**Sinais brutos:**")
            st.json(det["sinais"])
            st.write("**Escores por estilo:**")
            st.bar_chart({k: [v] for k, v in det["escores"].items()})
            if det["fallback"]:
                st.warning("Confiança baixa: os sinais não discriminaram bem o "
                           "estilo. O relatório usa crawl como padrão — se souber "
                           "o estilo, selecione-o na barra lateral.")
    else:
        st.subheader(f"Estilo analisado: {STYLE_CONFIG[rep['estilo']]['label']}")
        st.metric("Frames analisados", rep["frames_analisados"])

    # --- métricas
    st.subheader("Métricas")
    cols = st.columns(4)
    destaques = [k for k, v in m.items()
                 if isinstance(v, (int, float)) and k not in ("duracao_s", "torso_px")]
    for i, k in enumerate(destaques):
        cols[i % 4].metric(k, m[k])
    with st.expander("Todas as métricas (incl. não numéricas)"):
        st.json(m)

    # --- feedback
    st.subheader("Feedback")
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
