# Changelog — Analisador de Nado

## v9-limiares-ajustaveis (2026-11-11)

### Novo
- **Tabela de limiares ajustáveis por estilo** (novo passo 2 do app): os
  limiares que antes eram fixos no código agora aparecem numa tabela
  editável (`st.data_editor`), filtrada pelo estilo selecionado nos botões.
  - Colunas: parâmetro, valor (editável), unidade, fonte, descrição.
  - Edição persistida por estilo em `st.session_state`.
  - Botão **"Restaurar padrão deste estilo"**.
  - **Exportar/Importar** limiares em JSON (validado contra os parâmetros
    conhecidos do estilo).
  - Os valores ajustados são passados ao `SwimAnalyzer` via novo parâmetro
    `thresholds=` e **passam a reger a identificação de erros** — provado
    em teste: subir `tronco_max` de 20° para 70° removeu o erro de tronco;
    baixar para 5° mantém o erro com o campo "limiar" exibindo 5°.
- O relatório (tela, PDF e JSON) inclui agora o bloco **`limiares`**: a
  tabela efetiva usada naquela análise (valor, unidade, fonte, descrição).
- Novos limiares explicitados na tabela que antes estavam hardcoded:
  `simetria_min` (0,80), `joelho_flex_max` (165°), `batidas_tol` (±0,5).

### Alterado
- `swim_analyzer.py`: limiares movidos para `DEFAULT_THRESHOLDS` (dict por
  estilo com valor/unidade/fonte/descrição); `STYLE_CONFIG` interno agora é
  derivado da tabela (`_cfg_from_flat`), aplicando os ajustes do usuário.
  `SwimAnalyzer.__init__` aceita `thresholds=None` (padrão) e `set_style`
  reconstrói a cfg preservando os ajustes. Nenhum limiar hardcoded restou
  no caminho de análise (verificado por busca).
- `app.py`: passo 2 novo (tabela editável), análise recebe os limiares
  editados, seção de erros mostra expander com a tabela de limiares usados.

### Mantido
- Botões de seleção de estilo (v8), correções v5–v7 (JSON `StringIO`,
  blindagem de import, `packages.txt` sem comentários, `runtime.txt`
  python-3.12).

## v8-selecao-estilo
- Removida a autodetecção de estilo; seleção por botões (crawl, peito,
  costas, borboleta); seção "Erros de movimento identificados" com limiar,
  momento do vídeo e drill de correção.

## v7-apt-fix
- `packages.txt` sem comentários (o apt do cloud tratava cada palavra como
  pacote); `VERSION` importado com `getattr` blindado.

## v5/v6
- Correção `TypeError` do JSON (`StringIO`), sanitização de NaN, feedback
  blindado, traceback completo na tela em caso de falha de importação.
