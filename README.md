# 🏊 Analisador de Nado — Deploy no Streamlit Community Cloud

App web que analisa vídeos de natação nos 4 estilos (crawl, peito, costas,
borboleta), detecta o estilo automaticamente e gera relatório em PDF.

## ⚠️ Configuração crítica

O **MediaPipe só tem wheels para Python 3.9–3.12** — o Streamlit Cloud
vém usando Python 3.13 por padrão e o deploy **falha** se você não escolher
**Python 3.12** em "Advanced settings" durante o deploy.

## Passo a passo

### 1. Suba a pasta para um repositório GitHub

Crie um repositório (público, plano gratuito) e envie **todos os arquivos
desta pasta**, mantendo a estrutura:

```
seu-repo/
├── app.py                  # app Streamlit (arquivo principal)
├── swim_analyzer.py        # pipeline (MediaPipe → métricas → feedback → PDF)
├── requirements.txt        # dependências (versões testadas, sem conflitos)
├── packages.txt            # libs de sistema (apt) que o cloud instala
├── .streamlit/
│   └── config.toml         # upload até 400 MB
└── README.md
```

Pela interface do GitHub: "Add file" → "Upload files" → arraste os arquivos
(crie a pasta `.streamlit` nomeando um arquivo como `.streamlit/config.toml`).
Ou pela linha de comando:

```bash
git init && git add -A && git commit -m "deploy analisador de nado"
git remote add origin https://github.com/SEU_USUARIO/analisador-nado.git
git push -u origin main
```

### 2. Deploy em share.streamlit.io

1. Acesse [share.streamlit.io](https://share.streamlit.io) e entre com GitHub/Google
2. "Create app" / "New app" → "Deploy a public app from GitHub"
3. Repository: `SEU_USUARIO/analisador-nado` · Branch: `main`
4. Main file path: **`app.py`**
5. **App URL**: escolha um nome (ex.: `analisador-nado`)
6. **Clique em "Advanced settings" → Python version: 3.12** ← obrigatório
7. "Deploy!" — o build leva 5–10 min (MediaPipe é pesado)

### 3. Usar pelo celular

1. Abra a URL gerada (ex.: `https://analisador-nado.streamlit.app`) no
   navegador do celular — de qualquer rede (Wi-Fi ou 4G/5G)
2. **Dica:** use "Adicionar à Tela de Início" no navegador do celular —
   o app abre como se fosse aplicativo instalado
3. Filme o nadador pelo app de câmera (vista lateral, câmera fixa,
   10–15 s, 60 fps) e envie o vídeo pela opção "Vídeo do nado"
4. Baixe o relatório em PDF direto no celular

## Limitações no plano gratuito

- ~1 GB de RAM: clipes de 10–15 s passam; vídeos longos podem falhar
- O app "dorme" após inatividade e reativa no próximo acesso
- O vídeo do nadador trafega pela nuvem — considere a privacidade

## Atualizar o app

Basta fazer push das mudanças no GitHub — o cloud recompila automaticamente.
