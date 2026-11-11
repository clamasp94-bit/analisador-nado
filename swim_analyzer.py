#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisador de Nado — 4 estilos (crawl, peito, costas, borboleta)
================================================================
Pipeline: vídeo -> MediaPipe Pose (landmarks 2D) -> segmentação por ciclo
-> métricas por estilo -> regras de feedback com limiares citados
-> relatório (console + JSON).

Uso:
    python swim_analyzer.py <video.mp4> [--estilo crawl|peito|costas|borboleta]
                            [--exibir] [--saida relatorio.json] [--pdf relatorio.pdf]

Dependências: pip install mediapipe==0.10.14 opencv-python numpy reportlab matplotlib

Limitações (documentadas): vista LATERAL acima da água; parte do ciclo fica
submersa; refração/splash podem degradar a detecção — landmarks com
visibilidade baixa são descartados. Valores marcados como HEURÍSTICO não
têm fonte externa citada e servem apenas como referência inicial.
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    sys.exit("mediapipe não instalado. Rode: pip install mediapipe==0.10.14 opencv-python numpy")

# Versão do pipeline — exibida na barra lateral do app; o valor no ar
# confirma qual código está realmente rodando no Streamlit Cloud.
VERSION = "v11-chamada-blindada"


# ----------------------------------------------------------------------------
# TABELA DE LIMIARES PADRÃO POR ESTILO (ajustáveis pelo usuário no app)
# ----------------------------------------------------------------------------
# Cada limiar: valor padrão, unidade, descrição e fonte.
# "HEURÍSTICO" = sem fonte externa citada.
DEFAULT_THRESHOLDS = {
    "crawl": {
        "cotovelo_min": {"valor": 80.0, "unidade": "graus",
                         "fonte": "Vanderbilt YSJ 2025",
                         "descricao": "Ângulo mínimo do cotovelo na recuperação (alto cotovelo)"},
        "cotovelo_max": {"valor": 100.0, "unidade": "graus",
                         "fonte": "Vanderbilt YSJ 2025",
                         "descricao": "Ângulo máximo do cotovelo na recuperação"},
        "cadencia_min": {"valor": 55.0, "unidade": "cpm",
                         "fonte": "Triathlete (elite 55–100 cpm)",
                         "descricao": "Cadência mínima de braçadas por lado"},
        "cadencia_max": {"valor": 75.0, "unidade": "cpm",
                         "fonte": "Triathlete (elite 55–100 cpm)",
                         "descricao": "Cadência máxima de braçadas por lado (não-sprint)"},
        "tronco_max": {"valor": 20.0, "unidade": "graus",
                       "fonte": "SwimSmart (cabeça neutra)",
                       "descricao": "Inclinação máxima do tronco (ombro–quadril vs. vertical)"},
        "simetria_min": {"valor": 0.80, "unidade": "razão",
                         "fonte": "HEURÍSTICO",
                         "descricao": "Razão mínima de rotação entre os lados (1,00 = simétrico)"},
    },
    "peito": {
        "ciclo_min": {"valor": 1.6, "unidade": "s/ciclo",
                      "fonte": "SwimNow (referência 200 m)",
                      "descricao": "Duração mínima do ciclo de braçada (deslize completo)"},
        "ciclo_max": {"valor": 1.8, "unidade": "s/ciclo",
                      "fonte": "SwimNow (referência 200 m)",
                      "descricao": "Duração máxima do ciclo de braçada"},
        "deslize_max": {"valor": 22.0, "unidade": "% do ciclo",
                        "fonte": "SwimSwam (elite ≤22%)",
                        "descricao": "Fração máxima do ciclo em deslize passivo"},
        "joelho_flex_max": {"valor": 165.0, "unidade": "graus",
                            "fonte": "HEURÍSTICO",
                            "descricao": "Flexão máxima do joelho na recuperação da pernada"},
        "tronco_max": {"valor": 20.0, "unidade": "graus",
                       "fonte": "HEURÍSTICO (mesma lógica do crawl)",
                       "descricao": "Inclinação máxima do tronco entre as respirações"},
    },
    "costas": {
        "rotacao_min": {"valor": 35.0, "unidade": "graus/lado",
                        "fonte": "Swim Like A Fish / RocketSwim",
                        "descricao": "Rotação mínima do corpo por lado"},
        "rotacao_max": {"valor": 45.0, "unidade": "graus/lado",
                        "fonte": "Swim Like A Fish / RocketSwim",
                        "descricao": "Rotação máxima do corpo por lado"},
        "cabeca_osc_max": {"valor": 8.0, "unidade": "% do torso",
                           "fonte": "HEURÍSTICO",
                           "descricao": "Oscilação máxima da cabeça (desvio-padrão normalizado)"},
        "simetria_min": {"valor": 0.80, "unidade": "razão",
                         "fonte": "HEURÍSTICO",
                         "descricao": "Razão mínima de rotação entre os lados"},
    },
    "borboleta": {
        "batidas_alvo": {"valor": 2.0, "unidade": "batidas/ciclo",
                         "fonte": "USMS / Vasa",
                         "descricao": "Batidas de golfinho por ciclo de braços"},
        "batidas_tol": {"valor": 0.5, "unidade": "batidas/ciclo",
                        "fonte": "HEURÍSTICO",
                        "descricao": "Tolerância aceita em torno do alvo de batidas"},
        "ondulacao_min": {"valor": 0.08, "unidade": "× torso",
                          "fonte": "HEURÍSTICO",
                          "descricao": "Ondulação mínima do quadril (pico-a-pico / torso)"},
    },
}

# Metadados estruturais por estilo (não ajustáveis pelo usuário)
STYLE_META = {
    "crawl": {"label": "Crawl (freestyle)", "cycle_mode": "alternate", "min_cycle_s": 0.6},
    "costas": {"label": "Costas (backstroke)", "cycle_mode": "alternate", "min_cycle_s": 0.6},
    "peito": {"label": "Peito (breaststroke)", "cycle_mode": "symmetric", "min_cycle_s": 1.0},
    "borboleta": {"label": "Borboleta (butterfly)", "cycle_mode": "symmetric", "min_cycle_s": 0.6},
}

# Mapa: limiar ajustável (chave plana da tabela) -> posição na cfg interna
FLAT_TO_CFG = {
    "crawl": {"cotovelo_min": ("elbow_band", 0), "cotovelo_max": ("elbow_band", 1),
              "cadencia_min": ("sr_band", 0), "cadencia_max": ("sr_band", 1),
              "tronco_max": "trunk_max", "simetria_min": "symmetry_min"},
    "peito": {"ciclo_min": ("sec_por_bracada", 0), "ciclo_max": ("sec_por_bracada", 1),
              "deslize_max": "glide_max_pct", "joelho_flex_max": "knee_flex_max",
              "tronco_max": "trunk_max"},
    "costas": {"rotacao_min": ("rot_band", 0), "rotacao_max": ("rot_band", 1),
               "cabeca_osc_max": "head_std_max", "simetria_min": "symmetry_min"},
    "borboleta": {"batidas_alvo": ("kicks_por_ciclo", 0), "batidas_tol": "kicks_tol",
                  "ondulacao_min": "undulation_min"},
}

# Rótulo de fonte das faixas internas (mantém a citação no feedback)
BAND_SRC = {
    "crawl": {"elbow_band": "VAN", "sr_band": "TRI"},
    "peito": {"sec_por_bracada": "BNB"},
    "costas": {"rot_band": "SLF"},
    "borboleta": {"kicks_por_ciclo": "USM"},
}


def _cfg_from_flat(style, overrides=None):
    """Monta a cfg interna do estilo a partir da tabela de limiares padrão,
    aplicando overrides planos {param: valor} quando fornecidos. Sempre cria
    um dicionário novo (a cfg de uma análise nunca contamina outra)."""
    if style not in DEFAULT_THRESHOLDS:
        sys.exit(f"Estilo desconhecido: {style}. Opções: {list(DEFAULT_THRESHOLDS)}")
    over = overrides or {}
    cfg = {}
    for k, info in DEFAULT_THRESHOLDS[style].items():
        v = float(over.get(k, info["valor"]))
        alvo = FLAT_TO_CFG[style][k]
        if isinstance(alvo, tuple):
            key, idx = alvo
            cfg.setdefault(key, [None, None])
            cfg[key][idx] = v
        else:
            cfg[alvo] = v
    for key, src in BAND_SRC.get(style, {}).items():
        if cfg.get(key) is not None:
            if key == "kicks_por_ciclo":
                cfg[key] = (cfg[key][0], src)
            else:
                cfg[key] = (cfg[key][0], cfg[key][1], src)
    for k in ("elbow_band", "sr_band", "rot_band", "kicks_por_ciclo"):
        cfg.setdefault(k, None)
    cfg.update(STYLE_META[style])
    return cfg


# cfg padrão derivada da tabela (compatibilidade com o CLI)
STYLE_CONFIG = {s: _cfg_from_flat(s) for s in DEFAULT_THRESHOLDS}


def angle_2d(a, b, c):
    """Ângulo interno em graus entre vetores b->a e b->c (2D)."""
    ba, bc = np.array(a[:2]) - np.array(b[:2]), np.array(c[:2]) - np.array(b[:2])
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na * nc < 1e-9:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def _local_peaks(ys, fps, min_gap_s):
    """Índices de mínimos locais de y (pico de elevação) com separação mínima."""
    valid = [(i, y) for i, y in enumerate(ys) if y is not None]
    peaks = []
    for k in range(2, len(valid) - 2):
        i, y = valid[k]
        around = [valid[j][1] for j in range(max(0, k - 4), min(len(valid), k + 5))]
        if y == min(around):
            if not peaks or i - peaks[-1] > fps * min_gap_s:
                peaks.append(i)
    return peaks


class SwimAnalyzer:
    """Analisador multi-estilo a partir de vídeo com vista lateral fixa."""

    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    NOSE = 0

    def __init__(self, video_path, style="crawl", show=False, min_visib=0.5,
                 thresholds=None):
        if style not in STYLE_CONFIG:
            sys.exit(f"Estilo desconhecido: {style}. Opções: {list(STYLE_CONFIG)}")
        self.style = style
        self.cfg = _cfg_from_flat(style, thresholds)
        self.limiares_usados = dict(thresholds) if thresholds else {}
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            sys.exit(f"Erro: não foi possível abrir {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.show = show
        self.min_visib = min_visib
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True, enable_segmentation=False,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.frames = []
        self.keyframes = []   # (t, imagem BGR anotada) para o PDF
        self._next_kf_t = 1.0

    # ------------------------------------------------------------------ #
    def _landmark(self, lm, idx, w, h):
        p = lm[idx]
        return (p.x * w, p.y * h, getattr(p, "visibility", 1.0))

    def process(self):
        t = 0.0
        prev_mid_sho = None
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = {
                "t": t,
                "cotovelo": {"L": None, "R": None},
                "punho": {"L": None, "R": None},
                "ombro": {"L": None, "R": None},
                "joelho": {"L": None, "R": None},
                "tornozelo": {"L": None, "R": None},
                "quadril_pt": {"L": None, "R": None},
                "nose": None, "v_mid_sho": 0.0,
            }
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                g = lambda i: self._landmark(lm, i, w, h)
                joints = {
                    "L": (self.L_SHOULDER, self.L_ELBOW, self.L_WRIST,
                          self.L_HIP, self.L_KNEE, self.L_ANKLE),
                    "R": (self.R_SHOULDER, self.R_ELBOW, self.R_WRIST,
                          self.R_HIP, self.R_KNEE, self.R_ANKLE),
                }
                for side, (s_i, e_i, w_i, h_i, k_i, a_i) in joints.items():
                    sho, elb, wrs = g(s_i), g(e_i), g(w_i)
                    hip, knee, ank = g(h_i), g(k_i), g(a_i)
                    if min(sho[2], elb[2], wrs[2]) >= self.min_visib:
                        row["cotovelo"][side] = angle_2d(sho, elb, wrs)
                        row["punho"][side] = wrs
                    if sho[2] >= self.min_visib:
                        row["ombro"][side] = sho
                    if min(hip[2], knee[2], ank[2]) >= self.min_visib:
                        row["joelho"][side] = angle_2d(hip, knee, ank)
                        row["tornozelo"][side] = ank
                    if hip[2] >= self.min_visib:
                        row["quadril_pt"][side] = hip
                row["nose"] = g(self.NOSE)
                shos = [row["ombro"][s] for s in ("L", "R") if row["ombro"][s]]
                if shos:
                    mid = (np.mean([p[0] for p in shos]), np.mean([p[1] for p in shos]))
                    if prev_mid_sho is not None:
                        row["v_mid_sho"] = math.hypot(mid[0] - prev_mid_sho[0],
                                                      mid[1] - prev_mid_sho[1])
                    prev_mid_sho = mid
            idx = len(self.frames)
            if res.pose_landmarks and len(self.keyframes) < 4 and t >= self._next_kf_t:
                vis = frame.copy()
                self.mp_draw.draw_landmarks(
                    vis, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.keyframes.append((t, vis))
                self._next_kf_t += max(1.5, 6.0 / self.fps)
            self.frames.append(row)
            if self.show:
                self._draw(frame, res, row)
            t += 1.0 / self.fps
        self.cap.release()
        cv2.destroyAllWindows()
        return len(self.frames)

    def _draw(self, frame, res, row):
        if res.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        msgs = [(10, 30, f"estilo: {self.cfg['label']}")]
        for side, col in (("L", (0, 255, 0)), ("R", (0, 255, 255))):
            if row["cotovelo"][side]:
                msgs.append((10, 30 + 35 * (1 if side == "L" else 2),
                             f"cotovelo{side} {row['cotovelo'][side]:.0f}\u00b0"))
        for x, y, s in msgs:
            cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
        cv2.imshow("Analisador de Nado", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.show = False

    # ------------------------------------------------------------------ #
    def _arm_cycles(self):
        """Retorna {side: [índices de pico]} conforme o modo do estilo.
        alternate: picos de elevação por ombro (L e R separados).
        symmetric: série única (média dos dois ombros por frame)."""
        out = {}
        if self.cfg["cycle_mode"] == "alternate":
            for side in ("L", "R"):
                ys = [f["ombro"][side][1] if f["ombro"][side] else None
                      for f in self.frames]
                out[side] = _local_peaks(ys, self.fps, self.cfg["min_cycle_s"])
        else:
            ys = []
            for f in self.frames:
                shos = [f["ombro"][s][1] for s in ("L", "R") if f["ombro"][s]]
                ys.append(float(np.mean(shos)) if shos else None)
            pk = _local_peaks(ys, self.fps, self.cfg["min_cycle_s"])
            out = {"L": pk, "R": pk}
        return out

    # ------------------------------------------------------------------ #
    def metrics(self):
        m = {"estilo": self.cfg["label"]}
        n = len(self.frames)
        m["duracao_s"] = round(n / self.fps, 1) if self.fps else 0.0

        # escala corporal: distância média ombro-quadril (para normalizar)
        torso = [
            math.hypot(f["ombro"]["L"][0] - f["quadril_pt"]["L"][0],
                       f["ombro"]["L"][1] - f["quadril_pt"]["L"][1])
            for f in self.frames
            if f["ombro"]["L"] and f["quadril_pt"]["L"]
        ] or [
            math.hypot(f["ombro"]["R"][0] - f["quadril_pt"]["R"][0],
                       f["ombro"]["R"][1] - f["quadril_pt"]["R"][1])
            for f in self.frames
            if f["ombro"]["R"] and f["quadril_pt"]["R"]
        ]
        torso_len = float(np.median(torso)) if torso else None
        if torso_len:
            m["torso_px"] = round(torso_len, 1)

        # --- ciclos e cadência
        cycles = self._arm_cycles()
        for side, label in (("L", "esquerdo"), ("R", "direito")):
            pk = cycles[side]
            m[f"ciclos_{label}"] = len(pk)
            if len(pk) >= 2:
                dt = self.frames[pk[-1]]["t"] - self.frames[pk[0]]["t"]
                if dt > 0:
                    m[f"cadencia_{label}_cpm"] = round(60.0 * (len(pk) - 1) / dt, 1)
                    m[f"seg_por_ciclo_{label}"] = round(dt / (len(pk) - 1), 2)
            else:
                m[f"cadencia_{label}_cpm"] = None

        # --- ângulo de cotovelo (crawl: faixa 80–100° citada [VAN].
        #     Para borboleta não há faixa numérica citada — não aplicamos.)
        if self.style == "crawl":
            lo, hi, _ = self.cfg["elbow_band"]
            for side, label in (("L", "esquerdo"), ("R", "direito")):
                vals = [f["cotovelo"][side] for f in self.frames
                        if f["cotovelo"][side] is not None
                        and 30.0 < f["cotovelo"][side] < 175.0]
                if vals:
                    m[f"cotovelo_med_{label}"] = round(float(np.median(vals)), 1)
                    m[f"cotovelo_ok_{label}_pct"] = round(
                        100.0 * sum(lo <= v <= hi for v in vals) / len(vals), 1)

        # --- simetria de rotação (alternados) / deslocamento (simultâneos)
        dys = {}
        for side in ("L", "R"):
            pts = [f["ombro"][side][1] for f in self.frames if f["ombro"][side]]
            if len(pts) >= 4:
                dys[side] = float(np.ptp(pts))  # pico-a-pico vertical
        if dys:
            for side in ("L", "R"):
                if torso_len:
                    m[f"amplitude_ombro_{side.lower()}"] = round(
                        dys[side] / torso_len, 2)  # em "torso"
            if self.style in ("crawl", "costas") and len(dys) == 2:
                ratio = min(dys.values()) / max(max(dys.values()), 1e-6)
                m["simetria_roll_ratio"] = round(ratio, 2)

            # costas: estimativa de rotação a partir da oscilação do ombro
            if self.style == "costas" and torso_len:
                amp = float(np.mean(list(dys.values()))) / torso_len
                rot = math.degrees(math.asin(max(0.0, min(amp / 2.0, 1.0))))
                m["rotacao_estimada_deg"] = round(rot, 1)  # HEURÍSTICO

        # --- estabilidade da cabeça (costas)
        if self.style == "costas":
            noses = [f["nose"][1] for f in self.frames
                     if f["nose"] and f["nose"][2] >= self.min_visib]
            if noses and torso_len:
                m["cabeca_std_norm"] = round(
                    float(np.std(noses)) / torso_len * 100, 2)  # % do torso

        # --- tronco (não aplicável a costas: corpo supino)
        if self.style in ("crawl", "peito", "borboleta"):
            trunks = []
            for f in self.frames:
                for side in ("L", "R"):
                    sho, hip = f["ombro"][side], f["quadril_pt"][side]
                    if sho and hip:
                        trunks.append(angle_2d(
                            (sho[0], sho[1]), (hip[0], hip[1]),
                            (hip[0], hip[1] - 10)))
                        break
            trunks = [v for v in trunks if not math.isnan(v)]
            if trunks:
                m["tronco_med_deg"] = round(float(np.median(trunks)), 1)

        # --- borboleta: batidas por ciclo + ondulação do quadril
        if self.style == "borboleta":
            knees = [f["joelho"]["L"] if f["joelho"]["L"] is not None
                     else f["joelho"]["R"] for f in self.frames]
            knees = [v for v in knees if v is not None]
            if knees:
                m["flexao_joelho_med_deg"] = round(float(np.median(knees)), 1)
                pk_knee = _local_peaks(
                    [None if f["joelho"]["L"] is None and f["joelho"]["R"] is None
                     else (f["joelho"]["L"] or f["joelho"]["R"]) for f in self.frames],
                    self.fps, 0.25)
                n_arm_cycles = len(cycles["L"])
                if n_arm_cycles >= 1 and len(pk_knee) >= 1:
                    m["batidas_por_ciclo_est"] = round(
                        len(pk_knee) / max(n_arm_cycles, 1), 2)  # alvo ~2 [USM]
            hips = [f["quadril_pt"]["L"][1] if f["quadril_pt"]["L"]
                    else f["quadril_pt"]["R"][1] for f in self.frames
                    if f["quadril_pt"]["L"] or f["quadril_pt"]["R"]]
            if len(hips) >= 4 and torso_len:
                m["ondulacao_quadril_torso"] = round(
                    float(np.ptp(hips)) / torso_len, 2)  # HEURÍSTICO

        # --- peito: fração de deslize + flexão de joelho (chicote)
        if self.style == "peito":
            speeds = [f["v_mid_sho"] for f in self.frames]
            vmax = max(speeds) if speeds else 0.0
            cyc = cycles["L"]
            if vmax > 0 and len(cyc) >= 2:
                glide_fracs = []
                for a, b in zip(cyc[:-1], cyc[1:]):
                    seg = speeds[a:b]
                    if seg:
                        glide_fracs.append(
                            100.0 * sum(1 for v in seg if v < 0.35 * vmax) / len(seg))
                if glide_fracs:
                    m["deslize_med_pct"] = round(float(np.mean(glide_fracs)), 1)
            knees = [f["joelho"]["L"] if f["joelho"]["L"] is not None
                     else f["joelho"]["R"] for f in self.frames]
            knees = [v for v in knees if v is not None]
            if knees:
                m["flexao_joelho_max_deg"] = round(float(np.max(knees)), 1)
        return m

    # ------------------------------------------------------------------ #
    def feedback(self, m):
        # Sanitiza NaN: comparações/formatos com NaN não devem derrubar o relatório
        m = {k: (None if isinstance(v, float) and math.isnan(v) else v)
             for k, v in m.items()}
        cfg, fb = self.cfg, []
        S = {  # fontes por métrica
            "VAN": "[Vanderbilt YSJ 2025](https://www.vanderbilt.edu/youngscientistjournal/article/swimming-stroke-analysis-and-technique-improvement-via-pose-estimation/)",
            "TRI": "[Triathlete](https://www.triathlete.com/training/what-is-the-ideal-stroke-rate/)",
            "SWI": "[SwimSmart](https://www.swimsmarttoday.com/blogs/swimsmart-blog/how-head-position-controls-your-entire-body-alignment-in-swimming)",
            "BNB": "[SwimNow](https://www.swimnow.co.uk/learn-to-swim/the-breaststroke-blueprint/)",
            "GLI": "[SwimSwam](https://swimswam.com/physiological-proof-promoting-a-long-glide-in-breaststroke-decrea/)",
            "SLF": "[Swim Like A Fish](https://swimlikeafish.org/mastering-backstroke-rotation-drills/)",
            "USM": "[USMS](https://www.usms.org/fitness-and-training/guides/butterfly/kick)",
            "VAS": "[Vasa](https://vasatrainer.com/blog/butterfly-swimming-techniques-tips-drills/)",
            "HK": "[Human Kinetics](https://us.humankinetics.com/blogs/excerpt/correct-form-for-the-breaststroke)",
            "HEU": "heurístico (sem fonte externa)",
        }

        def src_ref(key):
            """Consulta blindada: chave ausente nunca derruba o relatório."""
            return S.get(key, "fonte não catalogada")

        def band_msg(name, v, lo, hi, src, abaixo, acima, ok):
            if v < lo:
                fb.append(f"[{name}] {v:.1f} < {lo:.1f}: {abaixo} ({src_ref(src)})")
            elif v > hi:
                fb.append(f"[{name}] {v:.1f} > {hi:.1f}: {acima} ({src_ref(src)})")
            else:
                fb.append(f"[{name}] {v:.1f} — {ok} ({src_ref(src)})")

        # ---- cotovelo (crawl / borboleta)
        if "elbow_band" in cfg:
            lo, hi, src = cfg["elbow_band"]
            for label in ("esquerdo", "direito"):
                v = m.get(f"cotovelo_med_{label}")
                if v is not None:
                    if v < lo:
                        fb.append(f"[Cotovelo {label}] {v:.0f}\u00b0 < {lo:.0f}\u00b0: "
                                  f"eleve o cotovelo na recuperação ({src_ref(src)})")
                    elif v > hi:
                        fb.append(f"[Cotovelo {label}] {v:.0f}\u00b0 > {hi:.0f}\u00b0: "
                                  f"abaixe levemente — recuperação relaxada e alta ({src_ref(src)})")
                    else:
                        fb.append(f"[Cotovelo {label}] {v:.0f}\u00b0 na faixa ótima "
                                  f"{lo:.0f}–{hi:.0f}\u00b0 ({src_ref(src)})")

        # ---- cadência (por estilo)
        if self.style == "crawl":
            lo, hi, src = cfg["sr_band"]
            for side, label in (("esquerdo", "esquerda"), ("direito", "direita")):
                v = m.get(f"cadencia_{side}_cpm")
                if v is not None:
                    band_msg(f"Cadência {label}", v, lo, hi, src,
                             "aumente a cadência para manter impulso contínuo",
                             "cadência muito alta encurta a braçada",
                             "dentro da faixa sugerida")
        elif self.style == "peito":
            lo, hi, src = cfg["sec_por_bracada"]
            for side, label in (("esquerdo", "esquerda"), ("direito", "direita")):
                v = m.get(f"seg_por_ciclo_{side}")
                if v is not None:
                    if v < lo:
                        fb.append(f"[Cadência {label}] {v:.2f} s/ciclo < {lo:.1f}: "
                                  f"acelere menos — deixe o deslize completar ({src_ref('BNB')})")
                    elif v > hi:
                        fb.append(f"[Cadência {label}] {v:.2f} s/ciclo > {hi:.1f}: "
                                  f"nado muito lento para 200 m de referência ({src_ref('BNB')})")
                    else:
                        fb.append(f"[Cadência {label}] {v:.2f} s/ciclo — perto da "
                                  f"referência de 200 m ({src_ref('BNB')})")
            g = m.get("deslize_med_pct")
            if g is not None:
                if g > cfg["glide_max_pct"]:
                    fb.append(f"[Deslize] {g:.0f}% do ciclo em deslize > {cfg['glide_max_pct']:.0f}%: "
                              f"nadadores rápidos limitam o deslize não propulsivo a ~22% ({src_ref('GLI')})")
                else:
                    fb.append(f"[Deslize] {g:.0f}% — dentro do padrão de elite ({src_ref('GLI')})")
            k = m.get("flexao_joelho_max_deg")
            if k is not None:
                fb.append(f"[Pernada] Flexão máx. do joelho {k:.0f}\u00b0 — alvo: recuperação "
                          f"com calcanhares aos glúteos e joelhos na largura do quadril, "
                          f"chicote no final ({src_ref('HK')}; sem faixa numérica citada)")

        elif self.style == "costas":
            v = m.get("rotacao_estimada_deg")
            if v is not None:
                lo, hi, src = cfg["rot_band"]
                band_msg("Rotação (estimada)", v, lo, hi, src,
                         "rotação fraca — gire mais o corpo para ativar as costas do braço",
                         "rotação exagerada — risco de nado 'serpenteante'",
                         "rotação adequada por lado")
            h = m.get("cabeca_std_norm")
            if h is not None:
                if h > cfg["head_std_max"]:
                    fb.append(f"[Cabeça] Oscilação {h:.1f}% do torso > {cfg['head_std_max']:.0f}%: "
                              "estabilize a cabeça (olhos para o teto, orelhas na água) "
                              "(HEURÍSTICO)")
                else:
                    fb.append(f"[Cabeça] Estável ({h:.1f}% do torso) — bom alinhamento "
                              "(HEURÍSTICO)")
            fb.append("[Entrada] Mire a entrada 'mindinho primeiro' atrás da linha do "
                      "ombro e saída pelo polegar — não mensurável em 2D lateral "
                      "([RocketSwim](https://rocketswim.com/backstroke-swimming-guide/))")

        elif self.style == "borboleta":
            k = m.get("batidas_por_ciclo_est")
            if k is not None:
                lo, src = cfg["kicks_por_ciclo"][0], cfg["kicks_por_ciclo"][1]
                tol = cfg.get("kicks_tol", 0.5)
                if abs(k - lo) > tol:
                    fb.append(f"[Batidas] ~{k:.1f} batidas/ciclo fora do alvo "
                              f"{lo:.0f}±{tol:.1f}: sincronize "
                              f"duas batidas por ciclo de braços ({src_ref(src)})")
                else:
                    fb.append(f"[Batidas] ~{k:.1f} batidas/ciclo — padrão 2:1 correto "
                              f"({lo:.0f}±{tol:.1f}) ({src_ref(src)})")
            u = m.get("ondulacao_quadril_torso")
            if u is not None:
                if u < cfg["undulation_min"]:
                    fb.append(f"[Ondulação] {u:.2f}× torso (pico-a-pico) < {cfg['undulation_min']:.2f}: "
                              "ondulação fraca — inicie a onda pela cabeça e propague até "
                              "os pés (HEURÍSTICO; conceito: "
                              "[USMS](https://www.usms.org/fitness-and-training/articles-and-videos/videos/what-your-hips-need-to-do-while-you-swim-butterfly))")
                else:
                    fb.append(f"[Ondulação] {u:.2f}× torso — amplitude presente "
                              "(HEURÍSTICO; conceito: [Strzała 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5765785/))")

        # ---- tronco (crawl, peito, borboleta)
        t = m.get("tronco_med_deg")
        if t is not None and cfg.get("trunk_max"):
            if t > cfg["trunk_max"]:
                fb.append(f"[Tronco] {t:.0f}\u00b0 > {cfg['trunk_max']:.0f}\u00b0: quadril 'caído' "
                          f"— cabeça neutra para subir o quadril ({src_ref('SWI')})")
            else:
                fb.append(f"[Tronco] {t:.0f}\u00b0 — boa linha horizontal ({src_ref('SWI')})")

        # ---- simetria (alternados)
        r = m.get("simetria_roll_ratio")
        if r is not None:
            sim_min = cfg.get("symmetry_min", 0.80)
            if r < sim_min:
                fb.append(f"[Simetria] {r:.2f} < {sim_min:.2f}: um lado gira mais que o outro "
                          "(HEURÍSTICO)")
            else:
                fb.append(f"[Simetria] {r:.2f} — rotação equilibrada (HEURÍSTICO)")

        if not fb:
            fb.append("Poucos frames válidos: grave com vista lateral, mais luz e "
                      "câmera fixa para uma análise mais rica.")
        return fb

    # ------------------------------------------------------------------ #
    def set_style(self, style):
        if style not in STYLE_CONFIG:
            style = "crawl"
        self.style = style
        # reconstrói a cfg mantendo os limiares ajustados pelo usuário
        self.cfg = _cfg_from_flat(style, self.limiares_usados or None)

    def timeseries(self):
        ts = {"t": [], "cotovelo_L": [], "cotovelo_R": [],
              "punho_y_L": [], "punho_y_R": [], "vel_sho": []}
        for f in self.frames:
            ts["t"].append(f["t"])
            ts["cotovelo_L"].append(f["cotovelo"]["L"])
            ts["cotovelo_R"].append(f["cotovelo"]["R"])
            ts["punho_y_L"].append(f["punho"]["L"][1] if f["punho"]["L"] else None)
            ts["punho_y_R"].append(f["punho"]["R"][1] if f["punho"]["R"] else None)
            ts["vel_sho"].append(f["v_mid_sho"])
        return ts

    def _torso_len(self):
        vals = []
        for f in self.frames:
            for s in ("L", "R"):
                sho, hip = f["ombro"][s], f["quadril_pt"][s]
                if sho and hip:
                    vals.append(math.hypot(sho[0] - hip[0], sho[1] - hip[1]))
                    break
        return float(np.median(vals)) if vals else None

    # ------------------------------------------------------------------ #
    def _tempos_fora(self, sinal, fora):
        """Tempos (s) dos frames em que o sinal viola a condição `fora(v)`.
        Usado para localizar o momento aproximado do erro no vídeo."""
        ts = []
        for f in self.frames:
            v = sinal(f)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if fora(v):
                ts.append(f["t"])
        return ts

    def _fontes(self):
        return {
            "VAN": "[Vanderbilt YSJ 2025](https://www.vanderbilt.edu/youngscientistjournal/article/swimming-stroke-analysis-and-technique-improvement-via-pose-estimation/)",
            "TRI": "[Triathlete](https://www.triathlete.com/training/what-is-the-ideal-stroke-rate/)",
            "SWI": "[SwimSmart](https://www.swimsmarttoday.com/blogs/swimsmart-blog/how-head-position-controls-your-entire-body-alignment-in-swimming)",
            "BNB": "[SwimNow](https://www.swimnow.co.uk/learn-to-swim/the-breaststroke-blueprint/)",
            "GLI": "[SwimSwam](https://swimswam.com/physiological-proof-promoting-a-long-glide-in-breaststroke-decrea/)",
            "SLF": "[Swim Like A Fish](https://swimlikeafish.org/mastering-backstroke-rotation-drills/)",
            "USM": "[USMS](https://www.usms.org/fitness-and-training/guides/butterfly/kick)",
            "HK": "[Human Kinetics](https://us.humankinetics.com/blogs/excerpt/correct-form-for-the-breaststroke)",
            "HEU": "HEURÍSTICO (sem fonte externa citada)",
        }

    def erros(self, m):
        """Erros de movimento estruturados por estilo. Cada item traz o que
        foi medido, o limiar de referência, o momento aproximado do vídeo em
        que ocorre (quando localizável frame a frame) e um drill de correção.
        Limiares marcados HEURÍSTICO não têm fonte externa citada."""
        cfg = self.cfg
        S = self._fontes()
        E = []

        def add(titulo, medido, limiar, tempos, drill, fonte):
            E.append({
                "titulo": titulo,
                "medido": medido,
                "limiar": limiar,
                "quando_s": round(float(np.median(tempos)), 1) if tempos else None,
                "drill": drill,
                "fonte": fonte,
            })

        def ang_tronco(f):
            for s in ("L", "R"):
                sho, hip = f["ombro"][s], f["quadril_pt"][s]
                if sho and hip:
                    return angle_2d((sho[0], sho[1]), (hip[0], hip[1]),
                                    (hip[0], hip[1] - 10))
            return None

        if self.style == "crawl":
            lo, hi, _ = cfg["elbow_band"]
            for side, label in (("L", "esquerdo"), ("R", "direito")):
                v = m.get(f"cotovelo_med_{label}")
                if v is None:
                    continue
                if v < lo:
                    add(f"Cotovelo {label} baixo demais na recuperação",
                        f"mediana {v:.0f}°", f"{lo:.0f}–{hi:.0f}° (ótimo ~90°)",
                        self._tempos_fora(lambda f: f["cotovelo"][side],
                                          lambda x: x < lo),
                        "Drill 'dedos arrastando a água': passe as pontas dos "
                        "dedos raspando a superfície durante toda a recuperação "
                        "para fixar o cotovelo alto", S["VAN"])
                elif v > hi:
                    add(f"Cotovelo {label} rígido/estendido na recuperação",
                        f"mediana {v:.0f}°", f"{lo:.0f}–{hi:.0f}° (ótimo ~90°)",
                        self._tempos_fora(lambda f: f["cotovelo"][side],
                                          lambda x: x > hi),
                        "Drill 'braço de pano': recupere o braço totalmente "
                        "relaxado, deixando o antebraço pendurar do cotovelo",
                        S["VAN"])
            lo_s, hi_s, _ = cfg["sr_band"]
            for side, label in (("esquerdo", "esquerda"), ("direito", "direita")):
                v = m.get(f"cadencia_{side}_cpm")
                if v is None:
                    continue
                if v < lo_s:
                    add(f"Cadência {label} baixa",
                        f"{v:.0f} cpm", f"{lo_s:.0f}–{hi_s:.0f} cpm", None,
                        "Progressão de cadência: 4×25 m subindo a cadência "
                        "gradualmente sem encurtar a braçada", S["TRI"])
                elif v > hi_s:
                    add(f"Cadência {label} alta (braçada curta)",
                        f"{v:.0f} cpm", f"{lo_s:.0f}–{hi_s:.0f} cpm", None,
                        "Golf de nado (SWOLF): conte as braçadas por comprimento "
                        "e busque reduzir o total mantendo o tempo", S["TRI"])
            t = m.get("tronco_med_deg")
            if t is not None and t > cfg["trunk_max"]:
                add("Tronco inclinado (quadril 'caído')",
                    f"mediana {t:.0f}°", f"<= {cfg['trunk_max']:.0f}°",
                    self._tempos_fora(ang_tronco, lambda x: x > cfg["trunk_max"]),
                    "Drill 'kick de costas com braços à frente': cabeça neutra "
                    "olhando o fundo e quadril na superfície", S["SWI"])
            r = m.get("simetria_roll_ratio")
            if r is not None and r < cfg.get("symmetry_min", 0.80):
                sim_min = cfg.get("symmetry_min", 0.80)
                add("Rotação assimétrica entre os lados",
                    f"razão {r:.2f}", f">= {sim_min:.2f}", None,
                    "Drill 6/1/6: 6 batidas de um lado, 1 braçada, 6 batidas do "
                    "outro — para equilibrar a rotação", S["HEU"])

        elif self.style == "peito":
            lo, hi, _ = cfg["sec_por_bracada"]
            for side, label in (("esquerdo", "esquerda"), ("direito", "direita")):
                v = m.get(f"seg_por_ciclo_{side}")
                if v is None:
                    continue
                if v < lo:
                    add(f"Ciclo {label} apressado (deslize curto)",
                        f"{v:.2f} s/ciclo", f"{lo:.1f}–{hi:.1f} s/ciclo", None,
                        "Drill '2 kicks 1 braçada': complete a extensão em "
                        "streamline antes da próxima braçada", S["BNB"])
                elif v > hi:
                    add(f"Ciclo {label} lento demais",
                        f"{v:.2f} s/ciclo", f"{lo:.1f}–{hi:.1f} s/ciclo", None,
                        "Reduza o deslize passivo: inicie a tração assim que a "
                        "velocidade começar a cair", S["BNB"])
            g = m.get("deslize_med_pct")
            if g is not None and g > cfg["glide_max_pct"]:
                add("Deslize passivo longo demais no ciclo",
                    f"{g:.0f}% do ciclo", f"<= {cfg['glide_max_pct']:.0f}%",
                    None,
                    "Encurte a fase morta: puxe mais cedo e mantenha velocidade "
                    "contínua (elite limita o deslize a ~22% do ciclo)", S["GLI"])
            k = m.get("flexao_joelho_max_deg")
            k_lim = cfg.get("knee_flex_max", 165.0)
            if k is not None and k > k_lim:
                add("Pernada sem flexão suficiente (chicote fraco)",
                    f"flexão máx. {k:.0f}°",
                    f"recuperação com calcanhares próximos dos glúteos "
                    f"(flexão < {k_lim:.0f}°)", None,
                    "Drill de pernada segurando a borda: calcanhares sobem até "
                    "os glúteos, joelhos na largura do quadril, pés virados "
                    "para fora no chicote", S["HK"])
            t = m.get("tronco_med_deg")
            if t is not None and t > cfg["trunk_max"]:
                add("Tronco inclinado (quadril 'caído')",
                    f"mediana {t:.0f}°", f"<= {cfg['trunk_max']:.0f}°",
                    self._tempos_fora(ang_tronco, lambda x: x > cfg["trunk_max"]),
                    "Ao respirar, o queixo desliza à frente sem levantar o "
                    "tronco — cabeça neutra entre as braçadas", S["SWI"])

        elif self.style == "costas":
            v = m.get("rotacao_estimada_deg")
            if v is not None:
                lo, hi, _ = cfg["rot_band"]
                if v < lo:
                    add("Rotação do corpo fraca",
                        f"≈ {v:.0f}° por lado", f"{lo:.0f}–{hi:.0f}° por lado",
                        None,
                        "Drill '6 kicks por lado': nade de costas rotacionado e "
                        "segure a posição antes de girar para o outro lado",
                        S["SLF"])
                elif v > hi:
                    add("Rotação do corpo exagerada",
                        f"≈ {v:.0f}° por lado", f"{lo:.0f}–{hi:.0f}° por lado",
                        None,
                        "Gire como um tronco rígido, não role: os quadris "
                        "acompanham os ombros sem passar da vertical", S["SLF"])
            h = m.get("cabeca_std_norm")
            if h is not None and h > cfg["head_std_max"]:
                add("Cabeça oscilando (alinhamento instável)",
                    f"oscilação {h:.1f}% do torso",
                    f"<= {cfg['head_std_max']:.0f}%", None,
                    "Drill do copo d'água: equilibre um copo na testa nadendo "
                    "costas, olhos para o teto", S["HEU"])
            r = m.get("simetria_roll_ratio")
            if r is not None and r < cfg.get("symmetry_min", 0.80):
                sim_min = cfg.get("symmetry_min", 0.80)
                add("Rotação assimétrica entre os lados",
                    f"razão {r:.2f}", f">= {sim_min:.2f}", None,
                    "Drill 6/1/6 de costas: segure a rotação de cada lado por 6 "
                    "batidas para equalizar", S["HEU"])

        elif self.style == "borboleta":
            k = m.get("batidas_por_ciclo_est")
            alvo = cfg["kicks_por_ciclo"][0]
            tol = cfg.get("kicks_tol", 0.5)
            if k is not None and abs(k - alvo) > tol:
                add("Batidas fora do alvo por ciclo",
                    f"≈ {k:.1f} batidas/ciclo",
                    f"{alvo:.0f} ± {tol:.1f} batidas por ciclo de braços",
                    None,
                    "Drill 'golfinho na vertical': uma batida na entrada das "
                    "mãos e outra na saída da água, sincronizadas com a "
                    "respiração", S["USM"])
            u = m.get("ondulacao_quadril_torso")
            if u is not None and u < cfg["undulation_min"]:
                add("Ondulação do corpo fraca",
                    f"{u:.2f}× torso (pico-a-pico)",
                    f">= {cfg['undulation_min']:.2f}× torso", None,
                    "Drill de golfinho submerso: propague a onda da cabeça até "
                    "os pés com o corpo inteiro, sem braços", S["HEU"])

        return E

    def report(self):
        m = self.metrics()
        m = {k: (None if isinstance(v, float) and math.isnan(v) else v)
             for k, v in m.items()}
        try:
            fb = self.feedback(m)
        except Exception:
            # um problema isolado no feedback não pode derrubar o relatório todo
            fb = ["(feedback indisponível para este vídeo — consulte as métricas "
                  "acima; se persistir, relate o erro com o traceback)"]
        try:
            erros = self.erros(m)
        except Exception:
            erros = []
        # tabela efetiva de limiares usados (padrão ou ajustados pelo usuário)
        limiares = {}
        for k, info in DEFAULT_THRESHOLDS[self.style].items():
            v = self.limiares_usados.get(k, info["valor"]) if self.limiares_usados \
                else info["valor"]
            limiares[k] = {"valor": v, "unidade": info["unidade"],
                           "fonte": info["fonte"], "descricao": info["descricao"]}
        return {"estilo": self.style, "metricas": m, "limiares": limiares,
                "erros": erros, "feedback": fb}


def gerar_pdf(rep, keyframes, ts, out_path):
    """Gera relatório PDF (reportlab + matplotlib) com acentos corretos:
    capas/meta, métricas, feedback, gráficos das séries temporais e
    frames-chave anotados."""
    import glob
    import re
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    # fonte com suporte completo a acentos, embutida no PDF
    font_name = "Helvetica"  # fallback (WinAnsi cobre português)
    for cand in glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True):
        pdfmetrics.registerFont(TTFont("DejaVuSans", cand))
        font_name = "DejaVuSans"
        break

    strip_md = lambda s: re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    h1 = ParagraphStyle("h1", fontName=font_name, fontSize=18, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName=font_name, fontSize=13,
                        spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", fontName=font_name, fontSize=9.5, leading=13)

    story = [Paragraph("Relatório de Análise de Nado", h1)]
    meta = [
        ["Vídeo", os.path.basename(rep.get("video", ""))],
        ["Estilo analisado", STYLE_CONFIG[rep["estilo"]]["label"]],
        ["Frames", str(rep["frames_analisados"])],
        ["Duração", f"{rep['metricas'].get('duracao_s', 0)} s"],
    ]
    t = Table(meta, colWidths=[4 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    story += [t, Paragraph("Métricas", h2)]

    rows = [[k, str(v)] for k, v in rep["metricas"].items()]
    mt = Table(rows, colWidths=[7 * cm, 9 * cm])
    mt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ]))
    story.append(mt)

    story.append(Paragraph("Erros de movimento identificados", h2))
    for e in rep.get("erros", []):
        quando = (f" (ocorre em ~{e['quando_s']:.1f} s)"
                  if e.get("quando_s") is not None else "")
        story.append(Paragraph(
            f"• {e['titulo']}{quando} — medido: {e['medido']}; "
            f"limiar: {e['limiar']}. Correção: {e['drill']} "
            f"[{strip_md(e['fonte'])}]", body))

    story.append(Paragraph("Feedback resumido", h2))
    for line in rep["feedback"]:
        story.append(Paragraph("• " + strip_md(line), body))

    # --- gráficos das séries temporais
    def _series(key):
        return np.array([np.nan if v is None else v for v in ts[key]], dtype=float)

    tt = np.array(ts["t"], dtype=float)
    plots = [
        ("Ângulo do cotovelo (graus)",
         [("esquerdo", _series("cotovelo_L")), ("direito", _series("cotovelo_R"))]),
        ("Trajetória vertical dos punhos (px) — fase dos braços",
         [("esquerdo", _series("punho_y_L")), ("direito", _series("punho_y_R"))]),
        ("Velocidade do ombro (px/frame) — deslize no peito",
         [("média", _series("vel_sho"))]),
    ]
    plot_files = []
    for title, series in plots:
        fig, ax = plt.subplots(figsize=(7, 2.4))
        for label, ys in series:
            ax.plot(tt, ys, label=label, lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tempo (s)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fd, fp = tempfile.mkstemp(suffix=".png")
        fig.savefig(fp, dpi=140)
        plt.close(fig)
        plot_files.append(fp)
    story.append(Paragraph("Séries temporais", h2))
    for fp in plot_files:
        story += [Image(fp, width=16 * cm, height=5.5 * cm), Spacer(1, 4)]

    # --- frames-chave anotados
    kf_files = []
    if keyframes:
        story.append(PageBreak())
        story.append(Paragraph("Frames-chave anotados", h2))
        for t_k, img in keyframes:
            fd, fp = tempfile.mkstemp(suffix=".png")
            cv2.imwrite(fp, img)
            kf_files.append(fp)
            story += [Image(fp, width=12 * cm, height=9 * cm),
                      Paragraph(f"t = {t_k:.1f} s", body), Spacer(1, 6)]

    doc = SimpleDocTemplate(out_path, pagesize=A4)
    doc.build(story)
    for fp in plot_files + kf_files:
        os.unlink(fp)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Analisador de nado (crawl/peito/costas/borboleta, vista lateral)")
    ap.add_argument("video", help="arquivo de vídeo (.mp4/.mov/.avi)")
    ap.add_argument("--estilo", default="crawl", choices=list(STYLE_CONFIG),
                    help="estilo do nado a ser analisado")
    ap.add_argument("--exibir", action="store_true", help="mostrar janela com overlay")
    ap.add_argument("--saida", help="salvar relatório em JSON")
    ap.add_argument("--pdf", help="gerar relatório em PDF com gráficos e frames-chave")
    args = ap.parse_args()

    an = SwimAnalyzer(args.video, style=args.estilo, show=args.exibir)
    n = an.process()

    rep = an.report()
    rep["video"] = args.video
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

    print(f"\n=== RELATÓRIO DE NADO ({rep['estilo']}) — {args.video} ===")
    print(f"Estilo: {an.cfg['label']} | Frames: {n} "
          f"({rep['metricas'].get('duracao_s', 0)} s @ {an.fps:.0f} fps)")
    print("\nErros de movimento identificados:")
    for e in rep.get("erros", []):
        q = f" (≈ {e['quando_s']:.1f} s)" if e.get("quando_s") is not None else ""
        print(f"  - {e['titulo']}{q} | medido {e['medido']} | limiar {e['limiar']}")
        print(f"      correção: {e['drill']}")
    print("Métricas:")
    for k, v in rep["metricas"].items():
        print(f"  {k:30s} {v}")
    print("\nFeedback:")
    for line in rep["feedback"]:
        print(f"  • {line}")

    if args.saida:
        with open(args.saida, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"\nRelatório salvo em {args.saida}")

    if args.pdf:
        gerar_pdf(rep, an.keyframes, an.timeseries(), args.pdf)
        print(f"Relatório PDF salvo em {args.pdf}")


if __name__ == "__main__":
    main()
