import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Video } from "@remotion/media";
import datos from "./datos.json";
import { C, Encabezado, Fondo, MONO, SANS, Tarjeta, Visor, fmt, prog } from "./tema";

const CONTEO: number[] = datos.conteo;
const DENSO: number[] = datos.conteoDenso;
const ENTRENO = datos.entreno;
const MEJOR = ENTRENO.find((r) => r.e === datos.mejor)!;
const MAXIMO = Math.max(...CONTEO);
const MEDIA = CONTEO.reduce((a, b) => a + b, 0) / CONTEO.length;

/* ---------------------------------------------------------------- 1. Portada */
export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const palabras = ["Detección", "de", "aglomeraciones"];
  const sub = prog(frame, 34, 30);
  const rec = Math.floor(frame / 15) % 2 === 0;
  return (
    <AbsoluteFill style={{ backgroundColor: C.fondo }}>
      <AbsoluteFill style={{ scale: interpolate(frame, [0, 150], [1.12, 1.02]), filter: "blur(10px) saturate(0.7)", opacity: 0.45 }}>
        <Video src={staticFile("camara.mp4")} muted objectFit="cover" style={{ width: "100%", height: "100%" }} />
      </AbsoluteFill>
      <AbsoluteFill style={{ background: "radial-gradient(ellipse at center, rgba(6,10,19,0.35), rgba(6,10,19,0.95) 75%)" }} />
      <div style={{ position: "absolute", inset: 70 }}>
        <Visor a={prog(frame, 0, 30)} largo={80} grosor={3} color="rgba(255,255,255,0.55)" />
      </div>
      <div style={{ position: "absolute", left: 110, top: 100, fontFamily: MONO, fontSize: 24, color: C.texto, display: "flex", gap: 14, alignItems: "center", opacity: prog(frame, 6, 20) }}>
        <div style={{ width: 16, height: 16, borderRadius: 8, background: rec ? C.rosa : "transparent", boxShadow: rec ? `0 0 18px ${C.rosa}` : "none" }} />
        REC · CAM 15
      </div>
      <div style={{ position: "absolute", right: 110, top: 100, fontFamily: MONO, fontSize: 24, color: C.tenue, opacity: prog(frame, 6, 20) }}>
        YOLO11s · 896 px
      </div>
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 30 }}>
          {palabras.map((p, i) => {
            const a = prog(frame, 8 + i * 7, 30);
            return (
              <div key={p} style={{ overflow: "hidden", paddingBottom: 12 }}>
                <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 112, letterSpacing: -3.5, color: C.texto, translate: `0 ${(1 - a) * 160}px` }}>
                  {p}
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ fontFamily: SANS, fontSize: 40, color: C.tenue, marginTop: 10, opacity: sub, translate: `0 ${(1 - sub) * 20}px` }}>
          Conteo de personas y mapas de calor en tiempo real con <span style={{ color: C.ambar, fontWeight: 700 }}>YOLO11</span>
        </div>
        <div style={{ display: "flex", gap: 16, marginTop: 46 }}>
          {["Visión por computador", "Cámaras existentes", "OpenCV", "Gradio"].map((t, i) => {
            const a = prog(frame, 52 + i * 5, 24);
            return (
              <div key={t} style={{ fontFamily: MONO, fontSize: 22, color: C.texto, padding: "10px 22px", borderRadius: 999, border: `1px solid ${C.borde}`, background: "rgba(255,255,255,0.05)", opacity: a, scale: interpolate(a, [0, 1], [0.9, 1]) }}>
                {t}
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 2. Problema */
export const Problema: React.FC = () => {
  const frame = useCurrentFrame();
  const v = prog(frame, 4, 30);
  const preguntas = [
    ["¿Cuántas personas hay ahora?", C.ambar],
    ["¿Cuál es el promedio y el pico?", C.cian],
    ["¿En qué zonas se concentran?", C.rosa],
  ];
  return (
    <AbsoluteFill>
      <Fondo />
      <Encabezado etiqueta="El punto de partida" titulo="Una cámara de calle común" />
      <div style={{ position: "absolute", left: 120, top: 270, width: 880, height: 720, opacity: v, translate: `${(1 - v) * -40}px 0` }}>
        <div style={{ position: "absolute", inset: 0, borderRadius: 18, overflow: "hidden", boxShadow: "0 40px 100px rgba(0,0,0,0.6)" }}>
          <Video src={staticFile("camara.mp4")} muted trimBefore={150} objectFit="cover" style={{ width: "100%", height: "100%" }} />
        </div>
        <Visor a={prog(frame, 14, 26)} color={C.ambar} />
        <div style={{ position: "absolute", left: 0, bottom: -52, fontFamily: MONO, fontSize: 22, color: C.tenue }}>352 × 288 px · 15 FPS · personas de pocos píxeles</div>
      </div>
      <div style={{ position: "absolute", left: 1090, top: 340, width: 710, display: "flex", flexDirection: "column", gap: 34 }}>
        {preguntas.map(([q, col], i) => {
          const a = prog(frame, 30 + i * 18, 30);
          return (
            <Tarjeta key={q} color={col} style={{ padding: "34px 40px", opacity: a, translate: `${(1 - a) * 60}px 0`, display: "flex", alignItems: "center", gap: 26 }}>
              <div style={{ fontFamily: MONO, fontSize: 26, color: col, fontWeight: 700 }}>0{i + 1}</div>
              <div style={{ fontFamily: SANS, fontSize: 38, fontWeight: 600, color: C.texto }}>{q}</div>
            </Tarjeta>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 3. Pipeline */
const PASOS = [
  { t: "Video o cámara", s: "archivo o webcam", c: C.cian, icono: "M4 7h11v10H4z M15 10l5-3v10l-5-3" },
  { t: "YOLO11s", s: "896 px · conf 0.15", c: C.ambar, icono: "M12 3l8 4.5v9L12 21l-8-4.5v-9z M12 12l8-4.5 M12 12v9 M12 12L4 7.5" },
  { t: "Cajas", s: "una por cabeza", c: C.rosa, icono: "M4 4h7v7H4z M13 13h7v7h-7z M13 4h7v5" },
  { t: "Conteo", s: "actual · media · máx", c: C.verde, icono: "M4 20V10 M10 20V4 M16 20v-7 M22 20H2" },
  { t: "Mapa de calor", s: "centros + gaussiano", c: C.violeta, icono: "M12 3c3 4 6 6.5 6 10.5a6 6 0 01-12 0C6 9.5 9 7 12 3z" },
];

export const Pipeline: React.FC = () => {
  const frame = useCurrentFrame();
  const x0 = 150;
  const paso = 340;
  const y = 560;
  const linea = prog(frame, 20, 110);
  const pulso = ((frame - 40) % 70) / 70;
  return (
    <AbsoluteFill>
      <Fondo acento={C.violeta} />
      <Encabezado etiqueta="Cómo funciona" titulo="Cada cuadro recorre el mismo camino" color={C.violeta} />
      <svg width={1920} height={1080} style={{ position: "absolute" }}>
        <defs>
          <linearGradient id="g" gradientUnits="userSpaceOnUse" x1={x0 + 130} x2={x0 + 130 + paso * 4} y1={0} y2={0}>
            {PASOS.map((p, i) => (
              <stop key={p.t} offset={i / (PASOS.length - 1)} stopColor={p.c} />
            ))}
          </linearGradient>
        </defs>
        <line x1={x0 + 130} y1={y - 55} x2={x0 + 130 + (paso * 4) * linea} y2={y - 55} stroke="url(#g)" strokeWidth={4} strokeLinecap="round" opacity={0.8} />
        {frame > 40 &&
          [0, 0.33, 0.66].map((d) => {
            const t = (pulso + d) % 1;
            return <circle key={d} cx={x0 + 130 + paso * 4 * t} cy={y - 55} r={9} fill="#fff" opacity={0.9 * Math.sin(Math.PI * t)} style={{ filter: "drop-shadow(0 0 10px #fff)" }} />;
          })}
      </svg>
      {PASOS.map((p, i) => {
        const a = prog(frame, 14 + i * 16, 30);
        return (
          <div key={p.t} style={{ position: "absolute", left: x0 + i * paso - 20, top: y - 130, width: 300, display: "flex", flexDirection: "column", alignItems: "center", opacity: a, translate: `0 ${(1 - a) * 40}px` }}>
            <div style={{ width: 150, height: 150, borderRadius: 40, background: `linear-gradient(160deg, ${p.c}33, ${p.c}0A), ${C.fondo}`, border: `2px solid ${p.c}`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 0 60px ${p.c}40` }}>
              <svg width={72} height={72} viewBox="0 0 24 24" fill="none" stroke={p.c} strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round">
                <path d={p.icono} />
              </svg>
            </div>
            <div style={{ height: 110 }} />
            <div style={{ fontFamily: SANS, fontWeight: 700, fontSize: 34, color: C.texto, whiteSpace: "nowrap" }}>{p.t}</div>
            <div style={{ fontFamily: MONO, fontSize: 21, color: C.tenue, marginTop: 8 }}>{p.s}</div>
          </div>
        );
      })}
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 120, textAlign: "center", fontFamily: SANS, fontSize: 32, color: C.tenue, opacity: prog(frame, 120, 30) }}>
        Mapa de calor dinámico: el acumulador se multiplica por <span style={{ color: C.violeta, fontFamily: MONO }}>0.9</span> en cada cuadro, así lo viejo se desvanece
      </div>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 4. Entrenamiento */
const W_G = 760;
const H_G = 470;
const camino = (k: "map50" | "p" | "r", hasta: number) =>
  ENTRENO.slice(0, hasta + 1).map((r, i) => `${i ? "L" : "M"}${(i / (ENTRENO.length - 1)) * W_G},${H_G - r[k] * H_G}`).join(" ");

export const Entrenamiento: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const t = prog(frame, 25, durationInFrames - 70);
  const idx = Math.max(0, Math.min(ENTRENO.length - 1, Math.round(t * (ENTRENO.length - 1))));
  const ep = ENTRENO[idx];
  const series = [
    { k: "p" as const, n: "Precisión", c: C.cian },
    { k: "map50" as const, n: "mAP50", c: C.ambar },
    { k: "r" as const, n: "Recall", c: C.rosa },
  ];
  const lote = Math.min(2, Math.floor(frame / (durationInFrames / 3)));
  const tl = (frame % (durationInFrames / 3)) / (durationInFrames / 3);
  const v = prog(frame, 4, 30);
  return (
    <AbsoluteFill>
      <Fondo acento={C.cian} />
      <Encabezado etiqueta={`Ajuste fino · ${datos.horas} h de entrenamiento`} titulo="YOLO11s aprende a ver cabezas" color={C.cian} />
      <div style={{ position: "absolute", left: 120, top: 280, width: 640, height: 640, borderRadius: 20, overflow: "hidden", opacity: v, boxShadow: "0 40px 100px rgba(0,0,0,0.6)" }}>
        {[0, 1, 2].map((i) => (
          <Img key={i} src={staticFile(`lote${i}.jpg`)} style={{ position: "absolute", width: "100%", height: "100%", objectFit: "cover", opacity: i === lote ? 1 : 0, scale: i === lote ? 1 + 0.1 * tl : 1 }} />
        ))}
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, padding: "60px 26px 22px", background: "linear-gradient(transparent, rgba(0,0,0,0.85))", fontFamily: MONO, fontSize: 20, color: C.texto }}>
          lotes de entrenamiento anotados
        </div>
      </div>
      <Tarjeta style={{ position: "absolute", left: 850, top: 280, width: 950, height: 640, opacity: v }}>
        <div style={{ position: "absolute", left: 50, top: 34, fontFamily: MONO, fontSize: 24, color: C.tenue }}>
          época <span style={{ color: C.texto, fontWeight: 700, fontSize: 34 }}>{String(ep.e).padStart(3, "0")}</span> / {ENTRENO.length}
        </div>
        <svg width={W_G + 40} height={H_G + 20} style={{ position: "absolute", left: 120, top: 100, overflow: "visible" }}>
          {[0, 0.2, 0.4, 0.6, 0.8, 1].map((g) => (
            <g key={g}>
              <line x1={0} x2={W_G} y1={H_G - g * H_G} y2={H_G - g * H_G} stroke="rgba(255,255,255,0.07)" />
              <text x={-18} y={H_G - g * H_G + 7} fill={C.tenue} fontFamily={MONO} fontSize={18} textAnchor="end">
                {g.toFixed(1)}
              </text>
            </g>
          ))}
          {series.map((s) => (
            <g key={s.k}>
              <path d={camino(s.k, idx)} fill="none" stroke={s.c} strokeWidth={4} strokeLinejoin="round" style={{ filter: `drop-shadow(0 0 8px ${s.c}88)` }} />
              <circle cx={(idx / (ENTRENO.length - 1)) * W_G} cy={H_G - ep[s.k] * H_G} r={8} fill={s.c} opacity={t > 0 ? 1 : 0} />
            </g>
          ))}
        </svg>
        <div style={{ position: "absolute", left: 120, bottom: 30, display: "flex", gap: 60 }}>
          {series.map((s) => (
            <div key={s.k} style={{ fontFamily: MONO, fontSize: 24, color: C.tenue }}>
              <span style={{ display: "inline-block", width: 14, height: 14, borderRadius: 7, background: s.c, marginRight: 12 }} />
              {s.n} <span style={{ color: C.texto, fontWeight: 700 }}>{ep[s.k].toFixed(3)}</span>
            </div>
          ))}
        </div>
      </Tarjeta>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 5. Detección en vivo */
const VEL = 4;

const Sparkline: React.FC<{ valores: number[]; hasta: number; w: number; h: number; color: string; max: number }> = ({ valores, hasta, w, h, color, max }) => {
  const paso = Math.max(1, Math.floor(valores.length / 400));
  const pts: string[] = [];
  for (let i = 0; i <= hasta; i += paso) pts.push(`${(i / (valores.length - 1)) * w},${h - (valores[i] / max) * h}`);
  const cx = (hasta / (valores.length - 1)) * w;
  return (
    <svg width={w} height={h} style={{ overflow: "visible" }}>
      <defs>
        <linearGradient id="area" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity={0.35} />
          <stop offset="1" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      {pts.length > 1 && <polygon points={`0,${h} ${pts.join(" ")} ${cx},${h}`} fill="url(#area)" />}
      {pts.length > 1 && <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={3} strokeLinejoin="round" />}
      <line x1={cx} x2={cx} y1={0} y2={h} stroke="rgba(255,255,255,0.4)" strokeDasharray="4 6" />
      <circle cx={cx} cy={h - (valores[hasta] / max) * h} r={7} fill={color} style={{ filter: `drop-shadow(0 0 8px ${color})` }} />
    </svg>
  );
};

const estado = (frame: number, fps: number, desde = 0) => {
  const i = Math.min(CONTEO.length - 1, Math.floor(((frame / fps) * VEL + desde) * datos.fpsFuente));
  let suma = 0;
  let max = 0;
  for (let k = 0; k <= i; k++) {
    suma += CONTEO[k];
    max = Math.max(max, CONTEO[k]);
  }
  return { i, ahora: CONTEO[i], media: suma / (i + 1), max };
};

export const EnVivo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { i, ahora, media, max } = estado(frame, fps);
  const v = prog(frame, 0, 26);
  const barrido = interpolate(frame, [10, 50], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill>
      <Fondo />
      <Encabezado etiqueta={`Salida real · ×${VEL} de velocidad`} titulo="Detección en vivo" />
      <div style={{ position: "absolute", left: 120, top: 260, width: 930, height: 761, opacity: v }}>
        <div style={{ position: "absolute", inset: 0, borderRadius: 18, overflow: "hidden", boxShadow: "0 40px 100px rgba(0,0,0,0.6)" }}>
          <Video src={staticFile("deteccion.mp4")} muted playbackRate={VEL} objectFit="cover" style={{ width: "100%", height: "100%" }} />
          {barrido < 1 && (
            <div style={{ position: "absolute", left: 0, right: 0, top: `${barrido * 100}%`, height: 6, background: C.ambar, boxShadow: `0 0 40px 12px ${C.ambar}88` }} />
          )}
        </div>
        <Visor a={prog(frame, 6, 26)} />
      </div>
      <div style={{ position: "absolute", left: 1120, top: 260, width: 680, display: "flex", flexDirection: "column", gap: 26 }}>
        <Tarjeta color={C.ambar} style={{ padding: "30px 40px", opacity: prog(frame, 10, 26) }}>
          <div style={{ fontFamily: MONO, fontSize: 22, color: C.tenue, letterSpacing: 2 }}>PERSONAS AHORA</div>
          <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 150, lineHeight: 1.05, color: C.ambar, fontVariantNumeric: "tabular-nums", textShadow: `0 0 50px ${C.ambar}55` }}>{ahora}</div>
        </Tarjeta>
        <div style={{ display: "flex", gap: 26 }}>
          {[
            ["PROMEDIO", fmt(media), C.verde],
            ["PICO", String(max), C.rosa],
          ].map(([e, val, col], k) => (
            <Tarjeta key={e} style={{ flex: 1, padding: "26px 32px", opacity: prog(frame, 18 + k * 6, 26) }}>
              <div style={{ fontFamily: MONO, fontSize: 20, color: C.tenue, letterSpacing: 2 }}>{e}</div>
              <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 64, color: col, fontVariantNumeric: "tabular-nums" }}>{val}</div>
            </Tarjeta>
          ))}
        </div>
        <Tarjeta style={{ padding: "26px 32px", opacity: prog(frame, 30, 26) }}>
          <div style={{ fontFamily: MONO, fontSize: 20, color: C.tenue, letterSpacing: 2, marginBottom: 18 }}>PERSONAS POR CUADRO</div>
          <Sparkline valores={CONTEO} hasta={i} w={616} h={150} color={C.ambar} max={MAXIMO} />
        </Tarjeta>
      </div>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 6. Mapa de calor dinámico */
export const CalorDinamico: React.FC = () => {
  const frame = useCurrentFrame();
  const v = prog(frame, 0, 26);
  return (
    <AbsoluteFill>
      <Fondo acento={C.rosa} />
      <Encabezado etiqueta="Mapa de calor dinámico" titulo="El calor sigue a la gente" color={C.rosa} />
      <div style={{ position: "absolute", left: 120, top: 260, width: 930, height: 761, opacity: v, scale: interpolate(v, [0, 1], [0.97, 1]) }}>
        <div style={{ position: "absolute", inset: 0, borderRadius: 18, overflow: "hidden", boxShadow: "0 40px 100px rgba(0,0,0,0.6)" }}>
          <Video src={staticFile("calor.mp4")} muted playbackRate={VEL} trimBefore={Math.round(60 * 30)} objectFit="cover" style={{ width: "100%", height: "100%" }} />
        </div>
        <Visor a={prog(frame, 6, 26)} color={C.rosa} />
      </div>
      <div style={{ position: "absolute", left: 1120, top: 300, width: 680, display: "flex", flexDirection: "column", gap: 30 }}>
        {[
          ["1", "Cada cabeza detectada suma su centro en un acumulador", C.rosa],
          ["2", "Un desenfoque gaussiano (kernel 31) lo vuelve una superficie suave", C.ambar],
          ["3", "En cada cuadro el acumulador se multiplica por 0.9", C.violeta],
        ].map(([n, t, col], k) => {
          const a = prog(frame, 20 + k * 16, 28);
          return (
            <div key={n} style={{ display: "flex", gap: 24, alignItems: "flex-start", opacity: a, translate: `${(1 - a) * 40}px 0` }}>
              <div style={{ minWidth: 56, height: 56, borderRadius: 28, border: `2px solid ${col}`, color: col, fontFamily: MONO, fontSize: 26, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center" }}>{n}</div>
              <div style={{ fontFamily: SANS, fontSize: 34, lineHeight: 1.3, color: C.texto, fontWeight: 500 }}>{t}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 7. Mapa acumulado */
export const Acumulado: React.FC = () => {
  const frame = useCurrentFrame();
  const v = prog(frame, 0, 26);
  const calor = prog(frame, 25, 60);
  return (
    <AbsoluteFill>
      <Fondo acento={C.ambar} />
      <Encabezado etiqueta={`Suma de ${CONTEO.length.toLocaleString("es-CO")} mapas`} titulo="Dónde se concentra la gente" />
      <div style={{ position: "absolute", left: 120, top: 260, width: 930, height: 761, opacity: v }}>
        <div style={{ position: "absolute", inset: 0, borderRadius: 18, overflow: "hidden", boxShadow: "0 40px 100px rgba(0,0,0,0.6)" }}>
          <Img src={staticFile("escena.png")} style={{ position: "absolute", width: "100%", height: "100%", filter: `saturate(${1 - 0.5 * calor}) brightness(${1 - 0.25 * calor})` }} />
          <Img src={staticFile("acumulado.png")} style={{ position: "absolute", width: "100%", height: "100%", opacity: calor, filter: "blur(2px)", mixBlendMode: "screen" }} />
          <Img src={staticFile("acumulado.png")} style={{ position: "absolute", width: "100%", height: "100%", opacity: calor * 0.8 }} />
        </div>
        <Visor a={prog(frame, 6, 26)} />
      </div>
      <div style={{ position: "absolute", left: 1120, top: 300, width: 680, display: "flex", flexDirection: "column", gap: 26 }}>
        {[
          ["PERSONAS POR CUADRO", `${fmt(MEDIA)}`, "de media", C.verde],
          ["PICO", `${MAXIMO}`, "personas", C.rosa],
          ["VELOCIDAD", `${fmt(datos.fpsMedio)}`, "FPS de media", C.cian],
        ].map(([e, val, u, col], k) => {
          const a = prog(frame, 40 + k * 12, 28);
          return (
            <Tarjeta key={e} style={{ padding: "28px 36px", opacity: a, translate: `${(1 - a) * 40}px 0` }}>
              <div style={{ fontFamily: MONO, fontSize: 20, color: C.tenue, letterSpacing: 2 }}>{e}</div>
              <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 70, color: col }}>
                {val} <span style={{ fontSize: 30, color: C.tenue, fontWeight: 500 }}>{u}</span>
              </div>
            </Tarjeta>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 8. Escena densa */
export const Densa: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const vel = 8.26 / (durationInFrames / fps);
  const v = prog(frame, 0, 26);
  return (
    <AbsoluteFill>
      <Fondo acento={C.cian} />
      <Encabezado etiqueta="Interfaz web con Gradio" titulo="Otra cámara, más gente" color={C.cian} />
      <div style={{ position: "absolute", left: 120, top: 260, width: 930, height: 761, opacity: v }}>
        <div style={{ position: "absolute", inset: 0, borderRadius: 18, overflow: "hidden", boxShadow: "0 40px 100px rgba(0,0,0,0.6)" }}>
          <Video src={staticFile("densa.mp4")} muted playbackRate={vel} objectFit="cover" style={{ width: "100%", height: "100%" }} />
        </div>
        <Visor a={prog(frame, 6, 26)} color={C.cian} />
      </div>
      <div style={{ position: "absolute", left: 1120, top: 300, width: 680, display: "flex", flexDirection: "column", gap: 26 }}>
        <Tarjeta color={C.cian} style={{ padding: "30px 40px", opacity: prog(frame, 10, 26) }}>
          <div style={{ fontFamily: MONO, fontSize: 22, color: C.tenue, letterSpacing: 2 }}>PICO EN UN CUADRO</div>
          <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 150, lineHeight: 1.05, color: C.cian }}>{Math.max(...DENSO)}</div>
          <div style={{ fontFamily: MONO, fontSize: 22, color: C.tenue }}>promedio de la escena: <span style={{ color: C.texto }}>{fmt(DENSO.reduce((x, y) => x + y, 0) / DENSO.length)}</span></div>
        </Tarjeta>
        {["Subir un video o grabar con la cámara", "Elegir salidas: video, mapa, PNG, CSV", "Ajustar kernel y decaimiento"].map((t, k) => {
          const a = prog(frame, 26 + k * 10, 26);
          return (
            <div key={t} style={{ display: "flex", gap: 18, alignItems: "center", fontFamily: SANS, fontSize: 30, color: C.texto, opacity: a, translate: `${(1 - a) * 30}px 0` }}>
              <div style={{ width: 12, height: 12, borderRadius: 6, background: C.cian, boxShadow: `0 0 12px ${C.cian}` }} />
              {t}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 9. Resultados */
const Anillo: React.FC<{ valor: number; color: string; a: number }> = ({ valor, color, a }) => {
  const r = 96;
  const l = 2 * Math.PI * r;
  return (
    <svg width={240} height={240} style={{ rotate: "-90deg" }}>
      <circle cx={120} cy={120} r={r} stroke="rgba(255,255,255,0.08)" strokeWidth={16} fill="none" />
      <circle cx={120} cy={120} r={r} stroke={color} strokeWidth={16} fill="none" strokeLinecap="round" strokeDasharray={l} strokeDashoffset={l * (1 - valor * a)} style={{ filter: `drop-shadow(0 0 10px ${color}99)` }} />
    </svg>
  );
};

export const Resultados: React.FC = () => {
  const frame = useCurrentFrame();
  const m = [
    ["mAP50", MEJOR.map50, C.ambar],
    ["Precisión", MEJOR.p, C.cian],
    ["Recall", MEJOR.r, C.rosa],
    ["mAP50-95", MEJOR.map, C.violeta],
  ] as const;
  return (
    <AbsoluteFill>
      <Fondo />
      <Encabezado etiqueta={`Validación · mejor época ${datos.mejor} de ${ENTRENO.length}`} titulo="Resultados del modelo" />
      <div style={{ position: "absolute", left: 120, right: 120, top: 330, display: "flex", justifyContent: "space-between" }}>
        {m.map(([n, val, col], k) => {
          const a = prog(frame, 14 + k * 10, 50);
          return (
            <div key={n} style={{ width: 360, display: "flex", flexDirection: "column", alignItems: "center", opacity: prog(frame, 10 + k * 10, 20) }}>
              <div style={{ position: "relative", width: 240, height: 240 }}>
                <Anillo valor={val} color={col} a={a} />
                <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: SANS, fontWeight: 800, fontSize: 58, color: C.texto, fontVariantNumeric: "tabular-nums" }}>
                  {(val * a).toFixed(3)}
                </div>
              </div>
              <div style={{ fontFamily: SANS, fontSize: 36, fontWeight: 600, color: col, marginTop: 24 }}>{n}</div>
            </div>
          );
        })}
      </div>
      <div style={{ position: "absolute", left: 200, right: 200, bottom: 150, textAlign: "center", fontFamily: SANS, fontSize: 32, lineHeight: 1.45, color: C.tenue, opacity: prog(frame, 70, 30) }}>
        Cabezas pequeñas en cámaras de baja resolución: la precisión es buena y el recall queda cerca de 0,40,
        por eso en operación el umbral de confianza baja a <span style={{ color: C.ambar, fontFamily: MONO }}>0.15</span>.
      </div>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- 10. Cierre */
export const Cierre: React.FC = () => {
  const frame = useCurrentFrame();
  const a = prog(frame, 0, 30);
  const b = prog(frame, 14, 30);
  return (
    <AbsoluteFill>
      <Fondo />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <div style={{ overflow: "hidden", paddingBottom: 10 }}>
          <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 120, letterSpacing: -4, color: C.texto, translate: `0 ${(1 - a) * 140}px` }}>Crowd Detection</div>
        </div>
        <div style={{ fontFamily: MONO, fontSize: 30, color: C.ambar, opacity: b, marginTop: 8 }}>YOLO11s · OpenCV · Gradio · Python</div>
        <div style={{ fontFamily: MONO, fontSize: 26, color: C.tenue, opacity: prog(frame, 28, 30), marginTop: 40, padding: "14px 30px", border: `1px solid ${C.borde}`, borderRadius: 999 }}>
          github.com/CARLYGP/Crowd_Detection
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
