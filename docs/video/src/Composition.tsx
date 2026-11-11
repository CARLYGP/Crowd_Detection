import React from "react";
import { AbsoluteFill, Composition, Img, Still, staticFile } from "remotion";
import { TransitionSeries, springTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { wipe } from "@remotion/transitions/wipe";
import datos from "./datos.json";
import { Acumulado, CalorDinamico, Cierre, Densa, EnVivo, Entrenamiento, Intro, Pipeline, Problema, Resultados } from "./escenas";
import { C, Fondo, MONO, SANS, Visor } from "./tema";

const T = 20; // frames de cada transición
const ESCENAS: [string, React.FC, number][] = [
  ["Portada", Intro, 150],
  ["Problema", Problema, 180],
  ["Pipeline", Pipeline, 210],
  ["Entrenamiento", Entrenamiento, 240],
  ["En vivo", EnVivo, 360],
  ["Calor dinámico", CalorDinamico, 240],
  ["Acumulado", Acumulado, 210],
  ["Escena densa", Densa, 180],
  ["Resultados", Resultados, 210],
  ["Cierre", Cierre, 150],
];
const DURACION = ESCENAS.reduce((s, e) => s + e[2], 0) - T * (ESCENAS.length - 1);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const transiciones: any[] = [fade(), slide({ direction: "from-right" }), wipe({ direction: "from-left" })];

export const CrowdVideo: React.FC = () => (
  <TransitionSeries>
    {ESCENAS.flatMap(([nombre, Escena, dur], i) => {
      const nodos = [
        <TransitionSeries.Sequence key={nombre} name={nombre} durationInFrames={dur}>
          <Escena />
        </TransitionSeries.Sequence>,
      ];
      if (i < ESCENAS.length - 1)
        nodos.push(
          <TransitionSeries.Transition
            key={nombre + "-t"}
            presentation={(i === 0 || i === ESCENAS.length - 2 ? fade() : transiciones[i % 3]) as ReturnType<typeof fade>}
            timing={springTiming({ config: { damping: 200 }, durationInFrames: T })}
          />,
        );
      return nodos;
    })}
  </TransitionSeries>
);

const MAXIMO = Math.max(...datos.conteo);
const MAP50 = datos.entreno.find((r) => r.e === datos.mejor)!.map50;

/** Escena con detección + mapa acumulado, base de las dos portadas. */
const Escena: React.FC<{ style: React.CSSProperties }> = ({ style }) => (
  <div style={{ position: "absolute", overflow: "hidden", borderRadius: 24, boxShadow: "0 40px 120px rgba(0,0,0,0.7)", ...style }}>
    <Img src={staticFile("escena.png")} style={{ position: "absolute", width: "100%", height: "100%", objectFit: "cover", filter: "saturate(0.6) brightness(0.8)" }} />
    <Img src={staticFile("acumulado.png")} style={{ position: "absolute", width: "100%", height: "100%", objectFit: "cover", filter: "blur(3px)", mixBlendMode: "screen" }} />
    <Img src={staticFile("acumulado.png")} style={{ position: "absolute", width: "100%", height: "100%", objectFit: "cover", opacity: 0.85 }} />
  </div>
);

const Chip: React.FC<{ e: string; v: string; c: string; escala?: number }> = ({ e, v, c, escala = 1 }) => (
  <div style={{ padding: `${16 * escala}px ${24 * escala}px`, borderRadius: 16 * escala, background: "rgba(8,12,22,0.72)", border: `1px solid ${c}66`, backdropFilter: "blur(8px)" }}>
    <div style={{ fontFamily: MONO, fontSize: 16 * escala, color: C.tenue, letterSpacing: 2 }}>{e}</div>
    <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 34 * escala, color: c }}>{v}</div>
  </div>
);

/** Portada del README (banner ancho). */
export const PortadaBanner: React.FC = () => (
  <AbsoluteFill>
    <Fondo />
    <Escena style={{ right: 70, top: 60, width: 640, height: 520 }} />
    <div style={{ position: "absolute", right: 70, top: 60, width: 640, height: 520 }}>
      <Visor largo={50} grosor={4} />
    </div>
    <div style={{ position: "absolute", left: 90, top: 150 }}>
      <div style={{ fontFamily: MONO, fontSize: 22, letterSpacing: 4, color: C.ambar }}>VISIÓN POR COMPUTADOR · YOLO11</div>
      <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 92, lineHeight: 1.02, letterSpacing: -3, color: C.texto, marginTop: 18 }}>
        Detección de
        <br />
        aglomeraciones
      </div>
      <div style={{ fontFamily: SANS, fontSize: 28, color: C.tenue, marginTop: 22 }}>Conteo de personas y mapas de calor en tiempo real</div>
      <div style={{ display: "flex", gap: 16, marginTop: 40 }}>
        <Chip e="mAP50" v={MAP50.toFixed(2)} c={C.ambar} />
        <Chip e="FPS MEDIO" v={String(datos.fpsMedio)} c={C.cian} />
        <Chip e="PICO" v={`${MAXIMO} personas`} c={C.rosa} />
      </div>
    </div>
  </AbsoluteFill>
);

/** Tarjeta del perfil de GitHub (4:3). */
export const PortadaTarjeta: React.FC = () => (
  <AbsoluteFill>
    <Fondo />
    <Img src={staticFile("destacado.png")} style={{ position: "absolute", width: 1200, height: 900, objectFit: "cover", objectPosition: "center 40%" }} />
    <AbsoluteFill style={{ background: "linear-gradient(180deg, transparent 0%, transparent 60%, rgba(6,10,19,0.9) 100%)" }} />
    <div style={{ position: "absolute", inset: 46 }}>
      <Visor largo={70} grosor={5} />
    </div>
    <div style={{ position: "absolute", left: 70, right: 70, bottom: 70, display: "flex", gap: 20 }}>
      <Chip e="PERSONAS / CUADRO" v={(datos.conteo.reduce((a, b) => a + b, 0) / datos.conteo.length).toFixed(1).replace(".", ",")} c={C.verde} escala={1.25} />
      <Chip e="PICO" v={String(MAXIMO)} c={C.rosa} escala={1.25} />
      <Chip e="FPS" v={String(datos.fpsMedio).replace(".", ",")} c={C.cian} escala={1.25} />
    </div>
  </AbsoluteFill>
);

export const Composiciones: React.FC = () => (
  <>
    <Composition id="CrowdDetection" component={CrowdVideo} durationInFrames={DURACION} fps={30} width={1920} height={1080} />
    <Still id="PortadaBanner" component={PortadaBanner} width={1600} height={640} />
    <Still id="PortadaTarjeta" component={PortadaTarjeta} width={1200} height={900} />
  </>
);
