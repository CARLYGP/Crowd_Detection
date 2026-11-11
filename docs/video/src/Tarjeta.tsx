import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Video } from "@remotion/media";
import datos from "./datos.json";
import { C, MONO, SANS } from "./tema";

const DESDE = 120; // segundo del video de salida donde empieza el clip (zona con más gente)
const VEL = 3;

/** Tarjeta animada del perfil: dibujo lineal de la calle + mapa de calor real + un solo dato. */
export const TarjetaAnimada: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const i = Math.min(datos.conteo.length - 1, Math.floor((DESDE + (frame / fps) * VEL) * datos.fpsFuente));
  const borde = interpolate(frame, [0, 10, durationInFrames - 10, durationInFrames - 1], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const barrido = (frame % 75) / 75;
  // la escena es 352x288; se recorta a 4:3 centrada
  const alto = height;
  const ancho = (alto * 352) / 288;
  const capa: React.CSSProperties = { position: "absolute", left: (width - ancho) / 2, top: 0, width: ancho, height: alto };
  return (
    <AbsoluteFill style={{ backgroundColor: C.fondo, overflow: "hidden" }}>
      <AbsoluteFill style={{ backgroundImage: "radial-gradient(rgba(255,255,255,0.06) 1px, transparent 1px)", backgroundSize: "22px 22px" }} />
      <Img src={staticFile("lineas.png")} style={{ ...capa, opacity: 0.85 }} />
      <div style={{ ...capa, mixBlendMode: "screen", opacity: borde }}>
        <Video src={staticFile("calor_estilo.mp4")} muted trimBefore={DESDE * fps} playbackRate={VEL} style={{ width: "100%", height: "100%" }} />
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: `${barrido * 100}%`, height: 2, background: "rgba(125,211,252,0.5)", boxShadow: "0 0 24px 6px rgba(125,211,252,0.25)" }} />
      <AbsoluteFill style={{ boxShadow: "inset 0 0 140px rgba(0,0,0,0.85)" }} />
      <AbsoluteFill style={{ background: "linear-gradient(180deg, transparent 55%, rgba(6,10,19,0.92) 100%)" }} />
      <div style={{ position: "absolute", left: 40, bottom: 34, display: "flex", alignItems: "baseline", gap: 14 }}>
        <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 104, lineHeight: 1, color: C.ambar, fontVariantNumeric: "tabular-nums", textShadow: `0 0 40px ${C.ambar}66`, minWidth: 120 }}>
          {datos.conteo[i]}
        </div>
        <div style={{ fontFamily: MONO, fontSize: 22, color: C.texto, letterSpacing: 1 }}>personas en escena</div>
      </div>
    </AbsoluteFill>
  );
};
