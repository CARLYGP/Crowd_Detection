"""Prepara public/ para el video de Remotion a partir de las salidas reales del repositorio.

Uso (desde la raíz del repo):  python docs/video/preparar.py
Después:                       cd docs/video && npm install && npx remotion render CrowdDetection ../crowd_detection.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "animacion"))
import generar as g  # noqa: E402  (reutiliza la lectura de datos y el mapa de calor acumulado)

PUB = AQUI / "public"
PUB.mkdir(exist_ok=True)
FF = g.FF


def h264(origen, destino, filtro="scale=iw*3:ih*3:flags=lanczos", antes=()):
    subprocess.run([FF, "-y", "-loglevel", "error", *antes, "-i", str(origen), "-vf", filtro, "-c:v", "libx264",
                    "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(destino)],
                   check=True)


def superpuesto(destino):
    """Detección + mapa de calor dinámico (sin teñir el fondo), cuadro a cuadro."""
    w, h = 352 * 3, 288 * 3
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "15.34",
                          "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", str(destino)], stdin=subprocess.PIPE)
    for fd, fc in zip(g.cuadros(g.RAIZ / "output" / "salida_detectada.mp4"), g.cuadros(g.RAIZ / "output" / "heatmap_salida_detectada.mp4")):
        p.stdin.write(g.superponer(fd, fc).resize((w, h), Image.LANCZOS).tobytes())
    p.stdin.close()
    p.wait()


def main():
    out = g.RAIZ / "output"
    h264(out / "salida_detectada.mp4", PUB / "deteccion.mp4")
    superpuesto(PUB / "calor.mp4")
    h264(out / "gradio" / "1763611667" / "gradio_1763611667.mp4", PUB / "densa.mp4",
         "fps=30,scale=iw*3:ih*3:flags=lanczos")
    h264(g.RAIZ / "data" / "input" / "input2.mp4", PUB / "camara.mp4", antes=("-ss", "60", "-t", "20"))

    # cuadro con el pico de personas (28) para la tarjeta del perfil
    pico = max(range(len(g.CONTEO)), key=lambda i: g.CONTEO[i])
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{pico / 15.34:.2f}", "-i", str(PUB / "calor.mp4"), "-frames:v", "1",
                    str(PUB / "destacado.png")], check=True)
    escena = g.cuadro_en(g.RAIZ / "data" / "input" / "input2.mp4", 75)
    escena.resize((352 * 3, 288 * 3), Image.LANCZOS).save(PUB / "escena.png")
    g.acumular_calor().resize((352 * 3, 288 * 3), Image.LANCZOS).save(PUB / "acumulado.png")

    for i in range(3):
        img = Image.open(g.RAIZ / "results" / f"train_batch{i}.jpg").convert("RGB")
        m = np.asarray(img.convert("L")) < 240
        cols, filas = np.where(m.mean(axis=0) > 0.25)[0], np.where(m.mean(axis=1) > 0.25)[0]
        img.crop((cols[0], filas[0], cols[-1] + 1, filas[-1] + 1)).save(PUB / f"lote{i}.jpg", quality=92)

    datos = {
        "conteo": g.CONTEO,
        "fpsFuente": 15.34,
        "fpsMedio": round(g.FPS_MEDIO, 1),
        "conteoDenso": g.CONTEO_DENSO,
        "horas": round(g.HORAS, 1),
        "entreno": [{"e": int(r["epoch"]), "map50": float(r["metrics/mAP50(B)"]), "p": float(r["metrics/precision(B)"]),
                     "r": float(r["metrics/recall(B)"]), "map": float(r["metrics/mAP50-95(B)"])} for r in g.ENTRENO],
        "mejor": int(g.MEJOR["epoch"]),
    }
    (AQUI / "src" / "datos.json").write_text(json.dumps(datos), encoding="utf-8")
    print("listo:", ", ".join(sorted(f.name for f in PUB.iterdir())))


if __name__ == "__main__":
    main()
