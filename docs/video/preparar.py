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


def lineas(destino):
    """Dibujo lineal de la calle (bordes Sobel del cuadro de la cámara), sin los textos de la cámara."""
    from PIL import ImageDraw, ImageFilter, ImageOps
    esc = g.cuadro_en(g.RAIZ / "data" / "input" / "input2.mp4", 75).resize((704, 576), Image.LANCZOS)
    gris = np.asarray(ImageOps.grayscale(esc).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32)
    gx, gy = np.zeros_like(gris), np.zeros_like(gris)
    gx[:, 1:-1] = gris[:, 2:] - gris[:, :-2]
    gy[1:-1, :] = gris[2:, :] - gris[:-2, :]
    mag = np.hypot(gx, gy)
    lo, hi = np.percentile(mag, 70), np.percentile(mag, 98)
    a = np.clip((mag - lo) / (hi - lo), 0, 1) ** 0.8
    m = Image.new("L", (704, 576), 255)
    d = ImageDraw.Draw(m)
    d.rectangle((0, 0, 470, 62), fill=0)          # fecha y hora de la cámara
    d.rectangle((480, 490, 704, 576), fill=0)     # "Camera 15"
    a = a * np.asarray(m.filter(ImageFilter.GaussianBlur(4))) / 255
    alfa = Image.fromarray((a * 255).astype(np.uint8)).resize((1056, 864), Image.LANCZOS)
    Image.merge("RGBA", [Image.new("L", alfa.size, v) for v in (125, 211, 252)] + [alfa]).save(destino)


def calor_estilo(destino):
    """Mapa de calor dinámico re-coloreado con la rampa del video, sobre negro (se mezcla con 'screen')."""
    w, h = 352 * 3, 288 * 3
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "15.34",
                          "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", str(destino)], stdin=subprocess.PIPE)
    rampa = g.RAMPA
    for fc in g.cuadros(g.RAIZ / "output" / "heatmap_salida_detectada.mp4"):
        a = np.asarray(fc).astype(np.float32)
        v = np.clip(np.maximum(a[..., 0], a[..., 1]) / 255 + a[..., 0] / 510 - 0.12, 0, 1.2) / 1.2
        rgb = np.zeros(v.shape + (3,), np.float32)
        for (t0, c0), (t1, c1) in zip(rampa, rampa[1:]):
            m = (v >= t0) & (v <= t1)
            f = ((v - t0) / (t1 - t0))[m][:, None]
            rgb[m] = np.array(c0) + (np.array(c1) - np.array(c0)) * f
        rgb *= np.clip(v * 2.5, 0, 1)[..., None]
        p.stdin.write(Image.fromarray(rgb.astype(np.uint8)).resize((w, h), Image.LANCZOS).tobytes())
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
    lineas(PUB / "lineas.png")
    calor_estilo(PUB / "calor_estilo.mp4")
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
