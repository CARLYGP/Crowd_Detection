"""Genera el video, el GIF y las imágenes del README a partir de las salidas REALES del proyecto.

Usa lo que el pipeline ya produjo y está en el repositorio (no vuelve a correr YOLO):
  output/salida_detectada.mp4 + output/metrics.csv   (los 2.125 cuadros coinciden uno a uno)
  output/heatmap_salida_detectada.mp4 / .png          mapa de calor dinámico y acumulado
  output/gradio/1763611667/                           corrida de la interfaz Gradio en una escena más densa
  results/                                            entrenamiento de YOLO11s (100 épocas)

Uso (desde la raíz del repo):  python docs/animacion/generar.py
Requiere: numpy, pillow, imageio-ffmpeg
"""
import csv
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RAIZ = Path(__file__).resolve().parents[2]
DOCS = RAIZ / "docs"
FF = imageio_ffmpeg.get_ffmpeg_exe()

W, H, S, FPS = 1280, 720, 2, 30
FONDO = (10, 14, 24)
PANEL = (19, 25, 40)
LINEA = (44, 56, 82)
TEXTO = (236, 241, 250)
TENUE = (130, 146, 172)
AMBAR = (245, 158, 11)
AZUL = (14, 165, 233)
VERDE = (34, 197, 94)
ROSA = (236, 72, 153)
MORADO = (168, 85, 247)
FT = "C:/Windows/Fonts/"


def fuente(n, t):
    return ImageFont.truetype(FT + n, int(t * S))


F_TIT, F_SUB, F_PEQ = fuente("segoeuib.ttf", 30), fuente("segoeui.ttf", 18), fuente("segoeui.ttf", 14)
F_NUM, F_GRANDE, F_MEGA = fuente("segoeuib.ttf", 34), fuente("segoeuib.ttf", 50), fuente("segoeuib.ttf", 64)
F_CHIP = fuente("segoeuib.ttf", 16)


def suave(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def mezcla(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def P(x, y):
    return (int(x * S), int(y * S))


def lienzo(color=FONDO):
    im = Image.new("RGB", (W * S, H * S), color)
    return im, ImageDraw.Draw(im, "RGBA")


def tx(d, xy, t, f, c, anchor="la"):
    d.text(P(*xy), t, font=f, fill=c, anchor=anchor)


def caja(d, x0, y0, x1, y1, r=14, relleno=PANEL, borde=None, ancho=1):
    d.rounded_rectangle(P(x0, y0) + P(x1, y1), int(r * S), fill=relleno, outline=borde, width=int(ancho * S))


def encabezado(d, tit, sub, a=1.0):
    tx(d, (56, 30), tit, F_TIT, mezcla(FONDO, TEXTO, a))
    tx(d, (56, 72), sub, F_SUB, mezcla(FONDO, TENUE, a))


def pegar(im, img, x, y, w, h=None, alpha=1.0, radio=12):
    """Pega una imagen PIL escalada en coordenadas lógicas, con esquinas redondeadas."""
    h = h or int(w * img.height / img.width)
    pz = img.convert("RGB").resize((int(w * S), int(h * S)), Image.LANCZOS)
    m = Image.new("L", pz.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0) + pz.size, int(radio * S), fill=int(255 * alpha))
    im.paste(pz, P(x, y), m)
    return h


class Video:
    def __init__(self, ruta):
        self.n, self.marcas = 0, {}
        self.p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(ruta)], stdin=subprocess.PIPE)

    def cuadro(self, im, veces=1, guardar=None):
        peq = im.resize((W, H), Image.LANCZOS)
        if guardar:
            peq.save(DOCS / guardar)
        b = peq.tobytes()
        for _ in range(veces):
            self.p.stdin.write(b)
            self.n += 1

    def marca(self, k):
        self.marcas[k] = self.n / FPS

    def cerrar(self):
        self.p.stdin.close()
        self.p.wait()


def cuadros(ruta, w=352, h=288):
    """Itera los cuadros de un video como imágenes PIL (decodificados por ffmpeg)."""
    p = subprocess.Popen([FF, "-loglevel", "error", "-i", str(ruta), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    tam = w * h * 3
    while True:
        b = p.stdout.read(tam)
        if len(b) < tam:
            break
        yield Image.frombytes("RGB", (w, h), b)
    p.wait()


def cuadro_en(ruta, seg):
    out = subprocess.run([FF, "-loglevel", "error", "-ss", str(seg), "-i", str(ruta), "-frames:v", "1", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    from io import BytesIO
    return Image.open(BytesIO(out)).convert("RGB")


def leer_csv(ruta):
    return list(csv.DictReader(open(ruta, encoding="utf-8")))


# --- Datos reales ------------------------------------------------------------------------------------
ENTRENO = leer_csv(RAIZ / "results" / "results (1).csv")
MEJOR = max(ENTRENO, key=lambda r: float(r["metrics/mAP50(B)"]))
METRICAS = leer_csv(RAIZ / "output" / "metrics.csv")
CONTEO = [int(r["count"]) for r in METRICAS]
FPS_MEDIO = sum(float(r["fps"]) for r in METRICAS) / len(METRICAS)
DENSA = RAIZ / "output" / "gradio" / "1763611667"
CONTEO_DENSO = [int(r["count"]) for r in leer_csv(DENSA / "gradio_1763611667.csv")]
HORAS = float(ENTRENO[-1]["time"]) / 3600


def calor_a_rgba(img):
    """Mapa JET de OpenCV -> capa RGBA: el fondo azul oscuro se vuelve transparente."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    intensidad = np.clip((a[..., 0] + a[..., 1] * 0.8) / 255.0, 0, 1)
    alfa = (np.clip(intensidad * 1.6, 0, 1) * 235).astype(np.uint8)
    rgba = np.dstack([a.astype(np.uint8), alfa])
    return Image.fromarray(rgba, "RGBA")


def superponer(escena, calor, fuerza=1.0):
    """Escena original con el mapa de calor encima, sin teñir de azul el fondo."""
    capa = calor_a_rgba(calor)
    if fuerza < 1:
        capa.putalpha(capa.getchannel("A").point(lambda p: int(p * fuerza)))
    return Image.alpha_composite(escena.convert("RGBA"), capa).convert("RGB")


RAMPA = [(0.0, (20, 10, 60)), (0.3, (120, 28, 120)), (0.55, (220, 60, 70)), (0.8, (250, 160, 30)), (1.0, (255, 245, 170))]


def acumular_calor():
    """Suma los 2.125 mapas de calor dinámicos del video de salida y lo colorea con una rampa propia."""
    total = None
    for fc in cuadros(RAIZ / "output" / "heatmap_salida_detectada.mp4"):
        a = np.asarray(fc).astype(np.float32)
        v = np.maximum(a[..., 0], a[..., 1]) / 255 + a[..., 0] / 510  # JET -> intensidad aproximada
        v = np.clip(v - 0.08, 0, None)
        total = v if total is None else total + v
    total = np.sqrt(total / np.percentile(total, 99.7))
    total = np.clip(total, 0, 1)
    rgb = np.zeros(total.shape + (3,), np.float32)
    for (t0, c0), (t1, c1) in zip(RAMPA, RAMPA[1:]):
        m = (total >= t0) & (total <= t1)
        f = ((total - t0) / (t1 - t0))[m][:, None]
        rgb[m] = np.array(c0) + (np.array(c1) - np.array(c0)) * f
    alfa = np.clip(total * 2.2, 0, 1) * 225
    return Image.fromarray(np.dstack([rgb, alfa]).astype(np.uint8), "RGBA")


# --- Escenas -------------------------------------------------------------------------------------------
def portada(v, seg=4.2):
    base = cuadro_en(RAIZ / "data" / "input" / "input2.mp4", 60).resize((W * S, int(W * S * 288 / 352)), Image.LANCZOS)
    base = base.crop((0, (base.height - H * S) // 2, W * S, (base.height - H * S) // 2 + H * S)).filter(ImageFilter.GaussianBlur(6 * S))
    for k in range(int(FPS * seg)):
        a = suave(k / (FPS * 0.9))
        im = Image.blend(Image.new("RGB", base.size, FONDO), base, 0.35 * a)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle((0, 0) + im.size, fill=FONDO + (110,))
        tx(d, (640, 290), "Detección de aglomeraciones", F_MEGA, mezcla(FONDO, TEXTO, a), "mm")
        tx(d, (640, 364), "en tiempo real con YOLO11", F_GRANDE, mezcla(FONDO, AMBAR, a), "mm")
        tx(d, (640, 432), "Conteo de personas y mapas de calor sobre cámaras de vigilancia existentes", F_SUB,
           mezcla(FONDO, TENUE, a), "mm")
        v.cuadro(im)


def problema(v, seg=4.5):
    img = cuadro_en(RAIZ / "data" / "input" / "input2.mp4", 75)
    n = int(FPS * seg)
    for k in range(n):
        t = k / (n - 1)
        im, d = lienzo()
        encabezado(d, "El punto de partida", "Una cámara de calle común: 352 × 288 píxeles, personas de pocos píxeles.")
        z = 1 + 0.08 * t
        w0, h0 = img.width / z, img.height / z
        rec = img.crop((int((img.width - w0) * 0.6), int((img.height - h0) * 0.5),
                        int((img.width - w0) * 0.6 + w0), int((img.height - h0) * 0.5 + h0)))
        pegar(im, rec, 56, 112, 700)
        d = ImageDraw.Draw(im, "RGBA")
        preguntas = ["¿Cuántas personas hay ahora?", "¿Cuál es el promedio y el pico?", "¿En qué zonas se concentran?"]
        for i, q in enumerate(preguntas):
            a = suave(t * 3 - i * 0.6)
            caja(d, 790, 180 + i * 120, 1226, 270 + i * 120, relleno=PANEL + (int(255 * a),), borde=AMBAR + (int(200 * a),))
            tx(d, (812, 225 + i * 120), q, F_SUB, mezcla(PANEL, TEXTO, a), "lm")
        v.cuadro(im)


def entrenamiento(v, seg=10):
    lotes = []
    for f in ("train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg"):
        img = Image.open(RAIZ / "results" / f).convert("RGB")
        # el mosaico de Ultralytics trae relleno blanco: recortar al contenido
        g = np.asarray(img.convert("L")) < 240
        cols = np.where(g.mean(axis=0) > 0.25)[0]
        filas = np.where(g.mean(axis=1) > 0.25)[0]
        lotes.append(img.crop((cols[0], filas[0], cols[-1] + 1, filas[-1] + 1)))
    n = int(FPS * seg)
    x0, y0, x1, y1 = 700, 150, 1226, 560
    claves = [("metrics/mAP50(B)", "mAP50", AMBAR), ("metrics/precision(B)", "Precisión", AZUL),
              ("metrics/recall(B)", "Recall", ROSA)]
    for k in range(n):
        t = k / (n - 1)
        im, d = lienzo()
        encabezado(d, "Ajuste fino de YOLO11s",
                   f"{len(ENTRENO)} épocas · imágenes de 896 px · lotes de 6 · {HORAS:.1f} h de entrenamiento")
        # lotes con zoom lento y fundido
        idx = min(2, int(t * 3))
        tl = t * 3 - idx
        lote = lotes[idx]
        z = 1 + 0.12 * tl
        w0, h0 = lote.width / z, lote.height / z
        rec = lote.crop((int((lote.width - w0) / 2), int((lote.height - h0) / 2), int((lote.width + w0) / 2), int((lote.height + h0) / 2)))
        hh = 450
        ww = min(600, int(hh * rec.width / rec.height))
        pegar(im, rec, 56 + (600 - ww) / 2, 112, ww, hh)
        d = ImageDraw.Draw(im, "RGBA")
        tx(d, (56, 590), "Lotes de entrenamiento con las anotaciones del dataset", F_PEQ, TENUE)
        # curvas
        caja(d, x0 - 20, y0 - 40, x1 + 14, y1 + 70)
        for g in range(0, 11, 2):
            yy = y1 - (y1 - y0) * g / 10
            d.line(P(x0, yy) + P(x1, yy), fill=LINEA, width=S)
            tx(d, (x0 - 8, yy), f"{g / 10:.1f}", F_PEQ, TENUE, "rm")
        hasta = max(2, int(len(ENTRENO) * suave(t / 0.8)))
        for col, nombre, color in claves:
            pts = [P(x0 + (x1 - x0) * (i / (len(ENTRENO) - 1)), y1 - (y1 - y0) * float(r[col])) for i, r in enumerate(ENTRENO[:hasta])]
            d.line(pts, fill=color, width=int(2.4 * S), joint="curve")
            circulo = pts[-1]
            d.ellipse((circulo[0] - 5 * S, circulo[1] - 5 * S, circulo[0] + 5 * S, circulo[1] + 5 * S), fill=color)
        ep = ENTRENO[hasta - 1]
        tx(d, (x0, y0 - 22), f"Época {ep['epoch']}", F_CHIP, TEXTO, "lm")
        for i, (col, nombre, color) in enumerate(claves):
            tx(d, (x0 + i * 170, y1 + 34), f"{nombre} {float(ep[col]):.3f}", F_CHIP, color, "lm")
        v.cuadro(im, guardar="entrenamiento.png" if k == n - 1 else None)


def flujo(v, seg=6.5):
    pasos = [("Video o cámara", "archivo o índice de cámara", AZUL),
             ("YOLO11s", "896 px · conf 0.15 · IoU 0.30", AMBAR),
             ("Cajas", "una por cabeza detectada", ROSA),
             ("Conteo", "actual · promedio · máximo", VERDE),
             ("Mapa de calor", "centros + desenfoque", MORADO)]
    n = int(FPS * seg)
    for k in range(n):
        t = k / (n - 1)
        im, d = lienzo()
        encabezado(d, "El pipeline", "Cada cuadro pasa por el mismo camino; las salidas son video, mapa de calor y CSV.")
        for i, (tit, sub, col) in enumerate(pasos):
            a = suave(t * 3.2 - i * 0.45)
            x = 56 + i * 238
            y = 300 - 18 * (1 - a)
            caja(d, x, y, x + 210, y + 150, relleno=PANEL + (int(255 * a),), borde=col + (int(255 * a),), ancho=2)
            d.ellipse(P(x + 20, y + 22) + P(x + 44, y + 46), fill=col + (int(255 * a),))
            tx(d, (x + 20, y + 72), tit, F_CHIP, mezcla(FONDO, TEXTO, a))
            tx(d, (x + 20, y + 104), sub, F_PEQ, mezcla(FONDO, TENUE, a))
            if i < 4:
                b = suave(t * 3.2 - i * 0.45 - 0.35)
                d.line(P(x + 214, y + 75) + P(x + 214 + 20 * b, y + 75), fill=TENUE + (int(255 * b),), width=2 * S)
        a = suave(t * 2 - 1.2)
        tx(d, (640, 540), "Mapa de calor dinámico: cada cuadro se multiplica por 0.9, así lo viejo se desvanece",
           F_SUB, mezcla(FONDO, TENUE, a), "mm")
        v.cuadro(im, guardar="pipeline.png" if k == n - 1 else None)


def grafica(d, valores, hasta, x0, y0, x1, y1, color, maximo=None):
    maximo = maximo or max(valores)
    caja(d, x0 - 14, y0 - 14, x1 + 14, y1 + 14, r=10)
    n = len(valores)
    paso = max(1, n // 700)
    pts = [P(x0 + (x1 - x0) * i / (n - 1), y1 - (y1 - y0) * valores[i] / maximo) for i in range(0, hasta, paso)]
    if len(pts) > 1:
        relleno = pts + [P(x0 + (x1 - x0) * (hasta - 1) / (n - 1), y1), P(x0, y1)]
        d.polygon(relleno, fill=color + (40,))
        d.line(pts, fill=color, width=int(1.8 * S))
        cx = x0 + (x1 - x0) * (hasta - 1) / (n - 1)
        d.line(P(cx, y0) + P(cx, y1), fill=(255, 255, 255, 90), width=S)


def chip(d, x, y, etiqueta, valor, color):
    caja(d, x, y, x + 180, y + 64, r=10, borde=color + (160,))
    tx(d, (x + 14, y + 18), etiqueta, F_PEQ, TENUE, "lm")
    tx(d, (x + 14, y + 44), valor, F_CHIP, TEXTO, "lm")


def en_vivo(v, paso=3):
    total = len(CONTEO)
    det = cuadros(RAIZ / "output" / "salida_detectada.mp4")
    cal = cuadros(RAIZ / "output" / "heatmap_salida_detectada.mp4")
    acum, maximo = 0, 0
    v.marca("vivo_ini")
    for i, (fd, fc) in enumerate(zip(det, cal)):
        if i >= total:
            break
        acum += CONTEO[i]
        maximo = max(maximo, CONTEO[i])
        if i % paso:
            continue
        im, d = lienzo()
        encabezado(d, "Detección en vivo", f"Salida real del pipeline · {total:,} cuadros ({total / 15.34 / 60:.0f} min {total / 15.34 % 60:.0f} s) a ×{paso * 2} de velocidad".replace(",", "."))
        pegar(im, fd, 56, 104, 560, 458)
        mezclado = superponer(fd, fc)
        pegar(im, mezclado, 664, 104, 560, 458)
        d = ImageDraw.Draw(im, "RGBA")
        tx(d, (70, 548), "Detecciones (cajas de YOLO11s)", F_PEQ, (255, 255, 255))
        tx(d, (678, 548), "Mapa de calor dinámico sobre la escena", F_PEQ, (255, 255, 255))
        grafica(d, CONTEO, i + 1, 70, 606, 820, 690, AMBAR, maximo=max(CONTEO))
        tx(d, (78, 608), "personas por cuadro", F_PEQ, TENUE)
        chip(d, 860, 616, "Ahora", f"{CONTEO[i]} personas", AMBAR)
        chip(d, 1050, 616, "Promedio · Pico", f"{acum / (i + 1):.1f} · {maximo}", VERDE)
        v.cuadro(im, guardar="en_vivo.png" if i == 1500 else None)
    v.marca("vivo_fin")


def acumulado(v, seg=5.5):
    escena = cuadro_en(RAIZ / "data" / "input" / "input2.mp4", 75)
    capa = ACUMULADO.filter(ImageFilter.GaussianBlur(1))
    n = int(FPS * seg)
    for k in range(n):
        t = k / (n - 1)
        a = suave(t * 1.8 - 0.2)
        im, d = lienzo()
        encabezado(d, "Dónde se concentra la gente",
                   f"Suma de los {len(CONTEO):,} mapas de calor del video".replace(",", ".") + ", sobre la escena original")
        base = escena.resize((int(720 * S), int(720 * S * 288 / 352)), Image.LANCZOS).convert("RGBA")
        cap = capa.resize(base.size, Image.LANCZOS)
        cap.putalpha(cap.getchannel("A").point(lambda p: int(p * a)))
        base = Image.alpha_composite(base, cap).convert("RGB")
        m = Image.new("L", base.size, 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0) + base.size, 12 * S, fill=255)
        im.paste(base, P(56, 104), m)
        d = ImageDraw.Draw(im, "RGBA")
        b = suave(t * 2 - 0.8)
        for j, (tit, val) in enumerate((("Personas por cuadro", f"{sum(CONTEO) / len(CONTEO):.1f} de media"),
                                         ("Pico", f"{max(CONTEO)} personas"),
                                         ("Velocidad", f"{FPS_MEDIO:.1f} FPS de media"))):
            caja(d, 860, 150 + j * 130, 1226, 250 + j * 130, relleno=PANEL + (int(255 * b),))
            tx(d, (884, 184 + j * 130), tit, F_PEQ, mezcla(FONDO, TENUE, b), "lm")
            tx(d, (884, 220 + j * 130), val, F_NUM, mezcla(FONDO, TEXTO, b), "lm")
        v.cuadro(im, guardar="mapa_acumulado.png" if k == n - 1 else None)


def escena_densa(v, seg=6):
    frames = list(cuadros(DENSA / "gradio_1763611667.mp4"))
    n = int(FPS * seg)
    paso = max(1, len(frames) // n)
    for k in range(n):
        i = min(len(frames) - 1, k * paso)
        im, d = lienzo()
        encabezado(d, "Escena densa (interfaz Gradio)",
                   f"Otra cámara, procesada desde la interfaz web: hasta {max(CONTEO_DENSO)} personas en un cuadro.")
        pegar(im, frames[i], 56, 104, 700, 573)
        d = ImageDraw.Draw(im, "RGBA")
        j = min(len(CONTEO_DENSO) - 1, int(i * len(CONTEO_DENSO) / len(frames)))
        caja(d, 800, 180, 1226, 330)
        tx(d, (824, 214), "Personas en este cuadro", F_PEQ, TENUE, "lm")
        tx(d, (824, 274), str(CONTEO_DENSO[j]), F_MEGA, AMBAR, "lm")
        caja(d, 800, 360, 1226, 560)
        for t_, y in (("Subir un video o grabar con la cámara", 400), ("Elegir salidas: video, mapa de calor, PNG, CSV", 450),
                      ("Ajustar kernel y decaimiento del mapa", 500)):
            d.ellipse(P(824, y - 5) + P(834, y + 5), fill=AZUL)
            tx(d, (846, y), t_, F_PEQ, TEXTO, "lm")
        v.cuadro(im)


def resultados(v, seg=6):
    tarjetas = [("mAP50", float(MEJOR["metrics/mAP50(B)"]), AMBAR), ("Precisión", float(MEJOR["metrics/precision(B)"]), AZUL),
                ("Recall", float(MEJOR["metrics/recall(B)"]), ROSA), ("mAP50-95", float(MEJOR["metrics/mAP50-95(B)"]), MORADO)]
    n = int(FPS * seg)
    for k in range(n):
        t = k / (n - 1)
        im, d = lienzo()
        encabezado(d, "Resultados de validación", f"Mejor época: {MEJOR['epoch']} de {len(ENTRENO)} (YOLO11s, una clase: cabeza de persona)")
        for i, (nom, val, col) in enumerate(tarjetas):
            a = suave(t * 2.5 - i * 0.3)
            x = 56 + i * 297
            caja(d, x, 200, x + 270, 430, borde=col + (int(220 * a),), ancho=2)
            tx(d, (x + 135, 262), nom, F_SUB, mezcla(PANEL, TENUE, a), "mm")
            tx(d, (x + 135, 340), f"{val * a:.3f}", F_MEGA, mezcla(PANEL, col, a), "mm")
            d.rounded_rectangle(P(x + 30, 392) + P(x + 240, 400), 4 * S, fill=LINEA)
            d.rounded_rectangle(P(x + 30, 392) + P(x + 30 + 210 * val * a, 400), 4 * S, fill=col)
        b = suave(t * 2 - 1)
        tx(d, (640, 510), "Personas pequeñas en cámaras de baja resolución: el umbral de confianza por defecto es bajo (0.15)",
           F_PEQ, mezcla(FONDO, TENUE, b), "mm")
        tx(d, (640, 536), "para favorecer el recall, a costa de algunos falsos positivos.", F_PEQ, mezcla(FONDO, TENUE, b), "mm")
        v.cuadro(im, guardar="resultados.png" if k == n - 1 else None)


def cierre(v, seg=3.5):
    for k in range(int(FPS * seg)):
        a = suave(k / (FPS * 0.7))
        im, d = lienzo()
        tx(d, (640, 320), "Crowd Detection", F_MEGA, mezcla(FONDO, TEXTO, a), "mm")
        tx(d, (640, 392), "YOLO11s · OpenCV · Gradio · Python", F_SUB, mezcla(FONDO, AMBAR, a), "mm")
        tx(d, (640, 450), "github.com/CARLYGP/Crowd_Detection", F_PEQ, mezcla(FONDO, TENUE, a), "mm")
        v.cuadro(im)


def portada_png():
    """Imagen de portada del README (1600 × 640)."""
    PW, PH = 1600, 640
    fd = cuadro_en(RAIZ / "data" / "input" / "input2.mp4", 75).convert("RGBA")
    fd = Image.alpha_composite(fd, ACUMULADO.filter(ImageFilter.GaussianBlur(1)).resize(fd.size)).convert("RGB")
    ancho = 1060
    escena = fd.resize((ancho * S, int(ancho * S * 288 / 352)), Image.LANCZOS)
    y0 = (escena.height - PH * S) // 2
    escena = escena.crop((0, y0, escena.width, y0 + PH * S))
    im = Image.new("RGB", (PW * S, PH * S), FONDO)
    im.paste(escena, (PW * S - escena.width, 0))
    grad = Image.new("L", (PW * S, 1))
    for x in range(PW * S):
        grad.putpixel((x, 0), int(255 * suave((x - 560 * S) / (420 * S))))
    capa = Image.new("RGB", im.size, FONDO)
    im = Image.composite(im, capa, grad.resize(im.size))
    d = ImageDraw.Draw(im, "RGBA")
    d.text(P(80, 200), "Detección de", font=F_MEGA, fill=TEXTO)
    d.text(P(80, 278), "aglomeraciones", font=F_MEGA, fill=TEXTO)
    d.text(P(84, 370), "YOLO11 en tiempo real · conteo · mapas de calor", font=F_SUB, fill=AMBAR)
    for i, (et, val) in enumerate((("mAP50", f"{float(MEJOR['metrics/mAP50(B)']):.2f}"), ("FPS medio", f"{FPS_MEDIO:.1f}"),
                                   ("pico", f"{max(CONTEO)} personas"))):
        x = 84 + i * 190
        d.rounded_rectangle(P(x, 430) + P(x + 170, 500), 10 * S, fill=PANEL, outline=LINEA, width=S)
        d.text(P(x + 16, 446), et, font=F_PEQ, fill=TENUE)
        d.text(P(x + 16, 466), val, font=F_CHIP, fill=TEXTO)
    im.resize((PW, PH), Image.LANCZOS).save(DOCS / "portada.png")


def gif(mp4, ini, fin, salida, ancho=640, fps=10):
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{ini:.2f}", "-t", f"{fin - ini:.2f}", "-i", str(mp4), "-vf",
                    f"fps={fps},scale={ancho}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle", str(salida)], check=True)


def main():
    global ACUMULADO
    ACUMULADO = acumular_calor()
    print(f"Mejor época {MEJOR['epoch']}: mAP50 {MEJOR['metrics/mAP50(B)']} | {len(CONTEO)} cuadros, media "
          f"{sum(CONTEO) / len(CONTEO):.1f}, pico {max(CONTEO)}, {FPS_MEDIO:.1f} FPS | escena densa: pico {max(CONTEO_DENSO)}")
    portada_png()
    mp4 = DOCS / "crowd_detection.mp4"
    v = Video(mp4)
    portada(v)
    problema(v)
    entrenamiento(v)
    flujo(v)
    en_vivo(v)
    acumulado(v)
    escena_densa(v)
    resultados(v)
    cierre(v)
    v.cerrar()
    gif(mp4, v.marcas["vivo_ini"] + 2, v.marcas["vivo_ini"] + 12, DOCS / "en_vivo.gif", ancho=600, fps=9)
    for f in ("crowd_detection.mp4", "en_vivo.gif", "portada.png"):
        print(f, f"{(DOCS / f).stat().st_size / 1e6:.1f} MB")
    print(f"duración {v.n / FPS:.1f} s")


if __name__ == "__main__":
    main()
