"""
overlay.py — HUD profesional para conteo y FPS.
"""

import cv2
from typing import Tuple

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Overlay:
    """
    Dibujos profesionales para anotaciones:
      - draw_box: caja de detección
      - draw_fps: indicador FPS
      - draw_counts: panel HUD elegante (Current / Avg / Max)
    """

    def __init__(self, font_scale: float = 0.6, font_thickness: int = 2):
        self.font = DEFAULT_FONT
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    # -------------------------------------------------------
    # CAJA DE DETECCIÓN
    # -------------------------------------------------------
    def draw_box(
        self, frame, box: Tuple[int, int, int, int], label: str, color=(255, 0, 0)   # Azul clásico OpenCV

    ) -> None:
        """Caja con borde fino + etiqueta."""
        x1, y1, x2, y2 = map(int, box)

        # Borde más estilizado
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Texto y fondo
        (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1, cv2.LINE_AA)

        cv2.putText(
            frame,
            label,
            (x1 + 3, y1 - 4),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------
    # FPS
    # -------------------------------------------------------
    def draw_fps(self, frame, fps: float) -> None:
        """Texto elegante FPS (arriba izquierda)."""
        cv2.putText(
            frame,
            f"{fps:.1f} FPS",
            (10, 28),
            self.font,
            0.8,
            (0, 255, 100),
            2,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------
    # PANEL PROFESIONAL DE CONTEO
    # -------------------------------------------------------
    def draw_counts(self, frame, current: int, average: float, max_count: int) -> None:
        """
        HUD profesional con:
        - Fondo transparente
        - Bordes redondeados
        - Iconos minimalistas
        """

        # ---- Texto del HUD ----
        lines = [
            f" Current : {current}",
            f" Average : {average:.1f}",
            f" Maximum : {max_count}",
        ]

        pad_x = 12
        pad_y = 12

        widths = []
        heights = []

        for text in lines:
            (tw, th), _ = cv2.getTextSize(text, self.font, 0.65, 2)
            widths.append(tw)
            heights.append(th)

        panel_w = max(widths) + pad_x * 2
        panel_h = sum(heights) + pad_y * 2 + (len(lines) - 1) * 6

        # Posición del panel
        x0, y0 = 10, 50

        # ---- Fondo con transparencia ----
        overlay = frame.copy()

        # Color negro con 45% de opacidad
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + panel_w, y0 + panel_h),
            (30, 30, 30),
            -1,
            cv2.LINE_AA,
        )

        # Fusionar con transparencia
        alpha = 0.45  # cambia entre 0.3 y 0.6 si quieres más o menos opaco
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # ---- Bordes redondeados (simulación con rectángulos pequeños) ----
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + panel_w, y0 + panel_h),
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # ---- Escribir texto ----
        y_text = y0 + pad_y + heights[0]

        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x0 + pad_x, y_text),
                self.font,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if i < len(lines) - 1:
                y_text += heights[i + 1] + 6
