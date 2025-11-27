import cv2
import numpy as np
from typing import Tuple
from .paths import ensure_odd_kernel
from .heatmap import heatmap_colorize


def process_and_write_frame(
    frame,
    detector,
    overlay,
    fps_sm,
    people_counter,
    summary_acc,
    writer,
    heatmap_video_writer,
    args,
    w,
    h,
    heatmap_frames_written,
    heatmap_warned,
    summary_acc_raw=None,
) -> Tuple[int, bool]:
    """
    Procesa un frame: detección, actualización de heatmap (con o sin decaimiento),
    overlay, guardado de video, y vista combinada en pantalla completa.
    """

    # ---------------------------
    # 1. DETECCIÓN
    # ---------------------------
    names = getattr(detector, "names", getattr(getattr(detector, "model", {}), "names", {}))
    dets = detector.infer(frame)
    current_count = len(dets)
    people_counter.update(current_count)

    centers = []

    for (x1, y1, x2, y2), conf, cid in dets:
        name = names[cid] if cid in names else f"id{cid}"
        label = f"{name} {conf:.2f}"
        overlay.draw_box(frame, (x1, y1, x2, y2), label)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centers.append((cx, cy))

    detector.last_centers = centers

    # ---------------------------
    # 2. HEATMAP ACCUMULATOR
    # ---------------------------
    try:
        decay = float(args.heatmap_decay)
    except Exception:
        decay = 1.0

    if decay < 1.0:
        summary_acc *= decay     # con decaimiento
    # si decay = 1 → NO se borra rastro

    # Sumamos al heatmap
    for cx, cy in centers:
        ix = int(round(cx))
        iy = int(round(cy))
        if 0 <= ix < w and 0 <= iy < h:
            summary_acc[iy, ix] += 1.0
            if summary_acc_raw is not None:
                summary_acc_raw[iy, ix] += 1.0  # heatmap sin decaimiento

    # ---------------------------
    # 3. GENERAR HEATMAP FRAME
    # ---------------------------
    heat_col = None
    if heatmap_video_writer is not None:
        k_vid = args.heatmap_video_kernel if args.heatmap_video_kernel else args.heatmap_kernel
        k = ensure_odd_kernel(k_vid, default=1)

        try:
            heat_col = heatmap_colorize(summary_acc, kernel=k)
        except Exception:
            summary_vis = summary_acc.copy()
            if k > 1:
                try:
                    summary_vis = cv2.GaussianBlur(summary_vis, (k, k), 0)
                except Exception:
                    pass

            maxv = summary_vis.max()
            if maxv <= 0:
                norm = np.zeros_like(summary_vis, dtype=np.uint8)
            else:
                norm = (summary_vis / maxv * 255).astype(np.uint8)

            heat_col = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        if heat_col.shape[:2] != (h, w):
            heat_col = cv2.resize(heat_col, (w, h))

        try:
            if hasattr(heatmap_video_writer, "writer") and heatmap_video_writer.writer.isOpened():
                heatmap_video_writer.write(heat_col)
                heatmap_frames_written += 1
        except Exception:
            if not heatmap_warned:
                print("[WARN] Could not write heatmap frame")
                heatmap_warned = True

    # ---------------------------
    # 4. OVERLAYS EN EL FRAME
    # ---------------------------
    overlay.draw_counts(frame, people_counter.current, people_counter.average, people_counter.max_count)
    overlay.draw_fps(frame, fps_sm.tick())

    # ---------------------------
    # 5. SPLIT VIEW (DETECCIONES | HEATMAP) — FULL SCREEN
    # ---------------------------
    if heat_col is not None:
        try:
            if heat_col.shape[:2] != frame.shape[:2]:
                heat_col = cv2.resize(heat_col, (frame.shape[1], frame.shape[0]))

            combined = np.hstack((frame, heat_col))

            # ---------- FULLSCREEN REAL ----------
            cv2.namedWindow("YOLO Split View (Detections | Heatmap)", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "YOLO Split View (Detections | Heatmap)",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
            # -------------------------------------

            cv2.imshow("YOLO Split View (Detections | Heatmap)", combined)

        except Exception as e:
            print("[WARN] Error generating split view:", e)

    # ---------------------------
    # 6. ESCRIBIR VIDEO DE DETECCIÓN
    # ---------------------------
    if writer is not None and getattr(writer, "writer", None) is not None:
        writer.write(frame)

    return heatmap_frames_written, heatmap_warned
