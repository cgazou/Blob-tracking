import numpy as np
import cv2
import argparse
import random
import sys
import os
import re
from collections import deque

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Puis pour charger config.txt
config_path = os.path.join(APP_DIR, 'config.txt')

# ── Paramètres trails ───────────────────────────────────────────────────────
TRAIL_MAX_LEN    = 40
TRAIL_MATCH_DIST = 150

class BlobTracker:
    def __init__(self):
        self.tracks     = {}
        self.next_id    = 0
        self.blob_names = {}
        self.names_list = ['real','double','station','world','dream',
                           'bizoocross','helmut','only','path','2world']

    def _catmull_rom(self, pts, resolution=8):
        if len(pts) < 4:
            return pts
        basis = 0.5 * np.array([
            [0, 2, 0, 0], [-1, 0, 1, 0],
            [2, -5, 4, -1], [-1, 3, -3, 1]
        ], dtype=np.float32)
        t = np.linspace(0, 1, resolution, dtype=np.float32)
        T = np.column_stack([np.ones(resolution), t, t*t, t*t*t]) @ basis
        points = np.array(pts, dtype=np.float32)
        result = []
        for i in range(1, len(pts) - 2):
            result.extend((T @ points[i-1:i+3]).tolist())
        return result

    def _match_and_update(self, detections):
        unmatched = set(self.tracks.keys())
        assigned  = []
        for cx, cy, w, h in detections:
            best_id, best_dist = None, float('inf')
            for tid in unmatched:
                tx, ty = self.tracks[tid]['center']
                dist = np.sqrt((cx-tx)**2 + (cy-ty)**2)
                if dist < best_dist and dist < TRAIL_MATCH_DIST:
                    best_dist, best_id = dist, tid
            if best_id is not None:
                unmatched.discard(best_id)
                self.tracks[best_id].update({'center': (cx,cy), 'w': w, 'h': h})
                self.tracks[best_id]['history'].append((cx, cy))
                assigned.append(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'center': (cx, cy), 'w': w, 'h': h,
                    'history': deque([(cx, cy)], maxlen=TRAIL_MAX_LEN)
                }
                self.blob_names[tid] = random.choice(self.names_list)
                assigned.append(tid)
        for tid in unmatched:
            del self.tracks[tid]
            self.blob_names.pop(tid, None)
        return assigned

    def _draw_dotted_line(self, img, pt1, pt2, color, thickness, gap=8):
        dist = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
        if dist == 0:
            return
        dx, dy = (pt2[0]-pt1[0])/dist, (pt2[1]-pt1[1])/dist
        d, dash = 0, gap // 2
        while d < dist:
            x1, y1 = int(pt1[0]+dx*d), int(pt1[1]+dy*d)
            x2, y2 = int(pt1[0]+dx*min(d+dash,dist)), int(pt1[1]+dy*min(d+dash,dist))
            cv2.line(img, (x1,y1), (x2,y2), color, thickness, cv2.LINE_4)
            d += gap

    def _draw_annotations(self, canvas, track_ids, config, out_w, out_h, is_alpha=False):
        """
        Dessine toutes les annotations sur canvas (BGR ou BGRA).
        is_alpha=True : couleurs opaques sur fond transparent (export PNG).
        is_alpha=False : couleurs sur fond vidéo (preview + mp4).
        """
        white = (255, 255, 255, 255) if is_alpha else (255, 255, 255)
        col   = config['outline_color'] + (255,) if is_alpha else config['outline_color']
        t_col = config['trail_color']   + (255,) if is_alpha else config['trail_color']
        font  = cv2.FONT_HERSHEY_DUPLEX

        for tid in track_ids:
            track = self.tracks[tid]
            cx, cy = int(track['center'][0]), int(track['center'][1])
            w,  h  = int(track['w']),         int(track['h'])
            x0 = np.clip(cx - w//2, 0, out_w)
            y0 = np.clip(cy - h//2, 0, out_h)
            x1 = np.clip(cx + w//2, 0, out_w)
            y1 = np.clip(cy + h//2, 0, out_h)
            th = config['blob_thickness']

            # Trail Catmull-Rom entre tous les blobs
            if config['draw_trails'] and len(track_ids) >= 2:
                pts = [(int(self.tracks[t]['center'][0]),
                        int(self.tracks[t]['center'][1])) for t in track_ids]
                smooth_pts = self._catmull_rom(pts) if len(pts) >= 4 else pts
                for i in range(len(smooth_pts)-1):
                    pt1_ = (int(smooth_pts[i][0]),   int(smooth_pts[i][1]))
                    pt2_ = (int(smooth_pts[i+1][0]), int(smooth_pts[i+1][1]))
                    if config['use_dotted']:
                        self._draw_dotted_line(canvas, pt1_, pt2_, t_col, config['trail_thickness'])
                    else:
                        cv2.line(canvas, pt1_, pt2_, t_col, config['trail_thickness'], cv2.LINE_4)

            # Box ou brackets
            if config['show_boxes']:
                if config['use_brackets']:
                    bw = int((x1-x0) * config['bracket_length'])
                    bh = int((y1-y0) * config['bracket_length'])
                    cv2.line(canvas, (x0,y0), (x0+bw,y0), col, th)
                    cv2.line(canvas, (x0,y0), (x0,y0+bh), col, th)
                    cv2.line(canvas, (x1,y0), (x1-bw,y0), col, th)
                    cv2.line(canvas, (x1,y0), (x1,y0+bh), col, th)
                    cv2.line(canvas, (x0,y1), (x0+bw,y1), col, th)
                    cv2.line(canvas, (x0,y1), (x0,y1-bh), col, th)
                    cv2.line(canvas, (x1,y1), (x1-bw,y1), col, th)
                    cv2.line(canvas, (x1,y1), (x1,y1-bh), col, th)
                else:
                    cv2.rectangle(canvas, (x0,y0), (x1,y1), col, th)

            # Point central
            if config['show_center_dot']:
                r       = config['center_dot_radius']
                col_dot = config['center_dot_color'] + (255,) if is_alpha else config['center_dot_color']
                if config['center_dot_style'] == 'cross':
                    cs = r * 3
                    cv2.line(canvas, (cx-cs,cy), (cx+cs,cy), col_dot, 2, cv2.LINE_4)
                    cv2.line(canvas, (cx,cy-cs), (cx,cy+cs), col_dot, 2, cv2.LINE_4)
                else:
                    cv2.circle(canvas, (cx,cy), r, col_dot, -1)

            # Label
            if config['show_ids']:
                name = self.blob_names.get(tid, f"ID{tid}")
                text = f"{name} X:{cx} Y:{cy}"
                fs, fth = config['fixed_font_scale'], config['fixed_font_thickness']
                (tw, th_), _ = cv2.getTextSize(text, font, fs, fth)
                tx = (x0+x1)//2 - tw//2
                ty = max(y0 - config['fixed_font_offset'], th_)
                if config['show_leaders']:
                    cv2.line(canvas, (cx,cy), (tx+tw//2,ty), col, 1, cv2.LINE_4)
                cv2.putText(canvas, text, (tx,ty), font, fs, white, fth, cv2.LINE_4)

        # Connexions
        if config['draw_connections'] and len(track_ids) > 1:
            centers  = [(int(self.tracks[t]['center'][0]), int(self.tracks[t]['center'][1])) for t in track_ids]
            avg_size = np.mean([int(self.tracks[t]['h']) for t in track_ids])
            conn_th  = max(1, config['blob_thickness']//2)
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dx = centers[j][0]-centers[i][0]
                    dy = centers[j][1]-centers[i][1]
                    if np.sqrt(dx*dx+dy*dy) <= avg_size * 3:
                        if config['use_dotted']:
                            self._draw_dotted_line(canvas, centers[i], centers[j], col, conn_th)
                        else:
                            cv2.line(canvas, centers[i], centers[j], col, conn_th, cv2.LINE_4)

        # Métriques
        if config['show_metrics'] and track_ids:
            fs, fth = config['fixed_font_scale'], config['fixed_font_thickness']
            (_, lh), _ = cv2.getTextSize("A", font, fs, fth)
            gap, y_pos = int(lh*1.8), int(lh*1.8)
            cv2.putText(canvas, "TRACKING DATA", (10,y_pos), font, fs, col, fth, cv2.LINE_4)
            y_pos += gap
            for tid in track_ids:
                t = self.tracks[tid]
                cx_, cy_ = int(t['center'][0]), int(t['center'][1])
                hist = list(t['history'])
                spd = np.sqrt((hist[-1][0]-hist[-2][0])**2+(hist[-1][1]-hist[-2][1])**2) if len(hist)>=2 else 0.0
                txt = f"ID:{tid} X:{cx_/out_w:.2f} Y:{cy_/out_h:.2f} SPD:{spd:.1f} {int(t['w'])}x{int(t['h'])}px"
                cv2.putText(canvas, txt, (10,y_pos), font, fs, white, fth, cv2.LINE_4)
                y_pos += gap

        # Grille
        if config['show_grid']:
            gc = col  # même couleur avec alpha si besoin
            x = 0
            while x < out_w:
                cv2.line(canvas, (int(x),0), (int(x),out_h), gc, 1, cv2.LINE_4)
                x += config['grid_spacing']
            y = 0
            while y < out_h:
                cv2.line(canvas, (0,int(y)), (out_w,int(y)), gc, 1, cv2.LINE_4)
                y += config['grid_spacing']

    def process_frame(self, frame, config, export_alpha=False):
        out_h, out_w = frame.shape[:2]

        # Threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if config['threshold_mode'] == 'auto':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, config['threshold_value'], 255, cv2.THRESH_BINARY)
        if config['invert_threshold']:
            thresh = cv2.bitwise_not(thresh)

        # Downsample détection
        res = config['resolution_scale']
        if res < 1.0:
            det_w = max(1, int(out_w * res))
            det_h = max(1, int(out_h * res))
            thresh_det = cv2.resize(thresh, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            thresh_det = thresh
            det_w, det_h = out_w, out_h

        sx, sy = out_w / det_w, out_h / det_h
        area_scale = res * res
        adj_min = max(1.0, config['min_area'] * area_scale)
        adj_max = max(adj_min + 1, config['max_area'] * area_scale)

        contours, _ = cv2.findContours(thresh_det, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [(c, cv2.boundingRect(c)) for c in contours
                    if adj_min <= cv2.contourArea(c) <= adj_max]
        filtered = sorted(filtered, key=lambda c: cv2.contourArea(c[0]), reverse=True)[:config['max_blobs']]

        detections = [((x+w/2)*sx, (y+h/2)*sy, w*sx, h*sy) for _, (x,y,w,h) in filtered]
        track_ids  = self._match_and_update(detections)

        # ── Canvas vidéo (BGR) — preview + mp4
        out_img = frame.copy()
        self._draw_annotations(out_img, track_ids, config, out_w, out_h, is_alpha=False)

        # ── Canvas alpha (BGRA) — PNG export
        alpha_img = None
        if export_alpha:
            alpha_img = np.zeros((out_h, out_w, 4), dtype=np.uint8)  # fond 100% transparent
            self._draw_annotations(alpha_img, track_ids, config, out_w, out_h, is_alpha=True)
            # Générer le canal alpha depuis les pixels dessinés
            drawn_mask = np.any(alpha_img[:,:,:3] > 0, axis=2)
            alpha_img[:,:,3] = np.where(drawn_mask, 255, 0)

        return out_img, alpha_img

def load_config_from_file(config_file='config.txt'):
    """Charge la configuration depuis un fichier texte (optionnel)"""
    config = {}
    
    if not os.path.exists(config_file):
        print(f"Info: {config_file} non trouvé, utilisation des valeurs par défaut")
        return config
    
    print(f"Info: chargement de la configuration depuis {config_file}")
    
    # Pattern pour capturer KEY = VALUE
    pattern = re.compile(r'^\s*([A-Z_]+)\s*=\s*(.+?)\s*$', re.IGNORECASE)
    
    with open(config_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            match = pattern.match(line)
            if not match:
                print(f"Warning: ligne {line_num} ignorée: {line}")
                continue
            
            key = match.group(1).upper()
            value = match.group(2).strip()
            
            # Conversion automatique
            if value.lower() in ('true', 'yes', 'on', '1'):
                config[key] = True
            elif value.lower() in ('false', 'no', 'off', '0'):
                config[key] = False
            elif value.isdigit():
                config[key] = int(value)
            elif re.match(r'^\d+\.\d+$', value):
                config[key] = float(value)
            elif re.match(r'^\d+,\d+,\d+$', value):
                config[key] = tuple(int(x) for x in value.split(','))
            else:
                config[key] = value
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Blob Tracker 4K')
    parser.add_argument('--input',      type=str,   default='0')
    parser.add_argument('--output',     type=str,   default=None)
    parser.add_argument('--png-output', type=str,   default=None)
    parser.add_argument('--threshold',  type=int,   default=None)
    parser.add_argument('--min-area',   type=float, default=None)
    parser.add_argument('--max-area',   type=float, default=None)
    parser.add_argument('--max-blobs',  type=int,   default=None)
    args = parser.parse_args()

    tracker = BlobTracker()

    # Valeurs par defaut
    config = {
        'threshold_mode':       'manual',
        'threshold_value':      127,
        'invert_threshold':     False,
        'min_area':             10000,
        'max_area':             100000,
        'max_blobs':            20,
        'resolution_scale':     0.5,
        'outline_color':        (255, 255, 255),
        'trail_color':          (255, 255, 255),
        'trail_thickness':      2,
        'blob_thickness':       3,
        'draw_connections':     True,
        'draw_trails':          True,
        'show_ids':             True,
        'show_leaders':         False,
        'show_metrics':         False,
        'show_grid':            False,
        'grid_spacing':         50.0,
        'use_brackets':         False,
        'bracket_length':       0.3,
        'use_dotted':           False,
        'show_boxes':           True,
        'show_center_dot':      False,
        'center_dot_style':     'dot',
        'center_dot_radius':    4,
        'center_dot_color':     (255, 255, 0),
        'fixed_font_scale':     1.2,
        'fixed_font_thickness': 2,
        'fixed_font_offset':    15,
    }
    
    # Charger le fichier config.txt s'il existe (optionnel)
    file_config = load_config_from_file('config.txt')
    
    # Appliquer les valeurs du fichier (en ignorant la casse)
    for key, value in file_config.items():
        found = False
        for existing_key in config.keys():
            if existing_key.upper() == key:
                config[existing_key] = value
                found = True
                break
        if not found:
            print(f"Warning: key '{key}' ignored (not found in config)")
    
    # CLI arguments effacent
    if args.threshold is not None:
        config['threshold_value'] = args.threshold
    if args.min_area is not None:
        config['min_area'] = args.min_area
    if args.max_area is not None:
        config['max_area'] = args.max_area
    if args.max_blobs is not None:
        config['max_blobs'] = args.max_blobs
    
    try:
        source = int(args.input)
    except ValueError:
        source = args.input

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    INITIAL_W = 1280
    INITIAL_H = int(INITIAL_W * cap_h / cap_w)
    cv2.namedWindow('Blob Tracker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Blob Tracker', INITIAL_W, INITIAL_H)

    # Writer mp4
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (cap_w, cap_h))

    # Dossier PNG alpha
    export_alpha = args.png_output is not None
    if export_alpha:
        os.makedirs(args.png_output, exist_ok=True)
        print(f"PNG alpha -> {args.png_output}/  ({total} frames attendues)")

    print(f"Source 4K : {cap_w}x{cap_h}  |  Preview : {INITIAL_W}x{INITIAL_H}  |  FPS : {fps}")
    print("Fenetre redimensionnable a la souris")
    print("q=quit  t=trails  c=connections  b=brackets  m=metrics  g=grid  d=dotted  x=boxes  p=dot")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_4k, alpha_4k = tracker.process_frame(frame, config, export_alpha=export_alpha)

        # Export mp4
        if writer:
            writer.write(output_4k)

        # Export PNG alpha
        if export_alpha and alpha_4k is not None:
            folder_name = os.path.basename(args.png_output.rstrip('/\\'))
            path = os.path.join(args.png_output, f"{folder_name}_{frame_count:06d}.png")
            cv2.imwrite(path, alpha_4k)
            if frame_count % 25 == 0:
                print(f"  frame {frame_count}/{total}", end='\r')

        cv2.imshow('Blob Tracker', output_4k)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('t'):
            config['draw_trails'] = not config['draw_trails']
            print(f"Trails: {'ON' if config['draw_trails'] else 'OFF'}")
        elif key == ord('c'):
            config['draw_connections'] = not config['draw_connections']
            print(f"Connections: {'ON' if config['draw_connections'] else 'OFF'}")
        elif key == ord('b'):
            config['use_brackets'] = not config['use_brackets']
            print(f"Brackets: {'ON' if config['use_brackets'] else 'OFF'}")
        elif key == ord('m'):
            config['show_metrics'] = not config['show_metrics']
            print(f"Metrics: {'ON' if config['show_metrics'] else 'OFF'}")
        elif key == ord('g'):
            config['show_grid'] = not config['show_grid']
            print(f"Grid: {'ON' if config['show_grid'] else 'OFF'}")
        elif key == ord('d'):
            config['use_dotted'] = not config['use_dotted']
            print(f"Dotted: {'ON' if config['use_dotted'] else 'OFF'}")
        elif key == ord('x'):
            config['show_boxes'] = not config['show_boxes']
            print(f"Boxes: {'ON' if config['show_boxes'] else 'OFF'}")
        elif key == ord('p'):
            states = [('off', False, 'dot'), ('dot', True, 'dot'), ('cross', True, 'cross')]
            current = 0 if not config['show_center_dot'] else (1 if config['center_dot_style'] == 'dot' else 2)
            next_s  = (current + 1) % len(states)
            label, config['show_center_dot'], config['center_dot_style'] = states[next_s]
            print(f"Center: {label.upper()}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if export_alpha:
        print(f"\nDone! {frame_count} PNG alpha exportes dans {args.png_output}/")
        print(f"Pour convertir en ProRes 4444 :")
        print(f"  ffmpeg -framerate {fps} -i {args.png_output}/frame_%06d.png -c:v prores_ks -profile:v 4444 output/alpha.mov")
    else:
        print("Done!")


if __name__ == '__main__':
    main()