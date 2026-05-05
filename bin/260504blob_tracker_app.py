import numpy as np
import cv2
import argparse
import random
from collections import deque

class BlobTracker:
    def __init__(self):
        self.detector = None
        self.last_params = None
        
        self.frame_skip_cache = {
            'counter': 0,
            'last_keypoints': [],
            'last_centers': [],
            'last_sizes': [],
            'velocities': [],
            'smoothed_centers': [],
            'smoothed_velocities': [],
            'smoothed_sizes': []
        }
        
        self.trail_cache = {'interpolated': None, 'cache_key': None}
        
        self.basis_matrix = 0.5 * np.array([
            [0, 2, 0, 0],
            [-1, 0, 1, 0],
            [2, -5, 4, -1],
            [-1, 3, -3, 1]
        ], dtype=np.float32)

        self.names_list = [
                'real',
                'double',
                'station',
                'world', 
                'dream',
                'bizoocross',
                'helmut',
                'only',
                'path',
                '2world'
            ]
        self.blob_names = {}

    
    def get_basis_matrix(self, resolution):
        t = np.linspace(0, 1, resolution, dtype=np.float32)
        t2 = t * t
        t3 = t2 * t
        return np.column_stack([
            np.ones(resolution, dtype=np.float32),
            t, t2, t3
        ]) @ self.basis_matrix
    
    def interpolate_catmull_rom_cached(self, centers, resolution=8):
        if len(centers) < 4:
            return centers
        cache_key = (tuple(map(tuple, centers)), resolution)
        if self.trail_cache.get('cache_key') == cache_key:
            return self.trail_cache['interpolated']
        points_array = np.array(centers, dtype=np.float32)
        basis_matrix = self.get_basis_matrix(resolution)
        result = []
        for i in range(1, len(centers) - 2):
            control_points = points_array[i-1:i+3]
            segment = basis_matrix @ control_points
            result.extend(segment.tolist())
        self.trail_cache['interpolated'] = result
        self.trail_cache['cache_key'] = cache_key
        return result
    
    def smooth_interpolate_positions(self, last_centers, last_sizes, smoothed_velocities,
                                     frame_fraction, smoothing_factor):
        if not last_centers or not smoothed_velocities:
            return last_centers, last_sizes
        if frame_fraction < 0.5:
            t = frame_fraction * 2
            eased_fraction = t * t * t / 2
        else:
            t = (frame_fraction - 0.5) * 2
            eased_fraction = 0.5 + (1 - (1 - t) * (1 - t) * (1 - t)) / 2
        final_fraction = eased_fraction * smoothing_factor + frame_fraction * (1 - smoothing_factor)
        interpolated_centers = []
        interpolated_sizes = []
        for i, (center, size) in enumerate(zip(last_centers, last_sizes)):
            if i < len(smoothed_velocities):
                vel = smoothed_velocities[i]
                new_x = int(center[0] + vel[0] * final_fraction)
                new_y = int(center[1] + vel[1] * final_fraction)
                interpolated_centers.append((new_x, new_y))
                interpolated_sizes.append(size)
            else:
                interpolated_centers.append(center)
                interpolated_sizes.append(size)
        return interpolated_centers, interpolated_sizes
    
    def exponential_smooth_velocity(self, new_velocity, old_velocity, alpha=0.3):
        if not old_velocity:
            return new_velocity
        smoothed_vx = alpha * new_velocity[0] + (1 - alpha) * old_velocity[0]
        smoothed_vy = alpha * new_velocity[1] + (1 - alpha) * old_velocity[1]
        return (smoothed_vx, smoothed_vy)
    
    def draw_dotted_line(self, img, pt1, pt2, color, thickness, gap=8):
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist == 0:
            return
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist
        current_dist = 0
        dash_length = gap // 2
        while current_dist < dist:
            x1 = int(pt1[0] + dx * current_dist)
            y1 = int(pt1[1] + dy * current_dist)
            end_dist = min(current_dist + dash_length, dist)
            x2 = int(pt1[0] + dx * end_dist)
            y2 = int(pt1[1] + dy * end_dist)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_4)
            current_dist += gap
    
    def process_frame(self, frame, config):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if config['threshold_mode'] == 'auto':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, config['threshold_value'], 255, cv2.THRESH_BINARY)
        
        if config['invert_threshold']:
            thresh = cv2.bitwise_not(thresh)
        
        out_h, out_w = frame.shape[:2]
        
        resolution_scale = config['resolution_scale']
        if resolution_scale < 1.0:
            detect_w = max(1, int(out_w * resolution_scale))
            detect_h = max(1, int(out_h * resolution_scale))
            thresh_detect = cv2.resize(thresh, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
            inv_scale = 1.0 / resolution_scale
        else:
            thresh_detect = thresh
            detect_w, detect_h = out_w, out_h
            inv_scale = 1.0
        
        self.frame_skip_cache['counter'] += 1
        should_detect = True
        frame_within_skip = self.frame_skip_cache['counter'] % config['frame_skip_interval']
        
        if config['enable_skip'] and frame_within_skip != 0:
            should_detect = False
            frame_fraction = frame_within_skip / config['frame_skip_interval']
            centers, sizes_for_interp = self.smooth_interpolate_positions(
                self.frame_skip_cache['smoothed_centers'],
                self.frame_skip_cache['last_sizes'],
                self.frame_skip_cache['smoothed_velocities'],
                frame_fraction,
                config['motion_smoothing']
            )
        
        if should_detect:
            area_scale = resolution_scale * resolution_scale
            adjusted_min = max(1.0, config['min_area'] * area_scale)
            adjusted_max = max(adjusted_min + 1, config['max_area'] * area_scale)
            
            params_tuple = (adjusted_min, adjusted_max, config['invert_threshold'])
            if self.detector is None or self.last_params != params_tuple:
                params = cv2.SimpleBlobDetector_Params()
                params.minThreshold = 50
                params.maxThreshold = 220
                params.thresholdStep = 10
                params.filterByArea = True
                params.minArea = adjusted_min
                params.maxArea = adjusted_max
                params.filterByColor = True
                params.blobColor = 255 if not config['invert_threshold'] else 0
                params.filterByCircularity = False
                params.filterByConvexity = False
                params.filterByInertia = False
                self.detector = cv2.SimpleBlobDetector_create(params)
                self.last_params = params_tuple
            
            keypoints = self.detector.detect(thresh_detect)
            
            if len(keypoints) > config['max_blobs']:
                keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:config['max_blobs']]
            
            self.frame_skip_cache['last_keypoints'] = keypoints
            centers = None
            sizes_for_interp = None
        
        out_img = frame.copy()
        
        scale_x = (out_w / detect_w) #* inv_scale
        scale_y = (out_h / detect_h) #* inv_scale
        scale_avg = (scale_x + scale_y) * 0.5
        white = (255, 255, 255)
        
        if should_detect:
            keypoints = self.frame_skip_cache['last_keypoints']
            num_kps = len(keypoints)
            
            if num_kps > 0:
                kp_data = np.array([[kp.pt[0], kp.pt[1], kp.size] for kp in keypoints], dtype=np.float32)
                centers_x = (kp_data[:, 0] * scale_x).astype(np.int32)
                centers_y = (kp_data[:, 1] * scale_y).astype(np.int32)
                sizes_scaled = (kp_data[:, 2] * scale_avg).astype(np.int32)
                detected_centers = list(zip(centers_x.tolist(), centers_y.tolist()))
                sizes_for_interp = sizes_scaled.tolist()
            
                if self.frame_skip_cache['smoothed_sizes'] and len(self.frame_skip_cache['smoothed_sizes']) == len(sizes_for_interp):
                    smoothed_sizes = []
                    size_alpha = 0.3 + (config['size_smoothing'] * 0.4)
                    for curr_size, prev_smooth_size in zip(sizes_for_interp, self.frame_skip_cache['smoothed_sizes']):
                        smooth_size = int(size_alpha * curr_size + (1 - size_alpha) * prev_smooth_size)
                        smoothed_sizes.append(smooth_size)
                    self.frame_skip_cache['smoothed_sizes'] = smoothed_sizes
                    sizes_scaled = np.array(smoothed_sizes, dtype=np.int32)
                else:
                    self.frame_skip_cache['smoothed_sizes'] = sizes_for_interp
                
                half_sizes = (sizes_scaled * 0.5).astype(np.int32)
                
                if (self.frame_skip_cache['last_centers'] and
                        len(self.frame_skip_cache['last_centers']) == len(detected_centers)):
                    raw_velocities = []
                    smoothed_velocities = []
                    for i, (curr, prev) in enumerate(zip(detected_centers, self.frame_skip_cache['last_centers'])):
                        vel_x = (curr[0] - prev[0]) / config['frame_skip_interval']
                        vel_y = (curr[1] - prev[1]) / config['frame_skip_interval']
                        raw_vel = (vel_x, vel_y)
                        raw_velocities.append(raw_vel)
                        if i < len(self.frame_skip_cache['smoothed_velocities']):
                            old_smooth_vel = self.frame_skip_cache['smoothed_velocities'][i]
                            alpha_vel = 0.7 - (config['motion_smoothing'] * 0.5)
                            smooth_vel = self.exponential_smooth_velocity(raw_vel, old_smooth_vel, alpha_vel)
                        else:
                            smooth_vel = raw_vel
                        smoothed_velocities.append(smooth_vel)
                    self.frame_skip_cache['velocities'] = raw_velocities
                    self.frame_skip_cache['smoothed_velocities'] = smoothed_velocities
                else:
                    self.frame_skip_cache['velocities'] = [(0, 0)] * len(detected_centers)
                    self.frame_skip_cache['smoothed_velocities'] = [(0, 0)] * len(detected_centers)
                
                if self.frame_skip_cache['smoothed_centers'] and len(self.frame_skip_cache['smoothed_centers']) == len(detected_centers):
                    smoothed_centers = []
                    center_alpha = 0.6
                    for curr, prev_smooth in zip(detected_centers, self.frame_skip_cache['smoothed_centers']):
                        smooth_x = int(center_alpha * curr[0] + (1 - center_alpha) * prev_smooth[0])
                        smooth_y = int(center_alpha * curr[1] + (1 - center_alpha) * prev_smooth[1])
                        smoothed_centers.append((smooth_x, smooth_y))
                    self.frame_skip_cache['smoothed_centers'] = smoothed_centers
                else:
                    self.frame_skip_cache['smoothed_centers'] = detected_centers
                
                centers = self.frame_skip_cache['smoothed_centers']
                self.frame_skip_cache['last_centers'] = detected_centers
                self.frame_skip_cache['last_sizes'] = sizes_for_interp
                
                centers_x = np.array([c[0] for c in centers], dtype=np.int32)
                centers_y = np.array([c[1] for c in centers], dtype=np.int32)
                x0_arr = np.clip(centers_x - half_sizes, 0, out_w)
                y0_arr = np.clip(centers_y - half_sizes, 0, out_h)
                x1_arr = np.clip(centers_x + half_sizes, 0, out_w)
                y1_arr = np.clip(centers_y + half_sizes, 0, out_h)

                #*****Affectation txt a chaque id
                for i in range(num_kps):
                    if i not in self.blob_names:
                        # Si le nom existe deja pour cet ID, on le garde
                        if i in self.blob_names:
                            pass  # Garder le nom existant
                        else:
                            self.blob_names[i] = random.choice(self.names_list)
                # Nettoyer txt
                existing_ids = set(range(num_kps))
                for blob_id in list(self.blob_names.keys()):
                    if blob_id not in existing_ids:
                        del self.blob_names[blob_id]
            else:
                centers = []
                sizes_for_interp = []
                num_kps = 0
        else:
            num_kps = len(centers) if centers else 0
            if num_kps > 0:
                centers_x = np.array([c[0] for c in centers], dtype=np.int32)
                centers_y = np.array([c[1] for c in centers], dtype=np.int32)
                sizes_scaled = np.array(sizes_for_interp, dtype=np.int32)
                half_sizes = (sizes_scaled * 0.5).astype(np.int32)
                x0_arr = np.clip(centers_x - half_sizes, 0, out_w)
                y0_arr = np.clip(centers_y - half_sizes, 0, out_h)
                x1_arr = np.clip(centers_x + half_sizes, 0, out_w)
                y1_arr = np.clip(centers_y + half_sizes, 0, out_h)
        
                
        if num_kps > 0:
            if config['show_boxes']:
                for i in range(num_kps):
                    if config['use_brackets']:
                        w = x1_arr[i] - x0_arr[i]
                        h = y1_arr[i] - y0_arr[i]
                        bracket_w = int(w * config['bracket_length'])
                        bracket_h = int(h * config['bracket_length'])
                        cv2.line(out_img, (x0_arr[i], y0_arr[i]), (x0_arr[i] + bracket_w, y0_arr[i]), config['outline_color'], config['blob_thickness'])
                        cv2.line(out_img, (x0_arr[i], y0_arr[i]), (x0_arr[i], y0_arr[i] + bracket_h), config['outline_color'], config['blob_thickness'])
                        cv2.line(out_img, (x1_arr[i], y0_arr[i]), (x1_arr[i] - bracket_w, y0_arr[i]), config['outline_color'], config['blob_thickness'])
                        cv2.line(out_img, (x1_arr[i], y0_arr[i]), (x1_arr[i], y0_arr[i] + bracket_h), config['outline_color'], config['blob_thickness'])
                        cv2.line(out_img, (x0_arr[i], y1_arr[i]), (x0_arr[i] + bracket_w, y1_arr[i]), config['outline_color'], config['blob_thickness'])
                        cv2.line(out_img, (x0_arr[i], y1_arr[i]), (x0_arr[i], y1_arr[i] - bracket_h), config['outline_color'], config['blob_thickness'])
                        cv2.line(out_img, (x1_arr[i], y1_arr[i]), (x1_arr[i] - bracket_w, y1_arr[i]), config['outline_color'], config['blob_thickness'])
                        cv2.line(out_img, (x1_arr[i], y1_arr[i]), (x1_arr[i], y1_arr[i] - bracket_h), config['outline_color'], config['blob_thickness'])
                    else:
                        cv2.rectangle(out_img, (x0_arr[i], y0_arr[i]), (x1_arr[i], y1_arr[i]), config['outline_color'], config['blob_thickness'])
                
                # if config['show_ids'] or config['show_leaders']:
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     for i in range(num_kps):
                #         text_parts = []
                #         if config['show_ids']:
                #             text_parts.append(f"ID {i}")
                #         if text_parts:
                #             text = " ".join(text_parts)
                #             h_box = y1_arr[i] - y0_arr[i]
                #             font_scale = np.clip(h_box / 100, 0.25, 0.4)
                #             (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                #             tx = x0_arr[i]
                #             ty = max(y0_arr[i] - 2, th)
                #             if config['show_leaders']:
                #                 cx = centers_x[i]
                #                 cy = centers_y[i]
                #                 cv2.line(out_img, (cx, cy), (tx, ty + th//2), config['outline_color'], 1, cv2.LINE_4)
                #             cv2.putText(out_img, text, (tx, ty), font, font_scale, white, 1, cv2.LINE_4)
            
            if config['show_center_dot']:
                for i in range(num_kps):
                    cv2.circle(out_img, (centers_x[i], centers_y[i]), config['center_dot_radius'], config['center_dot_color'], -1)

            if config['show_ids'] or config['show_leaders']:
                font = cv2.FONT_HERSHEY_DUPLEX  # Police plus epaisse
                FIXED_SCALE = config['fixed_font_scale']
                FIXED_THICKNESS = config['fixed_font_thickness']
                FIXED_OFFSET = config['fixed_font_offset']
                
                for i in range(num_kps):
                    text_parts = []
                    if config['show_ids']:
                        x_pos = centers_x[i]
                        y_pos = centers_y[i]
                        #**pour random
                        #blob_name = random.choice(self.names_list)
                        #text_parts.append(f"ID:{i} X:{x_pos} Y:{y_pos}")
                        blob_name = self.blob_names.get(i, f"ID{i}")
                        text_parts.append(f"{blob_name} X:{x_pos} Y:{y_pos}")
                    
                    if text_parts:
                        text = " ".join(text_parts)
                        (tw, th), _ = cv2.getTextSize(text, font, FIXED_SCALE, FIXED_THICKNESS)
                        
                        # Centrer le texte au-dessus du rectangle
                        rect_center_x = (x0_arr[i] + x1_arr[i]) // 2
                        tx = rect_center_x - (tw // 2)
                        ty = y0_arr[i] - FIXED_OFFSET
                        
                        # # Eviter de sortir de l'image
                        # tx = max(2, min(tx, out_w - tw - 2))
                        # if ty < th + 2:
                        #     ty = y1_arr[i] + FIXED_OFFSET
                        
                        if config['show_leaders']:
                            cx = centers_x[i]
                            cy = centers_y[i]
                            line_end_x = tx + (tw // 2)
                            line_end_y = ty - (th // 2)
                            cv2.line(out_img, (cx, cy), (line_end_x, line_end_y), config['outline_color'], 1, cv2.LINE_4)
                        
                        cv2.putText(out_img, text, (tx, ty), font, FIXED_SCALE, white, FIXED_THICKNESS, cv2.LINE_4)
                        
            if config['draw_connections'] and num_kps > 1:
                avg_size = np.mean(y1_arr - y0_arr)
                connection_distance = avg_size * 3
                connection_thickness = max(1, int(config['blob_thickness'] * 0.5))
                for i in range(num_kps):
                    for j in range(i + 1, num_kps):
                        dx = centers_x[j] - centers_x[i]
                        dy = centers_y[j] - centers_y[i]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist <= connection_distance:
                            pt1 = (centers_x[i], centers_y[i])
                            pt2 = (centers_x[j], centers_y[j])
                            if config['use_dotted']:
                                self.draw_dotted_line(out_img, pt1, pt2, config['outline_color'], connection_thickness)
                            else:
                                cv2.line(out_img, pt1, pt2, config['outline_color'], connection_thickness, cv2.LINE_4)
            
            if config['show_metrics']:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                line_height = 20
                padding = 10
                panel_x = padding
                y_pos = padding + line_height
                cv2.putText(out_img, "TRACKING DATA", (panel_x, y_pos), font, font_scale, config['outline_color'], 1, cv2.LINE_4)
                y_pos += int(line_height * 1.5)
                for i in range(num_kps):
                    x_norm = centers_x[i] / out_w
                    y_norm = centers_y[i] / out_h
                    size = y1_arr[i] - y0_arr[i]
                    speed = 0.0
                    if i < len(self.frame_skip_cache.get('smoothed_velocities', [])):
                        vel = self.frame_skip_cache['smoothed_velocities'][i]
                        speed = np.sqrt(vel[0]**2 + vel[1]**2)
                    data_text = f"ID:{i} X:{x_norm:.2f} Y:{y_norm:.2f} SPD:{speed:.1f} SZ:{size}"
                    cv2.putText(out_img, data_text, (panel_x, y_pos), font, font_scale * 0.9, white, 1, cv2.LINE_4)
                    y_pos += line_height
        
        if config['show_grid']:
            grid_col = tuple(int(c * 0.3) for c in config['outline_color'])
            x = 0
            while x < out_w:
                cv2.line(out_img, (int(x), 0), (int(x), out_h), grid_col, 1, cv2.LINE_4)
                x += config['grid_spacing']
            y = 0
            while y < out_h:
                cv2.line(out_img, (0, int(y)), (out_w, int(y)), grid_col, 1, cv2.LINE_4)
                y += config['grid_spacing']
        
        if config['draw_trails'] and centers and len(centers) >= 2:
            trail_pts = centers
            if len(trail_pts) >= 4:
                trail_pts = self.interpolate_catmull_rom_cached(trail_pts, resolution=config['line_smoothness'])
            if len(trail_pts) >= 2:
                trail_array = np.array(trail_pts, dtype=np.float32)
                diffs = trail_array[1:] - trail_array[:-1]
                segment_lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
                full_length = np.sum(segment_lengths)
                visible_length = full_length * np.clip(config['max_line_length'], 0.0, 1.0)
                if visible_length < full_length:
                    cumsum_rev = np.cumsum(segment_lengths[::-1])
                    cutoff_idx = np.searchsorted(cumsum_rev, visible_length)
                    if cutoff_idx < len(trail_pts) - 1:
                        start_idx = max(0, len(trail_pts) - cutoff_idx - 2)
                        trimmed = trail_pts[start_idx:]
                    else:
                        trimmed = trail_pts
                else:
                    trimmed = trail_pts
                if len(trimmed) >= 2:
                    if config['use_dotted']:
                        for i in range(len(trimmed) - 1):
                            pt1 = (int(trimmed[i][0]), int(trimmed[i][1]))
                            pt2 = (int(trimmed[i+1][0]), int(trimmed[i+1][1]))
                            self.draw_dotted_line(out_img, pt1, pt2, config['trail_color'], 1)
                    else:
                        pts = np.array(trimmed, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(out_img, [pts], False, config['trail_color'], config['trail_thickness'], cv2.LINE_4)
        
        return out_img


def main():
    parser = argparse.ArgumentParser(description='Blob Tracker - Standalone Application')
    parser.add_argument('--input', type=str, default='0', help='Input video file or camera index (default: 0)')
    parser.add_argument('--output', type=str, help='Output video file (optional)')
    parser.add_argument('--threshold', type=int, default=127, help='Threshold value (default: 127)')
    parser.add_argument('--min-area', type=float, default=10, help='Minimum blob area (default: 10)')
    parser.add_argument('--max-area', type=float, default=1000, help='Maximum blob area (default: 1000)')
    parser.add_argument('--max-blobs', type=int, default=100, help='Maximum number of blobs (default: 100)')
    
    args = parser.parse_args()
    
    tracker = BlobTracker()
    
    config = {
        'threshold_mode': 'manual',
        'threshold_value': args.threshold,
        'invert_threshold': True,
        'min_area': args.min_area,
        'max_area': args.max_area,
        'max_blobs': args.max_blobs,
        'resolution_scale': 0.25, #ne pas toucher a ca pour 4k
        'enable_skip': True,
        'frame_skip_interval': 2,
        'motion_smoothing': 0.1,
        'size_smoothing': 0.1,
        'outline_color': (255, 255, 255),
        'trail_color': (255, 255, 255),
        'trail_thickness': 2,
        'blob_thickness': 3,
        'draw_connections': True,
        'draw_trails': False,
        'line_smoothness': 8,
        'max_line_length': 1.0,
        'show_ids': True,
        'show_leaders': False,
        'show_metrics': False,
        'show_grid': False,
        'grid_spacing': 50.0,
        'use_brackets': False,
        'bracket_length': 0.3,
        'use_dotted': False,
        'show_boxes': True,
        'show_center_dot': False,
        'center_dot_radius': 4,
        'center_dot_color': (255, 0, 255),
        'fixed_font_scale': 1.2,        # Taille fixe pour la 4K
        'fixed_font_thickness': 2,      # Epaisseur du texte
        'fixed_font_offset': 15,        # Distance au-dessus du rectangle

    }
    
    # 1. Ouvrir la capture EN PREMIER
    try:
        source = int(args.input)
    except ValueError:
        source = args.input
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return
    
    # 2. Lire les dims réelles APRES ouverture
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 3. Fenêtre initiale respectant le ratio natif — redimensionnable à la souris
    INITIAL_W = 1280
    INITIAL_H = int(INITIAL_W * cap_h / cap_w)
    
    # namedWindow et resizeWindow UNE SEULE FOIS, JAMAIS dans la boucle
    cv2.namedWindow('Blob Tracker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Blob Tracker', INITIAL_W, INITIAL_H)
    
    # 4. Writer en résolution 4K native
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (cap_w, cap_h))
    
    print(f"Source 4K : {cap_w}x{cap_h}  |  Preview initial : {INITIAL_W}x{INITIAL_H}  |  FPS : {fps}")
    print("Fenetre redimensionnable a la souris")
    print("Press 'q' to quit")
    print("Press 't' to toggle trails")
    print("Press 'c' to toggle connections")
    print("Press 'b' to toggle brackets")
    print("Press 'm' to toggle metrics")
    print("Press 'g' to toggle grid")
    print("Press 'd' to toggle dotted lines")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tracker travaille en 4K natif — coords et annotations en 4K
        output_4k = tracker.process_frame(frame, config)

        # Write en 4K natif
        if writer:
            writer.write(output_4k)
        
        # imshow reçoit le 4K brut — WINDOW_NORMAL scale pour remplir la fenêtre
        # l'utilisateur peut redimensionner la fenêtre librement à la souris
        cv2.imshow('Blob Tracker', output_4k)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
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
            print(f"Dotted Lines: {'ON' if config['use_dotted'] else 'OFF'}")
        elif key == ord('x'):
            config['show_boxes'] = not config['show_boxes']
            print(f"Boxes: {'ON' if config['show_boxes'] else 'OFF'}")
        elif key == ord('p'):  # 'p' pour point central
            config['show_center_dot'] = not config['show_center_dot']
            print(f"Center Dot: {'ON' if config['show_center_dot'] else 'OFF'}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()