import numpy as np
import cv2
import argparse
import random
import os
import re
from collections import deque
import glob

class BlobTracker:
    def __init__(self, names_list=None):
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

        # Use provided names list or default
        self.names_list = names_list if names_list else ['path', 'pilote']
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
    
    def process_frame(self, frame, config, export_alpha=False, mask_frames=None, frame_index=0):
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

        # Apply animated mask — masked zones excluded from detection
        if mask_frames and should_detect:
            mask = mask_frames[frame_index % len(mask_frames)]
            if mask.shape[:2] != (detect_h, detect_w):
                mask = cv2.resize(mask, (detect_w, detect_h), interpolation=cv2.INTER_NEAREST)
            thresh_detect[mask > 0] = 0
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
        
        scale_x = (out_w / detect_w)
        scale_y = (out_h / detect_h)
        scale_avg = (scale_x + scale_y) * 0.5
        white = (255, 255, 255)
        
        # Variables pour dessin
        centers_x = np.array([])
        centers_y = np.array([])
        x0_arr = np.array([])
        y0_arr = np.array([])
        x1_arr = np.array([])
        y1_arr = np.array([])
        num_kps = 0
        
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

                # Affectation txt a chaque id
                for i in range(num_kps):
                    if i not in self.blob_names:
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
        
        # Dessin des annotations sur l'image principale (BGR)
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
            
            if config['show_center_dot']:
                for i in range(num_kps):
                    r = config['center_dot_radius']
                    if config['center_dot_style'] == 'cross':
                        cross_size = r * 3
                        thickness = max(1, r // 2)
                        cv2.line(out_img, (centers_x[i] - cross_size, centers_y[i]), (centers_x[i] + cross_size, centers_y[i]), config['center_dot_color'], thickness, cv2.LINE_4)
                        cv2.line(out_img, (centers_x[i], centers_y[i] - cross_size), (centers_x[i], centers_y[i] + cross_size), config['center_dot_color'], thickness, cv2.LINE_4)
                    else:
                        cv2.circle(out_img, (centers_x[i], centers_y[i]), r, config['center_dot_color'], -1)

            if config['show_ids'] or config['show_leaders']:
                font = cv2.FONT_HERSHEY_DUPLEX
                FIXED_SCALE = config['fixed_font_scale']
                FIXED_THICKNESS = config['fixed_font_thickness']
                FIXED_OFFSET = config['fixed_font_offset']
                
                for i in range(num_kps):
                    text_parts = []
                    if config['show_ids']:
                        x_pos = centers_x[i]
                        y_pos = centers_y[i]
                        blob_name = self.blob_names.get(i, f"ID{i}")
                        text_parts.append(f"{blob_name} X:{x_pos} Y:{y_pos}")
                    
                    if text_parts:
                        text = " ".join(text_parts)
                        (tw, th), _ = cv2.getTextSize(text, font, FIXED_SCALE, FIXED_THICKNESS)
                        
                        rect_center_x = (x0_arr[i] + x1_arr[i]) // 2
                        tx = rect_center_x - (tw // 2)
                        ty = y0_arr[i] - FIXED_OFFSET
                        
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
        
        # Export PNG avec alpha (fond transparent)
        alpha_img = None
        if export_alpha and num_kps > 0:
            alpha_img = np.zeros((out_h, out_w, 4), dtype=np.uint8)
            # Dessiner les memes annotations sur fond transparent
            for i in range(num_kps):
                if config['show_boxes']:
                    cv2.rectangle(alpha_img, (x0_arr[i], y0_arr[i]), (x1_arr[i], y1_arr[i]), config['outline_color'], config['blob_thickness'])
                if config['show_center_dot']:
                    r = config['center_dot_radius']
                    if config['center_dot_style'] == 'cross':
                        cross_size = r * 3
                        thickness = max(1, r // 2)
                        cv2.line(alpha_img, (centers_x[i] - cross_size, centers_y[i]), (centers_x[i] + cross_size, centers_y[i]), config['center_dot_color'], thickness, cv2.LINE_4)
                        cv2.line(alpha_img, (centers_x[i], centers_y[i] - cross_size), (centers_x[i], centers_y[i] + cross_size), config['center_dot_color'], thickness, cv2.LINE_4)
                    else:
                        cv2.circle(alpha_img, (centers_x[i], centers_y[i]), r, config['center_dot_color'], -1)
                if config['show_ids']:
                    name = self.blob_names.get(i, f"ID{i}")
                    text = f"{name} X:{centers_x[i]} Y:{centers_y[i]}"
                    font = cv2.FONT_HERSHEY_DUPLEX
                    fs = config['fixed_font_scale']
                    fth = config['fixed_font_thickness']
                    (tw, th), _ = cv2.getTextSize(text, font, fs, fth)
                    tx = (x0_arr[i] + x1_arr[i]) // 2 - tw // 2
                    ty = max(y0_arr[i] - config['fixed_font_offset'], th + 5)
                    cv2.putText(alpha_img, text, (tx, ty), font, fs, (255,255,255), fth, cv2.LINE_4)
            # Definir le canal alpha: 255 la ou il y a des dessins
            drawn_mask = np.any(alpha_img[:,:,:3] > 0, axis=2)
            alpha_img[:,:,3] = np.where(drawn_mask, 255, 0)
        
        return out_img, alpha_img


def load_config_from_file(config_file='config.txt'):
    """Load configuration from text file"""
    config = {}
    
    if not os.path.exists(config_file):
        print(f"Info: {config_file} not found, using default values")
        return config
    
    print(f"Info: loading config from {config_file}")
    
    pattern = re.compile(r'^\s*([A-Z_]+)\s*=\s*(.+?)\s*$', re.IGNORECASE)
    
    with open(config_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            match = pattern.match(line)
            if not match:
                print(f"Warning: line {line_num} ignored: {line}")
                continue
            
            key = match.group(1).upper()
            value = match.group(2).strip()
            
            # Handle colors with commas
            if re.match(r'^\d+,\d+,\d+$', value):
                config[key] = tuple(int(x) for x in value.split(','))
            elif value.lower() in ('true', 'yes', 'on', '1'):
                config[key] = True
            elif value.lower() in ('false', 'no', 'off', '0'):
                config[key] = False
            elif value.isdigit():
                config[key] = int(value)
            elif re.match(r'^\d+\.\d+$', value):
                config[key] = float(value)
            else:
                config[key] = value
    
    return config


def load_names_from_config(config_file='config.txt'):
    """Load blob names from config file"""
    names = []
    
    if not os.path.exists(config_file):
        return names
    
    pattern = re.compile(r'^\s*BLOB_NAMES\s*=\s*(.+?)\s*$', re.IGNORECASE)
    
    with open(config_file, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                names_str = match.group(1).strip()
                names = [n.strip() for n in names_str.split(',') if n.strip()]
                break
    
    return names


def main():
    parser = argparse.ArgumentParser(description='Blob Tracker - Standalone Application')
    parser.add_argument('--input', type=str, default='0', help='Input video file or camera index (default: 0)')
    parser.add_argument('--output', type=str, help='Output video file (optional)')
    parser.add_argument('--png-output', type=str, help='Output folder for PNG sequence with alpha (optional)')
    parser.add_argument('--threshold', type=int, default=None, help='Threshold value (overrides config)')
    parser.add_argument('--min-area', type=float, default=None, help='Minimum blob area (overrides config)')
    parser.add_argument('--max-area', type=float, default=None, help='Maximum blob area (overrides config)')
    parser.add_argument('--max-blobs', type=int, default=None, help='Maximum number of blobs (overrides config)')
    
    args = parser.parse_args()
    
  # Default configuration
    config = {
        'threshold_mode': 'manual',
        'threshold_value': 127,
        'invert_threshold': True,
        'min_area': 40000,
        'max_area': 85000,
        'max_blobs': 2,
        'resolution_scale': 0.5,
        'enable_skip': True,                    
        'frame_skip_interval': 2,               
        'motion_smoothing': 0.5,                
        'size_smoothing': 0.5,                  
        'outline_color': (255, 255, 255),
        'trail_color': (255, 255, 255),
        'trail_thickness': 2,
        'blob_thickness': 2,
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
        'center_dot_style': 'cross',
        'center_dot_radius': 4,
        'center_dot_color': (255, 255, 255),
        'fixed_font_scale': 1,
        'fixed_font_thickness': 2,
        'fixed_font_offset': 14,
}
    
    # Load config from file
    file_config = load_config_from_file('config.txt')
    for key, value in file_config.items():
        if key in config:
            config[key] = value
        elif key not in ['BLOB_NAMES']:
            print(f"Warning: key '{key}' ignored (not found in config)")
    
    # Load custom names from config
    custom_names = load_names_from_config('config.txt')
    if custom_names:
        print(f"Info: loaded {len(custom_names)} custom names")
        tracker = BlobTracker(names_list=custom_names)
    else:
        print("Info: no custom names, using defaults")
        tracker = BlobTracker()
    
    # Override with CLI arguments
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
        print(f"Error: Could not open video source {args.input}")
        return
    
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = int(cap.get(cv2.CAP_PROP_FPS))
    
    INITIAL_W = 1280
    INITIAL_H = int(INITIAL_W * cap_h / cap_w)
    
    cv2.namedWindow('Blob Tracker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Blob Tracker', INITIAL_W, INITIAL_H)
    
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (cap_w, cap_h))
    
    # Export PNG alpha
    export_alpha = args.png_output is not None
    if export_alpha:
        os.makedirs(args.png_output, exist_ok=True)
        print(f"PNG alpha sequence will be saved to: {args.png_output}/")

    # Load animated mask sequence (optional)
    mask_frames = []
    if args.mask_input and os.path.isdir(args.mask_input):
        mask_files = sorted(glob.glob(os.path.join(args.mask_input, '*.png')))
        for f in mask_files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                mask_frames.append(img[:, :, 3])
        if mask_frames:
            print(f"Mask loaded: {len(mask_frames)} frames from {args.mask_input}")
        else:
            print(f"Warning: no valid RGBA PNG found in {args.mask_input}, running without mask")
    elif args.mask_input:
        print(f"Warning: mask folder not found: {args.mask_input}, running without mask")
    
    print(f"Source: {cap_w}x{cap_h}  |  Preview: {INITIAL_W}x{INITIAL_H}  |  FPS: {fps}")
    print("Window resizable with mouse")
    print("Press 'q' to quit")
    print("Press 't' to toggle trails")
    print("Press 'c' to toggle connections")
    print("Press 'b' to toggle brackets")
    print("Press 'm' to toggle metrics")
    print("Press 'g' to toggle grid")
    print("Press 'd' to toggle dotted lines")
    print("Press 'x' to toggle boxes")
    print("Press 'p' to toggle center dot (cycle: off -> dot -> cross)")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_4k, alpha_4k = tracker.process_frame(
            frame, config,
            export_alpha=export_alpha,
            mask_frames=mask_frames if mask_frames else None,
            frame_index=frame_count
        )
        
        if writer:
            writer.write(output_4k)
        
        if export_alpha and alpha_4k is not None:
            folder_name = os.path.basename(args.png_output.rstrip('/\\'))
            path = os.path.join(args.png_output, f"{folder_name}_{frame_count:06d}.png")
            cv2.imwrite(path, alpha_4k)
            if frame_count % 25 == 0:
                print(f"  Exporting PNG frame {frame_count}", end='\r')
        
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
        elif key == ord('p'):
            # Cycle: off -> dot -> cross -> off
            if config['show_center_dot'] == False:
                config['show_center_dot'] = True
                config['center_dot_style'] = 'dot'
                print("Center Dot: DOT")
            elif config['center_dot_style'] == 'dot':
                config['center_dot_style'] = 'cross'
                print("Center Dot: CROSS")
            else:
                config['show_center_dot'] = False
                print("Center Dot: OFF")
        
        frame_count += 1
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    if export_alpha:
        print(f"\nDone! {frame_count} PNG alpha exported to {args.png_output}/")
    else:
        print("Done!")


if __name__ == '__main__':
    main()