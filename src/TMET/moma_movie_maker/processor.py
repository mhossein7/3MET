import cv2
import numpy as np
import tifffile
from pathlib import Path
import pickle
import json

def get_y_boundaries(img_16, buff1, buff2):
    img_8 = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    height = img_8.shape[0]

    # Robust Y_TOP Detection (+ signs)
    upper_half = img_8[:height//2, :]
    edges = cv2.Canny(upper_half, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    
    plus_bottom_y = 0
    for i in range(1, num_labels):
        _, y, w, h, area = stats[i]
        if 0.6 < (w/float(h)) < 1.4 and 400 < area < 5000:
            plus_bottom_y = max(plus_bottom_y, y + h)

    if plus_bottom_y == 0:
        plus_bottom_y = int(height * 0.1)

    # Intensity Profile Y_BOT Detection (Dark Area)
    row_means = np.mean(img_8, axis=1)
    row_means_smooth = np.convolve(row_means, np.ones(15)/15, mode='same')
    black_thresh = np.min(row_means_smooth) + (np.max(row_means_smooth) - np.min(row_means_smooth)) * 0.1
    
    black_top_y = height
    for y in range(height - 1, plus_bottom_y, -1):
        if row_means_smooth[y] > black_thresh:
            black_top_y = y
            break
            
    return max(0, plus_bottom_y + buff1), min(height, black_top_y - buff2)


def create_roi_microscopy_movies(root_folder, n_series = 30,t_buffer=50,l_buffer=150, fps=10,interval_mins = 5,input_labeling = True):
    root = Path(root_folder)
    
    # Create the 'movies' directory in the root folder
    output_root = root / 'movies'
    output_root.mkdir(exist_ok=True)
    
    # 1. Load ROI coordinates
    roi_path = root / 'roi_boxes.pkl'
    if not roi_path.exists():
        print(f"Error: {roi_path} not found.")
        return

    with open(roi_path, 'rb') as f:
        # Assuming roi_data is a list of dicts corresponding to the sorted pos folders
        roi_data = pickle.load(f)

    # 2. Get all position folders (sorted alphabetically to match ROI order)
    pos_folders = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith('pos')])
    
    n_channel = int(len(pos_folders)/n_series)
    if n_channel != len(roi_data):
        print(f"Warning: Number of positions ({len(pos_folders)}) does not match ROI entries ({len(roi_data)}).")
    coordinates = [[[]for j in range(n_series)] for i in range(n_channel)]
    for i in range(n_channel):
        for j in range(n_series):
            coordinates[i][j] = [min(roi_data[i][j][0]['ytl'],roi_data[i][j][-1]['ytl']),max(roi_data[i][j][0]['ybr'],roi_data[i][j][-1]['ybr'])]
            
    if input_labeling:
        cells_stims_path = root / 'cells_stims.npy'
        if not cells_stims_path.exists():
            print(f'Error: {cells_stims_path} not found.')
            return
        cell_stims = get_group_data(root)
        p = 0
        chamber_stims = [[[]for _ in range(n_series)] for _ in range(n_channel)]

        for i in range(n_channel):
            for j in range(n_series):
                k = len(roi_data[i][j])
                chamber_stims[i][j] = cell_stims[p:p+k]
                p += k
            
    for idx, pos_dir in enumerate(pos_folders):
        num = int(''.join(c for c in pos_dir.name if c.isdigit()))
        # Get ROI for this position
        current_channel = int((num-1)/n_series)
        current_fov = int((num-1)%n_series)
        y_top , y_bottom = coordinates[current_channel][current_fov][0],coordinates[current_channel][current_fov][1]
        

        # 3. Identify all channels in this position
        # Files: chan{c}_frame{t}.tif
        all_tifs = list(pos_dir.glob('chan*.tif'))
        
        # Extract unique channel names (e.g., 'chan1', 'chan2')
        channels = sorted(list(set(f.name.split('_')[0] for f in all_tifs)))

        for c_i, chan in enumerate(channels):
            # Find and sort frames for this specific channel numerically by frame number
            chan_files = sorted(
                [f for f in all_tifs if f.name.startswith(chan + '_')],
                key=lambda x: int(x.stem.split('frame')[-1])
            )

            if not chan_files:
                continue




            # --- PASS 1: Find Global Min/Max for this Channel/Position ---
            print(f"  - Calculating global scale for {chan}...")
            global_min = float('inf')
            global_max = float('-inf')

            for frame_path in chan_files:
                img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    f_min, f_max = np.min(img), np.max(img)
                    if f_min < global_min: global_min = f_min
                    if f_max > global_max: global_max = f_max

            # Prevent division by zero if image is blank
            if global_max == global_min:
                global_max += 1
            
            video_writer = None
            # Save file inside the 'movies' folder
            video_filename = f"{pos_dir.name}_{chan}.mp4"
            output_path = output_root / video_filename

            for f_i , frame_path in enumerate(chan_files):
                # Read 16-bit image
                img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                # --- Normalization [0,1] and conversion to 8-bit ---
                img_float = img.astype(np.float32)
                
                img_8bit = ((img_float - global_min) / (global_max - global_min) * 255).astype(np.uint8) #global normalization
                
                
                # --- Apply ROI Crop with Buffer ---
                h, w = img_8bit.shape[:2]
                y_start = max(0, y_top - t_buffer)
                y_end = min(h, y_bottom + l_buffer)

                # Crop y, keep full x
                cropped = img_8bit[y_start:y_end, :]

                # Prepare for VideoWriter (expects BGR)
                if len(cropped.shape) == 2:
                    display_img = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
                else:
                    display_img = cropped
                
                max_y = display_img.shape[0]    
                    
                # --- 3. Add Timestamp ---
                # Calculate time: frame 1 = 0 mins, frame 2 = 5 mins, etc.
                frame_num = int(frame_path.stem.split('frame')[-1])
                total_minutes = (frame_num - 1) * interval_mins
                hours = total_minutes // 60
                minutes = total_minutes % 60
                time_str = f"{hours:02d}h {minutes:02d}m"

                # Text settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                position = (20, 40) # x, y coordinates
                
                # Draw black shadow for readability
                cv2.putText(display_img, time_str, (position[0]+2, position[1]+2), 
                            font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                # Draw white main text
                cv2.putText(display_img, time_str, position, 
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                # Add chamber number
                if input_labeling:
                    current_rois = roi_data[current_channel][current_fov]
                    for k in range(len(current_rois)):
                        x_tl = current_rois[k]['xtl']
                        x_br = current_rois[k]['xbr']
                        y_tl = current_rois[k]['ytl']
                        x_center = int((x_tl+x_br)/2)
                        circle_center = (x_center,max_y-40)
                        circle_radius=10
                        stim_value = _stim_value_for_frame(
                            chamber_stims[current_channel][current_fov][k],
                            frame_num,
                        )
                        circle_color = (0,255,0) if stim_value == 1 else (0,0,255)
                        # Draw a black outline (thickness=2) so circle is visible on bright backgrounds
                        cv2.circle(display_img, circle_center, circle_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
                        # Draw the colored filled circle (thickness=-1 means filled)
                        cv2.circle(display_img, circle_center, circle_radius, circle_color, -1, cv2.LINE_AA)
                
                
                
                
                
                # Initialize VideoWriter on first frame
                if video_writer is None:
                    height, width = display_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                video_writer.write(display_img)

            if video_writer:
                video_writer.release()
                
                         
def get_group_data(root:Path):
    """
    Return one stimulation sequence per cell/chamber.

    cells_stims.npy is treated as the source of truth. group_config.json is read
    to validate compatibility with the OLP-style format:
    {"Group 1": [1, 0, ...], "Group 2": [1, 1, ...]}.
    """
    stims = np.load(root / 'cells_stims.npy')
    stims = np.asarray(stims)
    if stims.ndim != 2:
        raise ValueError(f"Expected cells_stims.npy to be 2D, got shape {stims.shape}.")

    group_config_path = root / 'group_config.json'
    if group_config_path.exists():
        with open(group_config_path, 'r') as f:
            config = json.load(f)
        _validate_group_config_sequences(config, stims.shape[1])

    return stims


def _validate_group_config_sequences(config, n_timepoints):
    if not isinstance(config, dict):
        raise ValueError("group_config.json must contain a dictionary.")

    for group_name, group_value in config.items():
        arr = np.asarray(group_value)
        if arr.ndim not in (1, 2):
            raise ValueError(f"{group_name} has invalid stimulation shape {arr.shape}.")
        if arr.shape[-1] != n_timepoints:
            raise ValueError(
                f"{group_name} has {arr.shape[-1]} stimulation timepoints, "
                f"but cells_stims.npy has {n_timepoints}."
            )


def _stim_value_for_frame(stim_sequence, frame_num):
    stim = np.asarray(stim_sequence).ravel()
    if stim.size == 0:
        return 0

    idx = max(0, min(frame_num - 1, stim.size - 1))
    return int(stim[idx])
    
