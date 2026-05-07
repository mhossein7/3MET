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
        group_config_path = root / 'group_config.json'
        if not group_config_path.exists():
            print(f'Error: {group_config_path} not found.')
            return
        group_ids = get_group_data(root)
        p = 0
        grouped_groups = [[[]for _ in range(n_series)] for _ in range(n_channel)]

        for i in range(n_channel):
            for j in range(n_series):
                k = len(roi_data[i][j])
                grouped_groups[i][j] = group_ids[p:p+k]
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
                        text = 'GP1' if grouped_groups[current_channel][current_fov][k]=='group1' else 'GP2'
                        #cv2.putText(display_img,str(text),(x_tl+2,max_y-40+2),font,font_scale,(0,0,0),thickness+1,cv2.LINE_AA)
                        #cv2.putText(display_img,str(text),(x_tl,max_y-40),font,font_scale,(0,255,0) if text == 'GP1' else (0,0,255),thickness,cv2.LINE_AA)
                        circle_center = (x_center,max_y-40)
                        circle_radius=10
                        circle_color = (0,255,0) if text=='GP1' else (0,0,255)
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
    stims = np.load(root / 'cells_stims.npy')
    group_db = np.zeros(stims.shape[0])
    group_db = group_db.astype(str)
    with open(root / 'group_config.json', 'r') as f:
        config = json.load(f)
    
    gp1_r_g = config['group1']['red_to_green']
    gp1_g_r = config['group1']['green_to_red']
    gp2_r_g = config['group2']['red_to_green']
    gp2_g_r = config['group2']['green_to_red']
    
    
    def transition_finder(stim_vec):
        diffs = np.diff(stim_vec)
        ind_r_g = np.array(np.where(diffs==1)) +1 
        ind_g_r = np.array(np.where(diffs == -1)) +1 
        if ind_g_r[0].size == 0: ind_g_r = ['NaN']
        if ind_r_g[0].size == 0: ind_r_g = ['NaN']
        return (np.array(ind_r_g))[0] , (np.array(ind_g_r))[0]

    for i in range(stims.shape[0]):
        r_g , g_r = transition_finder(stims[i])
        if np.all(r_g==gp1_r_g) and np.all(g_r==gp1_g_r): group_db[i] = 'group1'
        elif np.all(r_g==gp2_r_g) and np.all(g_r==gp2_g_r): group_db[i] = 'group2'
        else: group_db[i] = 'Ungrouped'
    
    return group_db
    