import argparse
from email.policy import default
import cv2
import tifffile
import re
from pathlib import Path
from .processor import get_y_boundaries
from .processor import create_roi_microscopy_movies


def add_arguments(parser):
    parser.add_argument("-a", "--address", required=True, help="Experiment root folder")
    parser.add_argument("--top-buffer", "-buff1", type=int, default=40, help="Top crop buffer")
    parser.add_argument("--bottom-buffer", "-buff2", type=int, default=40, help="Bottom crop buffer")
    parser.add_argument("-fps", type=int, default=15, help="Playback speed")
    parser.add_argument("--minutes-per-frame", "-mpf", type=int, default=5, help="Minutes per frame")
    parser.add_argument(
        "--no-roi",
        action="store_false",
        dest="use_roi",
        help="Detect crop boundaries from image content instead of roi_boxes.pkl.",
    )
    parser.add_argument(
        "--n-series",
        "-n_s",
        type=int,
        default=30,
        help="Number of series/FOVs per channel",
    )
    parser.add_argument(
        "--input_labeling",
        "-i_l",
        action = "store_false",
        dest="input_labeling",
        help = "Include input labeling for single chambers"
    )
    parser.set_defaults(func=run)
    return parser


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "moma-movie-maker",
        aliases=["movie-maker"],
        help="Create annotated movies from mother-machine microscopy TIFF frames.",
        description="Create annotated movies from mother-machine microscopy TIFF frames.",
    )
    return add_arguments(parser)


def run(args):
    root = Path(args.address)
    movie_dir = root / "movies"
    movie_dir.mkdir(exist_ok=True)

    if args.use_roi:
        create_roi_microscopy_movies(
            root,
            n_series=args.n_series,
            t_buffer=args.top_buffer,
            l_buffer=args.bottom_buffer,
            fps=args.fps,
            interval_mins=args.minutes_per_frame,
            input_labeling = args.input_labeling
        )
    
    else:
        pos_folders = sorted([f for f in root.iterdir() if f.is_dir() and f.name.startswith("pos")],
                            key=lambda x: int(re.findall(r'\d+', x.name)[0] or 0))

        for pos in pos_folders:
            print(f"Processing {pos.name}...")
            all_files = sorted(list(pos.glob("*.tif")))
            if not all_files: continue
            
            channels = sorted(list(set(re.findall(r'chan\d+', f.name)[0] for f in all_files)))
            ref_files = sorted([f for f in all_files if channels[0] in f.name])
            
            y_top, y_bot = get_y_boundaries(
                tifffile.imread(str(ref_files[0])),
                args.top_buffer,
                args.bottom_buffer,
            )
            
            for chan in channels:
                chan_files = sorted([f for f in all_files if chan in f.name])
                first_img = tifffile.imread(str(chan_files[0]))
                h, w = (y_bot - y_top), first_img.shape[1]
                
                output_path = movie_dir / f"{pos.name}_{chan}.mp4"
                video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))

                for idx, f_path in enumerate(chan_files):
                    img_16 = tifffile.imread(str(f_path))
                    crop_8 = cv2.normalize(img_16[y_top:y_bot, :], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                    frame = cv2.cvtColor(crop_8, cv2.COLOR_GRAY2BGR)
                    
                    t = idx * args.minutes_per_frame
                    cv2.putText(frame, f"{t//60:02d}:{t%60:02d}", (20, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    video.write(frame)
                video.release()

    return 0


def main():
    parser = argparse.ArgumentParser(description="MOMA Movie Maker CLI")
    add_arguments(parser)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
