import re
import tifffile
from pathlib import Path
import numpy as np
import argparse


def batch_organizer(path,imaged = 'imaged', unimaged = 'unimaged' , constant = '_'):

    def extract_channels(input_folder, output_folder):
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Pattern to match files ending with something like 12.ome.tif
        tif_files = sorted(
            [f for f in input_folder.glob("*.ome.tif")],
            key=lambda f: int(re.search(r'(\d+)\.ome\.tif$', f.name).group(1))
        )

        stack = tifffile.imread(tif_files[0])
        for i in np.arange(1,stack.shape[0]+1):
            image = stack[i-1,:,:,:]
            for c in range(3):
                out_path = output_folder / f"Pos{i}_chan{c+1}_t1.tif"
                tifffile.imwrite(out_path, image[c,:,:])

            print(f"Processed {i}file -> Pos{i}_chan1/2/3_t1.tif")

    # Example usage:
    # extract_channels("/path/to/tif_folder", "/path/to/output_folder")

    extract_channels(path+f'/{imaged}/{constant}1',path+f'/{imaged}/DeLTA_compatible')
    extract_channels(path+f'/{unimaged}/{constant}1',path+f'/{unimaged}/DeLTA_compatible')


def main():
    parser = argparse.ArgumentParser(description="Manual Image Organizer CLI")
    parser.add_argument("-a", "--address", required=True, help="Experiment root folder")
    parser.add_argument("-imaged", type=str, default='imaged', help="name of subdirectory of imaged files")
    parser.add_argument("-unimaged", type=str, default='umimaged', help="name of subdirectory of unimaged files")
    parser.add_argument("-constant", type=str, default='_', help="name of subsubdirectory of tiff files")  
    
    args = parser.parse_args()
    root = Path(args.address)
    imaged = args.imaged
    unimaged = args.unimaged
    constant = args.constant
    batch_organizer(str(root),imaged,unimaged,constant)
    
    
if __name__=='__main__':
    main()
