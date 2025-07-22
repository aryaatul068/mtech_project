import deepdish as dd
import os.path as osp
import os
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

def process_data(args, csv_file, output_name):
    data_dir = os.path.join(args.root_path, 'ABIDE_pcp/cpac/filt_noglobal/raw')
    timeseires = os.path.join(args.root_path, 'ABIDE_pcp/cpac/filt_noglobal/')

    # Load the appropriate metadata file (ASD or TD)
    meta_file = os.path.join(args.root_path, 'ABIDE_pcp', csv_file)
    meta_data = pd.read_csv(meta_file, header=0)

    # Map subject IDs to site
    id2site = meta_data[["SUB_ID", "SITE_ID"]].set_index("SUB_ID").to_dict()["SITE_ID"]

    times, labels, pcorrs, corrs, site_list = [], [], [], [], []

    for f in os.listdir(data_dir):
        if osp.isfile(osp.join(data_dir, f)):
            fname = f.split('.')[0]

            if int(fname) not in id2site:  # Ignore subjects not in the current CSV
                continue

            site = id2site[int(fname)]

            files = os.listdir(osp.join(timeseires, fname))
            file = list(filter(lambda x: x.endswith("1D"), files))

            if not file:
                continue  # Skip if no matching file found

            time = np.loadtxt(osp.join(timeseires, fname, file[0]), skiprows=0).T

            if time.shape[1] < 100:
                continue

            temp = dd.io.load(osp.join(data_dir, f))
            pcorr, att, label = temp['pcorr'][()], temp['corr'][()], temp['label'][()]

            pcorr[pcorr == float('inf')] = 0
            att[att == float('inf')] = 0

            times.append(time[:, :100])
            labels.append(label[0])
            corrs.append(att)
            pcorrs.append(pcorr)
            site_list.append(site)

    # Save processed data as .npy
    output_path = Path(args.root_path) / f'ABIDE_pcp/{output_name}.npy'
    np.save(output_path, {'timeseries': np.array(times), "label": np.array(labels), 
                          "corr": np.array(corrs), "pcorr": np.array(pcorrs), 'site': np.array(site_list)})
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ASD/TD dataset separately')
    parser.add_argument('--root_path', default="", type=str, help='Root directory containing dataset')
    args = parser.parse_args()

    # Process ASD subjects
    process_data(args, "ASD_subjects.csv", "abide_asd")

    # Process TD subjects
    process_data(args, "TD_subjects.csv", "abide_td")
