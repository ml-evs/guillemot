import json
import os
import numpy as np
from pybaselines import Baseline

def patternFromJson(json_file_path, out_dir= "examples/opxrd"): 
    """ Extracts the two_theta and intensities from a json file and saves them as a csv and xy file.
    """
    
    with open(json_file_path, "r") as f:
        data= json.load(f)

    two_theta = np.array(data["two_theta_values"])
    inten= np.array(data["intensities"]) # raw intensities
    baseline_fitter= Baseline(x_data=two_theta)
    baseline, params= baseline_fitter.airpls(inten, lam= 1e5) # getting the baseline
    intensities= inten - baseline  # baseline-corrected intensities

    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}.xy")
    
    topas_xy= np.savetxt(
        out_path,
        np.column_stack((two_theta, intensities)),
        delimiter= " ",
        header= "two_theta_values intensities",
        comments= "",
    )

    return {
        "xy_file_path": out_path,
        "number_of_points": len(two_theta),
        "two_theta_range": [float(two_theta.min()), float(two_theta.max())],
    }


  

