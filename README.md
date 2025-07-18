# MATSim Sensitivity and Calibration Toolkit

This repository contains the full pipeline used to identify and calibrate key parameters in agent-based transport simulations using MATSim from the paper "*Identifying Key and Detrimental Parameters in Agent-Based Transport Models: A Sensitivity-Guided Calibration of MATSim*".

It combines simulation filtering, machine learning, and global sensitivity analysis to isolate the most influential inputs and improve calibration efficiency.

---

## 🚀 Key Features

- ✅ **Simulation filtering** using output-based heuristics and a random forest classifier  
- 🔍 **Parameter space refinement** based on feature importance and partial dependence plots  
- 📈 **Global sensitivity analysis** using:
  - Morris method (Elementary Effects)
  - Sobol indices (via Gaussian Process surrogate models)  
- 🎯 **Calibration** against traffic counts using refined parameter set

---

## 🧰 Project Structure

```
TODO : FIX THIS 
matsim-sensitivity-calibration/
│
├── data/                    # Input/output data and simulation results
├── notebooks/              # Jupyter notebooks for analysis and visualization
├── scripts/                # Python scripts for training, filtering, and calibration
├── config/                 # MATSim config files and parameter ranges
├── models/                 # Trained surrogate models and classifiers
└── README.md               # Project documentation
```

---

## 🧪 Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt` (install via `pip install -r requirements.txt`)
- Java 11+ (for MATSim)
- [MATSim](https://www.matsim.org/) scenario (e.g. Sioux Falls)

---

## 🗘️ Reproducibility

Each step of the pipeline is documented and reproducible:
1. Generate or load simulation samples.
2. Filter invalid configurations.
3. Train surrogate models (Gaussian Process).
4. Compute sensitivity metrics.
5. Calibrate using selected parameter sets.

---

## 📄 Reference

If you use this code in your work, please cite our paper:

> TODO : PUT CITATIOn
---

## 📬 Contact

For questions or collaboration inquiries, please contact:  
**Olivier Bussière** – [olbus4@ulaval.ca](mailto:olbus4@ulaval.ca)
