# BHOOmiKAVACH AI — System Limitations & Failure Modes

## 1. Technical & Algorithmic Limitations
1. **Discrete Hazard Tiers:** The model categorizes ground instability into four discrete tiers (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`). Rock mass failure is physically continuous; transitional states require conservative engineering oversight.
2. **Sensor Precision & Noise:** Geotechnical instruments (tell-tales, bore extensometers, convergence pegs) are subject to blasting shocks, anchor slippage, and human recording errors. High input noise degrades classifier confidence.
3. **Synthetic Operational Training:** Because operational non-fatal shift records are not publicly published across all Indian mining companies, Dataset B was generated using CMRI-ISM empirical distributions. While physically grounded, site-specific calibration is mandatory prior to live colliery deployment.

## 2. Geological & Geotechnical Limitations
1. **Hidden Discontinuities:** Slip planes, fossilized kettle bottoms, and sandstone channel rolls can occur unexpectedly without precursor convergence. Sounding rods and bore endoscopes remain indispensable.
2. **Dynamic Seismic Events:** The platform models quasistatic strata delamination and blasting vibrations. It is not designed to predict sudden deep rockbursts or tectonic bump phenomena.

## 3. Human & Operational Constraints
1. **No Automation of Evacuation:** The system acts strictly as an advisory decision-support tool. It cannot initiate automated acoustic alarms or physically seal roadways.
2. **Primacy of Competent Supervision:** Statutory inspection by certified personnel under Regulation 129 of CMR 2017 cannot be replaced by any software system.
