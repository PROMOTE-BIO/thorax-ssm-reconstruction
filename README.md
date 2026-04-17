📌 Title
# Thorax Reconstruction from Sparse Inputs using Statistical Shape Models
📖 Description
This repository provides the implementation of a reconstruction framework for estimating subject-specific thorax geometries from sparse input data.

The method takes as input:
- Skin landmark coordinates
- Subject-specific variables (age, sex, height, and weight)

and outputs a reconstructed 3D thorax surface based on a pre-trained statistical shape model (SSM).

This repository focuses on the **application stage** of the method, enabling researchers to reconstruct thorax geometries without requiring medical imaging data.
📄 Associated Publication
This work is associated with the following publication:

[Full citation here]

If you use this code, please cite the above publication.
⚠️ Scope of This Repository
This repository includes:
- Reconstruction algorithms
- Pre-trained statistical shape models (if permitted)
- Pre-trained regression models (if permitted)
- Example scripts for running the reconstruction pipeline

This repository does NOT include:
- Code for statistical shape model (SSM) training
- Code for regression model training
- Original CT imaging data

The focus is on enabling the use and reproducibility of the reconstruction methodology.
⚙️ Installation
### Requirements
- Python 3.x
- numpy
- scipy
- open3d
- scikit-learn

Install dependencies:
pip install -r requirements.txt
🚀 Usage
### Running the reconstruction

python scripts/run_reconstruction.py

### Example

python scripts/example_usage.py
📥 Input Data
The reconstruction requires:

1. Skin landmarks  
   - 3D coordinates of predefined anatomical landmarks

2. Subject-specific variables  
   - Age  
   - Sex  
   - Height  
   - Weight  

Input format examples are provided in the `data/example/` directory.
📤 Output
The pipeline outputs:
- Reconstructed 3D thorax mesh
- Estimated bone landmark positions (optional)

Output files are saved in standard formats (e.g., .ply, .obj).
🧠 Method Overview
The reconstruction pipeline consists of:

1. Mapping skin landmarks to bone landmarks using pre-trained regression models
2. Estimating shape parameters of the statistical shape model
3. Reconstructing the thorax surface via optimization

The statistical shape model and regression models are pre-trained and provided for direct use.
⚠️ Notes and Limitations
- Reconstruction accuracy depends on the quality and consistency of the input landmarks
- The method is designed for sparse input scenarios and may not capture fine anatomical details
- The current implementation prioritizes reconstruction accuracy over computational efficiency
🔒 Data Availability
Due to ethical and data protection constraints, the original CT datasets used to develop the models cannot be shared.

Only derived models and reconstruction tools are provided.
📜 License
MIT License
📚 Citation
If you use this work, please cite:

@article{yourname2026,
  title={...},
  author={...},
  journal={...},
  year={2026}
}
