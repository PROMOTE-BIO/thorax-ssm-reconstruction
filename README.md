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
This repository accompanies the manuscript:
"Thorax Shape Reconstruction from Limited CT-digitized Palpable Landmarks using Statistical Shape Modeling"
currently under review.

If you use this code, please cite the above publication.
⚠️ Scope of This Repository
This repository includes:
- Reconstruction algorithms
- Pre-trained statistical shape models of the thorax
- Pre-trained regression models for mapping skin landmarks to bone landmarks
- Example scripts for running the reconstruction pipeline

⚙️ Installation
### Requirements
- Python >=3.10, <3.13 (due to `open3d` compatibility)

For standard use:
```
pip install -r requirements.txt
```
For exact reproducibility:
```
pip install -r requirements-lock.txt
```

```md
- `requirements.txt`: minimal dependencies for running the pipeline  
- `requirements-lock.txt`: exact environment used for development and testing  
```

🚀 Usage

### Quick Start
Clone the repository:
```bash
git clone https://github.com/PROMOTE-BIO/thorax-ssm-reconstruction.git
cd thorax-ssm-reconstruction
```

Install dependencies:
```
pip install -r requirements.txt
```

### Running the reconstruction
Run the reconstruction pipeline:
```
python SSM-thorax-reconstruction.py --filename data/example_subject --method SSM-SL-based
```

📥 Input
--filename: name of the input CSV file (without extension), e.g. example_subject.csv
--method: reconstruction method
   - SSM-SL-based (reconstruction using the SSM with embedded skin landmarks)
   - SSM-BL-based (skin landmark to bone landmark mapping followed by reconstruction using the SSM without embedded skin landmarks)

Optional arguments:
--PCs: number of principal components (default: 8)
--Plot: True to visualize the reconstruction at the end
--compare: name of .stl file (without extension) for comparison (if applicable)

📥 Output
Reconstructed thorax surface saved as .stl in the Results/ folder

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
