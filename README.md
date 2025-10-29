# Kidney Abnormality Segmentation model

This repository contains the code to setup the pipeline to run the Kidney Abnormality Segmentation model on CT scans. When you use this code base and the corresponding model weights, please cite our work: [kidney abnormality segmentation paper](https://arxiv.org/abs/2505.07573).

The pipeline uses nnUNet [[1]](#1),[[2]](#2) and TotalSegmentator [[3]](#3), please also cite these works. 

## Issues
Please feel free to raise any issues you encounter [here](https://github.com/DIAGNijmegen/oncology-kidney-abnormality-segmentation/issues).

## Grand-challenge algorithm
A ready to use algorithm for research purposes is also available on grand-challenge.org: [algorithm](https://grand-challenge.org/algorithms/kidney-abnormality-segmentation/).

## Setup
The repository contains a [Dockerfile]() that can be used to create the environment. Alternatively, you can use conda or a venv with at least python 3.11 and the requirements (see [requirements]()).

The package can be installed from source: 

```bash
git clone https://github.com/DIAGNijmegen/oncology-kidney-abnormality-segmentation.git
cd oncology-kidney-abnormality-segmentation
pip install -e .
```

TotalSegmentator will install the needed models into your setup. Before running the main script, you need to set the following environment variables:

```bash
export nnUNet_preprocessed='/path/to/.totalsegmentator/nnunet/results'
export nnUNet_raw='/path/to/.totalsegmentator/nnunet/results'
export nnUNet_results='/path/to/.totalsegmentator/nnunet/results'
```

## Running the algorithm
When using the container build using the Dockerfile you can mount the input path containing the CT scans to "/input", the output path to "/output", and the folder containing downloaded model weights to "/opt/ml/model". 

When using a virtual or conda environment, you can either use the command line arguments to set the folder paths or set them as environment variables. 

### To start the script
You can kick-off the script with:

```bash
python main.py --use-cropping (optional) --input-path (optional) --output-path(optional) --model-path (optional)
```

## Pipeline:
The algorithm does the following steps for each CT scan (in .mha format) in the input folder:
1. read ct scan (.mha) using SimpleITK
2. if flag --use-cropping is set: use TotalSegmentator to find bladder and lungs, if found, crop around that in all three directions. 
3. use trained nnUNet to segment the kidneys and kidney abnormalities if present
4. postprocess the kidney abnormality masks by removing small components (<3mm) and only keeping components attached to a kidney region, except when the abnormality component is larger than 100,000 mm^3.

We made the cropping using TotalSegmentator optional. We sporadically noticed that for patients with a larger BMI, the cropping using the lungs as reference point resulted in removing parts of the kidney from the ROI. Performance gains using the cropping were minimal to begin with and therefore not using this step will not result in any performance loss.

## References
<a id="1" href="https://www.nature.com/articles/s41592-020-01008-z">[1]</a> 
Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen and Klaus H. Maier-Hein. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation". Nature Methods 18.2 (2021): 203-211.

<a id="2" href="https://arxiv.org/pdf/2404.09556.pdf">[2]</a>
Fabian Isensee, Tassilo Wald, Constantin Ulrich, Michael Baumgartner, Saikat Roy, Klaus H. Maier-Hein, Paul F. Jaeger. "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation". Pre-print: arXiv:2404.09556v2 (2024).

<a id="3" href="https://pubs.rsna.org/doi/10.1148/ryai.230024">[3]</a>
Jakob Wasserthal, Hanns-Christian Breit, Manfred T. Meyer, Maurice Pradella, Daniel Hinck, Alexander W. Sauter, Tobias Heye, Daniel T. Boll, Joshy Cyriac, Shan Yang, Michael Bach, Martin Segeroth. "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images". Radiology: Artificial Intelligence 5:5 (2023): e230024.

## Managed By
Diagnostic Image Analysis Group,
Radboud University Medical Center,
Nijmegen, The Netherlands

## Contact Information
- Sarah de Boer: Sarah.deBoer@radboudumc.nl
- Alessa Hering: Alessa.Hering@radboudumc.nl