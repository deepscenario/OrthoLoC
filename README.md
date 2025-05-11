<div align="center">

  <img src="ortholoc/assets/logo.png" alt="logo" width="200" height="auto" />
  <h1>✈️🌍️: OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata</h1>

<!-- Badges -->
<p>
  <a href="https://github.com/deepscenario/OrthoLoC/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/deepscenario/OrthoLoC" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/deepscenario/OrthoLoC" alt="last update" />
  </a>
  <a href="https://github.com/deepscenario/OrthoLoC/network/members">
    <img src="https://img.shields.io/github/forks/deepscenario/OrthoLoC" alt="forks" />
  </a>
  <a href="https://github.com/deepscenario/OrthoLoC/stargazers">
    <img src="https://img.shields.io/github/stars/deepscenario/OrthoLoC" alt="stars" />
  </a>
  <a href="https://github.com/deepscenario/OrthoLoC/issues/">
    <img src="https://img.shields.io/github/issues/deepscenario/OrthoLoC" alt="open issues" />
  </a>
  <a href="https://github.com/deepscenario/OrthoLoC/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/deepscenario/OrthoLoC.svg" alt="license" />
  </a>
</p>

<h4>
    <a href="">📄 Paper (Under Review)</a>
  <span> · </span>
    <a href="https://deepscenario.github.io/OrthoLoC/">💻 Project Page</a>
  </h4>
</div>

<div align="center">
  <img src="ortholoc/assets/overview.png" alt="overview" />
</div>

<p>
🎯 We present a new paradigm for UAV camera localization and calibration using geospatial data (geodata), specifically orthophotos (DOPs) and digital surface models (DSMs).
This approach is particularly useful for UAVs operating in urban environments, where traditional localization methods may struggle due to the complexity of the surroundings. By leveraging geodata, we can achieve accurate localization and calibration even in challenging conditions.
We propose a large-scale benchmark dataset for UAV visual localization, which includes a diverse set of images and geodata from various environments. This dataset serves as a valuable resource for researchers and practitioners in the field, enabling them to develop and evaluate new localization algorithms.
</p>

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [⭐ About the Project](#About-the-Project)
- [🔍 Dataset Features](#Dataset-Features)
- [📊 Dataset Samples](#Dataset-Samples)
- [🧰 Getting Started](#Getting-Started)
  * [❗ Prerequisites](#Prerequisites)
  * [⚙️ Installation](#Installation)
  * [📥 Download the Dataset](#Download-the-Dataset)
  * [📂 Structure of the Dataset](#Structure-of-the-Dataset)
- [🔧 Usage](#Usage)
- [🏃 Run Scripts](#Run-Scripts)
  * [🔄 Matching Only](#Matching-Only)
  * [📍 Localization and/or Calibration](#Localization-and-Calibration)
  * [📏 Benchmarking](#Benchmarking)
  * [👁️ Visualization of a Sample](#Visualization-of-a-Sample)
  * [🖼️ Visualization of Samples in the Dataset](#Visualization-of-Dataset-Samples)
- [⚠️ License](#License)
- [🙏 Acknowledgements](#Acknowledgements)

<a name="About-the-Project"></a>
## :star2: About the Project

OrthoLoC is a framework for UAV camera localization and calibration using orthographic geodata. The project provides a large-scale benchmark dataset and implementation of algorithms for matching, localization, and calibration of UAV imagery using orthophotos and digital surface models.

<a name="Dataset-Features"></a>
## :mag: Dataset Features

- **📸 Large-scale Dataset:** **16,427** high-resolution UAV images with high-precision camera parameters (intrinsics + extrinsics), sourced from **multiple** geographic environments (Europe + US).
- **🛫 Multi-altitude:** Imagery captured at **varied altitudes** (23m-154m), offering diverse perspectives and coverage.
- **🔄 Multi-viewpoint:** Diverse viewpoints with wide camera angles (tilting: 0°-86.8°), offering challenging and dynamic perspectives.
- **🌍 Multi-geographic Environments:** Coverage of **urban** 🏙️, **rural** 🌾, **highway** 🚗, and **suburban** 🏡 landscapes, with high-res geodata alignment for varied terrain.
- **📍 High-precision Ground-truth:** Paired UAV-geodata images for precise localization and calibration, minimizing retrieval errors and ensuring accuracy.
- **🌐 Geospatial Alignment:** Well-aligned orthographic maps (orthophotos and elevation maps) for precise UAV localization and calibration.
- **🔬 Real-world Applicability:** A foundation for evaluating decoupled UAV localization and calibration, enabling practical, real-world use cases.

<a name="Dataset-Samples"></a>
## :bar_chart: Dataset Samples

<div align="center"> 
  <img src="ortholoc/assets/samples.png" alt="Dataset Samples" />
</div>

<a name="Getting-Started"></a>
## :toolbox: Getting Started

<a name="Prerequisites"></a>
### :bangbang: Prerequisites

- Python 3.10

<a name="Installation"></a>
### :gear: Installation

#### via pip

To install the package, you can use pip to install it directly from GitHub:
```
pip install git+https://github.com/deepscenario/OrthoLoC.git
```
#### via git clone

Clone the project
```
git clone https://github.com/deepscenario/OrthoLoC.git
```
Install the dependencies
```
cd OrthoLoC
pip install -e .
```
<a name="Download-the-Dataset"></a>
### :inbox_tray: Download the Dataset

Our dataset is available here: [OrthoLoC Dataset](https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/)
For simple runs, you can download a very small sample of the dataset (less than 250MB) from [here](https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/dataset_sample/).

<a name="Structure-of-the-Dataset"></a>
### :file_folder: Structure of the Dataset
```
├── train
│   ├── L01_R0000.npz
│   ├──  ...
├── val
│   ├── L01_R0014.npz
│   ├──  ...
├── test_inPlace
│   ├── L01_R0012.npz
│   ├──  ...
├── train
│   ├── L08_R0000.npz
│   ├──  ...
```
Each .npz file contains the following keys: ['sample_id', 'image_query', 'point_map', 'image_dop', 'dsm', 'scale', 'extrinsics', 'intrinsics', 'keypoints', 'vertices', 'faces', 'extrinsics_refined']
- **sample_id**: The ID of the sample as string
- **image_query**: The query image as numpy array of shape (H, W, 3)
- **point_map**: The point map as numpy array of shape (H, W, 3)
- **image_dop**: The DOP image as numpy array of shape (H_geo, W_geo, 3), H_geo = W_geo = 1024
- **dsm**: The DSM image as numpy array of shape (H_geo, W_geo, 3), H_geo = W_geo = 1024
- **scale**: The scale of a single pixel in the DOP and DSM images in meters
- **extrinsics**: The extrinsics (world to cam) of the query image as numpy array of shape (3, 4)
- **intrinsics**: The intrinsics of the query image as numpy array of shape (3, 3)
- **keypoints**: The 3D keypoints of the query image as numpy array of shape (N, 3)
- **vertices**: The 3D vertices of the local mesh as numpy array of shape (M, 3)
- **faces**: The faces of the local mesh as numpy array of shape (L, 3)
- **extrinsics_refined**: The refined extrinsics (world to cam) of the query image as numpy array (overfitted pose to the raster elevations, i.e. we refined the pose by establishing GT matches between the query image and the DOP/DSM images and solving the pose using PnP Ransac. The pose is kinda refined to make sure that the GT pose is eliminating rasterization errors). Note that we don't use this pose for our benchmarking. The purpose was to deliver this data for researchers who want to use the dataset for other purposes (e.g. Training).

<a name="Usage"></a>
## :wrench: Usage

To use the dataset as a PyTorch Dataset, you can do the following:
```
from ortholoc.dataset import OrthoLoC

dataset = OrthoLoC(
    dirpath="./assets/samples",  # path to the dataset
    sample_paths=None,  # path to the samples (if dirpath is not specified), e.g. ["./assets/samples/highway_rural.npz", "./assets/samples/urban_residential_xDOPDSM.npz"]
    start=0.,  # start of the dataset
    end=1.,  # end of the dataset
    mode=0,  # mode 0 for all samples, 1 for samples with same UAV imagery and geodata domain, 2 for samples with DOP domain shift, 3 for samples with DOP and DSM domains shift
    new_size=None,  # new size of the images (useful for training)
    limit_size=None,  # limit size of the images (useful for debugging)
    shuffle=True,  # shuffle the dataset
    scale_query_image=1.0,  # scale of the query image (1.0 keep the original size)
    scale_dop_dsm=1.0,  # scale of the DOP and DSM images
    gt_matching_confidences_decay=1.0,  # decay of the matching confidences (the larger the less confident will be the GT for non-unique points like points on facades)
    covisibility_ratio=1.0,  # ratio of the covisibility (0.0 to 1.0, 0.0 exclusive, the larger the more area in the geodata will be visible for the UAV)
    return_tensor=True  # return the samples while iterating over the dataset as dict of torch tensors
)
```
<a name="Run-Scripts"></a>
## :running: Run Scripts

For each script, consult the help message for more options.
All the weights for matching algorithms will be downloaded automatically.

<a name="Matching-Only"></a>
### :repeat: Matching Only

To run the image matching from a sample of the dataset or from two images, you can do the following:
```
run-matching.py \
    --sample assets/urban.npz --matcher superpoint+lightglue --device cuda --angles 0 90 180 270 --show  
```
<a name="Localization-and-Calibration"></a>
### :round_pushpin: Localization and/or Calibration

To run the localization and/or calibration from a sample of the dataset or from custom data, you can do the following:
```
run-localization \
    --sample assets/urban.npz --matcher Mast3R --device cuda --angles 0 --output_dir ./outputs/ --show 
```
You can use your own images and geodata by specifying the paths to the files directly:
```
run-localization \
    --image assets/urban_residential.jpg --dop assets/urban_residential_DOP.tif --dsm assets/urban_dsm.tif --intrinsics assets/urban_query_intrinsics.json --matcher Mast3R --device cuda --angles 0 --output_dir ./outputs/ --show 
```
**Important notes:**
- If you do not provide intrinsics parameters, the system will automatically estimate them (performing calibration).
- Ensure your geodata covers the area visible in the query image. The localization and calibration framework requires sufficient overlap between the query image and geodata. Using geodata with large areas not visible in the query image may lead to poor results.

<a name="Benchmarking"></a>
### :straight_ruler: Benchmarking

To benchmark performance across a set of samples from the dataset or custom data:
```
run-benchmarking \
    --dirpath assets/samples/ --matcher Mast3R --device cuda --angles 0 --output_dir ./outputs/ --show
```
<a name="Visualization-of-a-Sample"></a>
### :eye: Visualization of a Sample

To visualize a single sample from the dataset:
```
visualize-sample \
    --sample assets/urban.npz --output_path ./outputs/sample.jpg --show
```
<a name="Visualization-of-Dataset-Samples"></a>
### :framed_picture: Visualization of Samples in the Dataset

To create a visualization of all samples in a dataset directory:
```
visualize-dataset \
    --dirpath assets/samples/ --output_path ./outputs/samples.jpg --show
```
<a name="License"></a>
## :warning: License

Distributed under CC BY-NC-SA 4.0. See LICENSE.txt for more information.

<a name="Acknowledgements"></a>
## :pray: Acknowledgements

Big thanks to the German government for making geospatial data freely available to everyone. These open data portals are
a goldmine for developers, researchers, and anyone curious about spatial information. Here's where you can find them:

### 🇩🇪 State Open Data Portals

- **Bavaria**: [geodaten.bayern.de/opengeodata](https://geodaten.bayern.de/opengeodata/)
- **Berlin**: [gdi.berlin.de](https://www.berlin.de/sen/sbw/stadtdaten/geoportal/)
- **Hesse**: [gds.hessen.de](https://gds.hessen.de/)
- **Hamburg**: [geoportal-hamburg.de](https://www.geoportal-hamburg.de/)
- **North Rhine-Westphalia**: [opengeodata.nrw.de](https://www.opengeodata.nrw.de/)
- **Baden-Württemberg**: [opengeodata.lgl-bw.de](https://opengeodata.lgl-bw.de/)

Special thanks to [Vincentqyw](https://github.com/Vincentqyw) for developing the [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui). This tool provides a user-friendly interface for matching between images using various state-of-the-art algorithms.
