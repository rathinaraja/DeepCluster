# What is DeepCluster++? 
Modern computer vision projects, across research and industry, often rely on supervised learning, which in turn demands well-curated, diverse training data. To efficiently gather representative samples from large image collections, we introduce DeepCluster++, a semi-automated dataset curation framework with three stages: (1) extract feature embeddings for all images using a domain-appropriate encoder (e.g., an autoencoder or a pre-trained backbone) or suitable pretrained encoder; (2) cluster the embeddings (e.g., with k-means) to group visually similar items and then apply equal-frequency sampling within clusters to capture diverse patterns for each class; and (3) have subject-matter experts review the selected samples to confirm label quality. By tuning a small set of parameters, DeepCluster++ lets practitioners balance the number of samples and the level of diversity, substantially reducing manual effort while yielding high-quality training data for robust models.
# Typical Workflow (at a glance)
This example demonstrates how to use DeepCluster++ to curate a diverse training set from tiles extracted out of whole-slide images (WSIs) in pathology.
1. Select WSIs that are representative of your cohort (e.g., cases spanning different tissue types).
2. Extract tiles (e.g., 256×256 pixels) from each WSI.
3. Preprocess tiles to retain quality tiles.
4. Arrange tiles on disk in a folder structure (single folder, multiple folders, or nested subfolders). DeepCluster++ is designed to work with any of the layouts below.
5. Feature extraction: Use a domain-appropriate encoder (pre-trained autoencoder or backbone) to embed all tiles in the input directory.
6. Clustering: Run k-means on embeddings to group morphologically similar tiles.
7. Diverse sampling: Apply equal-frequency binning (per cluster) to select a balanced, diverse subset for each class.
8. Data collection: Review the samples for each WSI and include them in the appropriate class type.
9. Expert review (optional but recommended): Have a subject-matter expert validate the sampled tiles before finalizing the training set.

# DeepCluster++ Usage Guide 
We assume representative WSIs have been selected, tiles extracted, and the resulting images filtered using appropriate preprocessing methods. The AutoEncoder (AE) used in this experiement was trained on a set of tiles (images) until the reconstruction quality of test samples become prominent. 

Important to note: 
1. RGB required: Ensure that all the images (tiles) are in RGB format.
2. Encoder input size: The pre-trained autoencoder used here was trained on images of size 256x256 pixels. If your tiles have a different size, either retrain an autoencoder at that size or use a compatible pre-trained encoder to extract features.
3. If you have a single folder with images or multiple folders with images or folders with subfolders or etc. We have designed the program work with any folder structure.
4. Explore the <a href="https://drive.google.com/drive/folders/193rN6BcE98ZMhPVWSHuKr1XEpZ1kJjbK?usp=drive_link" target="_blank" rel="noopener">input and output folder structure</a> to understand the following instructions.

Input folder structure
-------
Make sure the folder structure is followed as input_folder_1 or input_folder_2. The input folder may either contain images directly (flat structure input_folder_2) or include subfolders (input_folder_1) with images inside as given below. 

Refer to the Test_samples_1 or Test_samples_2 folder to visualize the outcomes of the following executions with various inputs. The command-line arguments can be adjusted based on the folder structure.

<pre>/input_path/Test_samples_1
├── input_folder_1 (WSI_1)
│   ├── sub_folder_1 (Informative_Part1)
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── sub_folder_2 (Informative_Part2)
│   │   └── ...
│   └── sub_folder_m (Informative_Partm)
├── input_folder_2 (WSI_2)
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── input_folder_n (WSI_n)
    ├── image1.png
    ├── image2.png
    └── ...
</pre>

Output folder structure
-------
The following set of files are created in the output folder.
<pre> 
Output
├── clusters
├── features
├── plots
└── samples
</pre>     
<pre> 
cluster (contains clusters of each WSI before sampling)
├── WSI_1
│ ├── Cluster_0
│ ├── Cluster_1
│ └── ...
├── WSI_2
│ └── ...
└── ...
</pre> 
<pre> 
features (2 csv files: features of images and its cluster assignment)
├── WSI_1
│ ├── cluster_assignments.csv
│ └── features.csv
├── WSI_2
│ └── ...
└── ...
</pre>
<pre> 
plots (2 image files: t-SNE visualization with k-means clusters - with and without cluster number)
├── WSI_1
│ ├── tsne_with_legend.png
│ └── tsne_with_numbers.png
├── WSI_2
│ └── ...
└── ...
</pre>
<pre> 
├── WSI_1
│ ├── Cluster_0
│ ├── Cluster_1
│ ├── Cluster_2
│ └── ...
├── WSI_2
│ └── ...
└── ...
</pre>

Create a virtual environment and install the required packages
-------
```bash
conda create -n DeepCluster python==3.9

pip install -r Requirements.txt 
```
Command-line arguments 
-------
+ Each input folder (WSI) should include a minimum of 256 images to match the 256 PCA components used. Folder with less than 256 images is still valid but there is no DeepCluster++ applied.
+ Consider the following key details about Test_samples to better understand the command-line arguments.
    - Input_path - /path/Test_samples
    - Input folders (WSIs) - WSI_1, WSI_2, WSI_3, WSI_4, and WSI_5
    - Sub folders (if available) - Informative_Part1, Informative_Part2, Informative_Part3, Less_Informative
+ Number of cluster for each input folder is determined by taking square root of number of samples in each input folder. 

| Argument                   | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `--input_path /path/Test_samples_1`      | The input path containing a set of input folders, each corresponding to a WSI. |
| `--selected_input_folders "WSI_1,WSI_2"`       | The input folders in the path (e.g., WSI names). By default, if not passed, all folders in the path are considered. |
| `--sub_folders "Informative_Part1,Informative_Part2"`         | If a WSI contains subfolders, specify which ones to process. By default, all subfolders in the input folder are considered. |
| `--process_all True`                  | Process all the images in the given input path regardless of sub_folders. |
| `--output_path /path/Output`          | Output path to store extracted features, clusters, plots, and samples. |
| `--device cpu`                       | Optional. Default: None. Device type (CPU or all_gpus). |
| `--gpu_ids 4,5`                          | Optional. Default: GPU `0` is assigned. |
| `--batch_size 128`                     | Optional. Default: `128`. Recommended: `256`. |
| `--distance_groups 5`                 | Default: `5`. |
| `--sample_percentage 0.2`             | Default: `0.2` (20%). Increase this value to collect more samples. |
| `--model AE_CRC.pth`                  | Model path. By default, `AE_CRC.pth` in the current path is used. |
| `--seed 42`                           | Optional. Default: `42`. |
| `--store_features True`               | Stores the features of input folders. Default: `False` (features are not stored). |
| `--store_clusters True`               | Stores the clusters. Default: `False` (clusters are not stored). |
| `--store_plots True`                  | Stores the plots of clusters. Default: `False` (plots are not stored). |
| `--store_samples True`                | Stores the samples of clusters. Default: `False` (samples are not stored). |
| `--store_samples_group_wise True`     | Stores the samples group-wise for each cluster. Default: `False` (samples are not stored). |

Command-line usage
-------  
### Basic Usage

To process all the input folders (WSIs) independently in the input path regardless of subfolders or images in each input folder using two GPUs. 

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output/ --gpu_ids 1
```
### Logging Output

To log all print statements into a text file, append `| tee output.txt` at the end of your command in the terminal:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output/ --gpu_ids 1 | tee Output.txt
``` 

> **Note:** The above execution does not store any details by default.

### Storing Results

To store extracted features, clusters, samples, and plots, pass `True` for the appropriate command line arguments:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output/ --gpu_ids 1 --store_features True --store_clusters True --store_plots True --store_samples True
```

### Processing Images Directly

If the input path contains only images (no input folders), those images will be processed directly. **Ensure at least 256 images are present in the input folder:**

```bash
python Main.py --input_path /path/Test_samples_1/WSI_2 --output_path /path/Output/ --gpu_ids 1 --store_features True --store_clusters True --store_plots True --store_samples True
```

## Advanced Usage

### Processing Specific Input Folders

To process specific input folders (e.g., WSI_1 and WSI_4), use the `--input_folders` parameter:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output/ --selected_input_folders "WSI_1,WSI_4" --gpu_ids 1 --store_features True --store_clusters True --store_plots True --store_samples True
```

### Storing Specific Results

#### Store Only Clusters
```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output/ --store_clusters True
```

#### Store Only Samples
```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output/ --store_samples True
```

### Processing Specific Subfolders

To process specific subfolders (e.g., "Informative_Part1, Informative_Part3") within the folders in the input path, use the `--sub_folders` parameter:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output/ --sub_folders "Informative_Part1,Informative_Part3" --store_clusters True --store_samples True
```

### Combined Folder and Subfolder Selection

To process specific subfolders (e.g., "Informative_Part1, Informative_Part3") within specific input folders (e.g., WSI_1 and WSI_3), use both `--input_folders` and `--sub_folders` parameters:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output --selected_input_folders "WSI_1,WSI_3" --sub_folders "Informative_Part1,Informative_Part3" --store_clusters True --store_samples True
```

### Sample Organization Options

By default, with `--store_samples True`, samples are stored cluster by cluster. To store samples within group folders instead:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Output --store_samples_group_wise True
```

### CPU and GPU usage

#### Single GPU Processing

Process input folders/WSIs serially using a single GPU, regardless of subfolders:

```bash
python Main.py  --input_path /path/Test_samples_1/ --output_path /path/Output --selected_input_folders "WSI_1,WSI_3" --gpu_ids 4 --store_features True --store_clusters True --store_plots True --store_samples True 
```
#### Multiple GPU Processing

Process input folders/WSIs in parallel using multiple specified GPUs, regardless of subfolders:

```bash
python Main.py  --input_path /path/Test_samples/ --output_path /path/Output/ --selected_input_folders "WSI_1,WSI_3" --gpu_ids 4,5 --store_features True --store_clusters True --store_plots True --store_samples True 
```

#### All Available GPUs Processing
Process input folders/WSIs in parallel using all available GPUs, regardless of subfolders:
```bash
python Main.py  --input_path /path/Test_samples/ --output_path /path/Output/ --selected_input_folders "WSI_1,WSI_3" --device all_gpus --store_features True --store_clusters True --store_plots True --store_samples True 
```

#### CPU Processing
Process input folders/WSIs serially using CPU only, regardless of subfolders:
```bash
python Main.py  --input_path /path/Test_samples/ --output_path /path/Output/ --selected_input_folders "WSI_1,WSI_3" --device all_gpus --store_features True --store_clusters True --store_plots True --store_samples True 
```

## Parameters Reference

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--input_path` | Path to input directory containing WSIs | Required | `/path/Test_samples/` |
| `--output_path` | Path to output directory | Required | `/path/Output/` |
| `--input_folders` | Comma-separated list of specific folders to process | `None` (all folders) | `"WSI_1,WSI_4"` |
| `--sub_folders` | Comma-separated list of specific subfolders to process | `None` (all subfolders) | `"Informative_Part1,Informative_Part3"` |
| `--store_features` | Store extracted features | `False` | `True`/`False` |
| `--store_clusters` | Store cluster results | `False` | `True`/`False` |
| `--store_plots` | Store visualization plots | `False` | `True`/`False` |
| `--store_samples` | Store sample images | `False` | `True`/`False` |
| `--store_samples_group_wise` | Organize samples by groups instead of clusters | `False` | `True`/`False` |

## Requirements

- Minimum 256 images per input folder for effective clustering
- Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

## Explore Clusters and Collect Representative Tiles

### Overview
Explore the different clusters of each WSI to collect representative tiles across the following tissue types:

| Tissue Type | Abbreviation | Description |
|-------------|--------------|-------------|
| **ADI** | Adipose | Adipose tissue/Fat cells |
| **LYM** | Lymphocyte | Lymphocytic infiltration |
| **MUS** | Muscle | Muscle tissue |
| **FCT** | Fibrous Connective Tissue | Loose connective tissue |
| **MUC** | Mucin | Mucin-rich areas |
| **NCS** | Necrotic Debris | Necrotic/Dead tissue |
| **BLD** | Blood | Red blood cells/Vascular areas |
| **TUM** | Tumor | Tumor tissue |
| **NOR** | Normal | Normal epithelial tissue |

### Cluster Exploration Process

**Review Generated Clusters**
   ```bash
   # Navigate to cluster output directory
   cd /path/to/output/samples/
   
   # Review clusters for each WSI
   ls -la WSI_*/Cluster_*
   ```
**Visual Inspection** 
   - Review sample images in each sample folder
   - Identify clusters that predominantly contain specific tissue types
**Manual Curation**
   - Review sampled tiles from each cluster
   - Create tissue-type specific folders:
     ```
     representative_tiles/
     ├── ADI/
     ├── LYM/
     ├── MUS/
     ├── FCT/
     ├── MUC/
     ├── NCS/
     ├── BLD/
     ├── TUM/
     └── NOR/
     ```

### Quality Assurance
- Ensure balanced representation across all 9 tissue types
- Verify cluster purity for each tissue type
- Document cluster-to-tissue-type mapping for reproducibility

## Step 7: Pathologist Verification

### Training Set Validation

The collected representative tiles undergo rigorous pathologist verification to ensure:

#### Annotation Quality Control
- **Expert Review**: Board-certified pathologists examine each representative tile
- **Consensus Building**: Multiple pathologists review ambiguous cases
- **Documentation**: All annotations are documented with reasoning

#### Verification Process
```bash
# Organize tiles for pathologist review
mkdir -p pathologist_review/{pending,verified,rejected}

# Move tiles to pending review folder
cp -r representative_tiles/* pathologist_review/pending/
```

#### Verification Criteria
| Criteria | Description | Action |
|----------|-------------|---------|
| **Tissue Type Accuracy** | Correct classification of tissue type | Accept/Reject/Reclassify |
| **Image Quality** | Clear, well-stained, artifact-free | Accept/Reject |
| **Representative Nature** | Typical example of tissue type | Accept/Request alternatives |
| **Diagnostic Relevance** | Clinically relevant features present | Accept/Enhance dataset |

#### Verification Workflow
**Initial Review**: Pathologist examines tiles by tissue type

**Quality Assessment**: Rate each tile (1-5 scale)

**Consensus Meeting**: Resolve disagreements

**Final Dataset**: Create verified training set

```bash
# After verification, organize final training set
mkdir -p verified_training_set/{ADI,LYM,MUS,FCT,MUC,NCS,BLD,TUM,NOR}

# Move verified tiles to final training folders
# (based on pathologist recommendations)
```

