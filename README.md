DeepCluster++ Usage Guide
---------------------------------
Step 1: Understand the input folder structure
-------
Select a representative set of WSIs to build the training dataset, extract 256×256 tiles, preprocess them, and store them in the folder structure (like input_folder_1 or ) shown below. Refer to the Test_samples folder to visualize the outcomes of the following executions with various inputs. Ensure that all tiles are in RGB format.

The input folder may either contain images directly (flat structure input_folder_2) or include subfolders (input_folder_1) with images inside as given below. Command-line arguments can be adjusted based on the folder structure.
<pre>input_path/Test_samples
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

Step 2: Understand the output folder structure
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

Step 3: Create a virtual environment and install the required packages.
-------
```bash
pip install -r requirements.txt
```
Step 4: Understand the command-line arguments 
-------
+ Each input folder (WSI) should include a minimum of 256 images to match the 256 PCA components used. 
+ Consider the following key details about Test_samples to better understand the command-line arguments.
    - Input_path - /path/Test_samples
    - Input folders (WSIs) - WSI_1, WSI_2, WSI_3, WSI_4, and WSI_5
    - Sub folders (if available) - Informative_Part1, Informative_Part2, Informative_Part3, Less_Informative
+ Number of cluster for each input folder is determined by taking square root of number of samples in each input folder.

### Commandline arguments 

| Argument                   | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `--input_path /path/WSI/folders`      | The input path containing a set of input folders, each corresponding to a WSI. |
| `--input_folders "WSI_1,WSI_2"`       | The input folders in the path (e.g., WSI names). By default, if not passed, all folders in the path are considered. |
| `--sub_folders "Part2,Part3"`         | If a WSI contains subfolders, specify which ones to process. By default, all subfolders in the input folder are considered. |
| `--process_all True`                  | Process all the images in the given input path. |
| `--output_path /path/Output`          | Output path to store extracted features, clusters, plots, and samples. |
| `--batch_size 64`                     | Optional. Default: `64`. Recommended: `256`. |
| `--distance_groups 5`                 | Default: `5`. |
| `--sample_percentage 0.2`             | Default: `0.2` (20%). Increase this value to collect more samples. |
| `--device cuda`                       | Optional. Device type (CPU or CUDA). |
| `--gpu_id 7`                          | Optional. Default: GPU `0` is assigned. |
| `--model AE_CRC.pth`                  | Model path. By default, `AE_CRC.pth` in the current path is used. |
| `--seed 42`                           | Optional. Default: `42`. |
| `--store_features True`               | Stores the features of input folders. Default: `False` (features are not stored). |
| `--store_clusters True`               | Stores the clusters. Default: `False` (clusters are not stored). |
| `--store_plots True`                  | Stores the plots of clusters. Default: `False` (plots are not stored). |
| `--store_samples True`                | Stores the samples of clusters. Default: `False` (samples are not stored). |
| `--store_samples_group_wise True`     | Stores the samples group-wise for each cluster. Default: `False` (samples are not stored). |

Step 5: Command-line usage.
-------  
### Basic Usage

To process all the input folders (WSIs) independently in the input path regardless of subfolders or only images in each input folder:

```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output
```
### Logging Output

To log all print statements into a text file, append `| tee output.txt` at the end of your command in the terminal:

```bash
python DeepCluster++_Serial_Full_Code.py --input_path /data_64T_1/Raja/TEST/Input/Test_samples --output_path /data_64T_1/Raja/TEST/Output | tee output.txt
``` 

> **Note:** The above execution does not store any details by default.

### Storing Results

To store extracted features, clusters, samples, and plots, pass `True` for the appropriate command line arguments:

```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_features True --store_clusters True --store_plots True --store_samples True
```

### Processing Images Directly

If the input path contains only images (no folders), those images will be processed directly. **Ensure at least 256 images are present in the input folder:**

```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_features True --store_clusters True --store_plots True --store_samples True
```

## Advanced Usage

### Processing Specific Input Folders

To process specific input folders (e.g., WSI_1 and WSI_4), use the `--input_folders` parameter:

```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --input_folders "WSI_1,WSI_4" --store_clusters True --store_samples True
```

### Storing Specific Results

#### Store Only Clusters
```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_clusters True
```

#### Store Only Samples
```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_samples True
```

### Processing Specific Subfolders

To process specific subfolders (e.g., "Informative_Part1, Informative_Part3") within the input folders, use the `--sub_folders` parameter:

```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --sub_folders "Informative_Part1,Informative_Part3" --store_clusters True --store_samples True
```

### Combined Folder and Subfolder Selection

To process specific subfolders (e.g., "Informative_Part1, Informative_Part3") within specific input folders (e.g., WSI_1 and WSI_3), use both `--input_folders` and `--sub_folders` parameters:

```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --input_folders "WSI_1,WSI_3" --sub_folders "Informative_Part1,Informative_Part3" --store_clusters True --store_samples True
```

### Sample Organization Options

By default, with `--store_samples True`, samples are stored cluster by cluster. To store samples within group folders instead:

```bash
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_samples_group_wise True
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

## Step 7: Explore Clusters and Collect Representative Tiles

### Overview
Explore the different clusters of each WSI to collect representative tiles across the following tissue types:

| Tissue Type | Abbreviation | Description |
|-------------|--------------|-------------|
| **ADI** | Adipose | Adipose tissue/Fat cells |
| **LYM** | Lymphocyte | Lymphocytic infiltration |
| **MUS** | Muscle | Muscle tissue |
| **FCT** | Fibrous Connective Tissue | Loose connective tissue/Stroma |
| **MUC** | Mucin | Mucin-rich areas |
| **NCS** | Necrotic Debris | Necrotic/Dead tissue |
| **BLD** | Blood | Red blood cells/Vascular areas |
| **TUM** | Tumor | Tumor tissue |
| **NOR** | Normal | Normal epithelial tissue |

### Cluster Exploration Process

1. **Review Generated Clusters**
   ```bash
   # Navigate to cluster output directory
   cd /path/to/output/clusters/
   
   # Review clusters for each WSI
   ls -la WSI_*/Cluster_*
   ```

2. **Visual Inspection**
   - Examine t-SNE plots in the `plots/` directory to understand cluster distribution
   - Review sample images in each cluster folder
   - Identify clusters that predominantly contain specific tissue types

3. **Representative Tile Selection**
   ```bash
   # Use the sampling functionality to extract representative tiles
   python Main.py --input_path /path/WSI_data/ \
                  --output_path /path/representative_tiles/ \
                  --store_samples True \
                  --store_samples_group_wise True \
                  --sample_percentage 0.1
   ```
