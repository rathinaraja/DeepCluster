DeepCluster++
---------------------------------
Step 1: Understand the input folder structure
-------
Select a representative set of WSIs to build the training dataset, extract 256×256 tiles, preprocess them, and store them in the folder structure (like input_folder_1 or ) shown below. Refer to the Test_samples folder to visualize the outcomes of the following executions with various inputs. Ensure that all tiles are in RGB format.

The input folder may either contain images directly (flat structure input_folder_2) or include subfolders (input_folder_1) with images inside as given below. Command-line arguments can be adjusted based on the folder structure.
<pre> input_path/Test_samples
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
pip install -r requirements.txt

Step 4: Understand the command-line arguments 
-------
+ Each input folder (WSI) should include a minimum of 256 images to match the 256 PCA components used. 
+ Consider the following key details about Test_samples to better understand the command-line arguments.
    - Input_path - /path/Test_samples
    - Input folders (WSIs) - WSI_1, WSI_2, WSI_3, WSI_4, and WSI_5
    - Sub folders (if available) - Informative_Part1, Informative_Part2, Informative_Part3, Less_Informative
+ Number of cluster for each input folder is determined by taking square root of number of samples in each input folder.

--input_path /path/WSI/folders       # The input path contains a set of input folders, each corresponding to a WSI.

--input_folders "WSI_1,WSI_2"        # The input folders in the path (e.g., WSI names). By default, if --input_folders 
                                     # is not passed, all folders in the path are considered.
--sub_folders "Part2,Part3"          # If a WSI contains subfolders, specify which ones to process. By default (None or if
                                     # --sub_folders is not provided), all subfolders in the input folder are considered.
--process_all True                   # To process all the images in the given input path

--output_path /path/Output           # Output path to store extracted features, clusters, plots, and samples

--batch_size 64                      # Optional. Default: 64. Recommended: 256.

--distance_groups 5                  # Default 5

--sample_percentage 0.2              # Default: 0.2 (20%). Increase this value to collect more samples.

--device cuda                        # Optional 

--gpu_id 7                           # Optional. GPU 0 is assigned by default.    

--model AE_CRC.pth                   # Model path. By default, AE_CRC.path is used in the current path.

--seed 42                            # Optional, default 42

--store_features True                # Stores the features of input folders. Default: False (features are not stored).                

--store_clusters True                # Stores the clusters. Default: False (clusters are not stored).   

--store_plots True                   # Stores the plots of clusters. Default: False (plots are not stored).

--store_samples True                 # Stores the samples of clusters. Default: False (samples are not stored).

--store_samples_group_wise True      # Stores the samples group-wise for each cluster. Default: False (samples are not stored).

Step 6: Execute the program using different sets of command-line arguments.
-------
To process all the input folders (WSIs) independently in the input path regardless of subfolders or only images in each input folder. 
python Main.py --input_path /path/Test_samples/ --output_path /path/Output  

To log all print statements into a text file, append | tee output.txt at the end of your command in the terminal.
python DeepCluster++_Serial_Full_Code.py --input_path /data_64T_1/Raja/TEST/Input/Test_samples --output_path /data_64T_1/Raja/TEST/Output | tee output.txt

The above execution does not store any details. To store extracted features, clusters, samples, and plots, pass True for the appropriate command line arguments. 
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_features True --store_clusters True --store_plots True --store_samples True

If the input path contains only images (no folders), those images will be processed directly. Ensure at least 256 images are present in the input folder.
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_features True --store_clusters True --store_plots True --store_samples True

To process specific input folders (e.g., WSI_1 and WSI_4), use the --input_folders parameter.
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --input_folders "WSI_1,WSI_4" --store_clusters True --store_samples True 

To store only the clusters.
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_clusters True  

To store only the samples.
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_samples True  

To process specific subfolders (e.g., "Informative_Part1, Informative_Part3") within the input folders, use the --sub_folders parameter.
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --sub_folders "Informative_Part1,Informative_Part3" --store_clusters True --store_samples True 

To process specific subfolders (e.g., "Informative_Part1, Informative_Part3") within specific input folders (e.g., WSI_1 and WSI_3), use the --input_folders and --sub_folders parameters.
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --input_folders "WSI_1,WSI_3" --sub_folders "Informative_Part1,Informative_Part3" --store_clusters True --store_samples True 
 
To store only samples within group folder (by default with --store_samples True, samples are stored cluster by cluster)
python Main.py --input_path /path/Test_samples/ --output_path /path/Output --store_samples_group_wise
---------------------------------

Step 7: Explore the different clusters of each WSI to collect representative tiles across tissue types: ADI, LYM, MUS, FCT, MUC, NCS, BLD, TUM, and NOR.

---------------------------------

Step 8: The training set has been verified by pathologists.

---------------------------------

Step 9: Perform tile normalization (Macenko) and validate with the models.

---------------------------------
