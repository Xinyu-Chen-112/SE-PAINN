# This is the source code for paper [Structure embedding method for machine learning models accelerates research on stacked 2D materials].

## Some files were not uploaded due to their large size. Readers can use these codes to obtain them themselves or request them from the author.

## file tree directory
root_dir  
├── convert  
&emsp;├── input  
&emsp;&emsp;├── binding_energy  
&emsp;&emsp;&emsp;├── *.vasp  
&emsp;&emsp;├── band_gap  
&emsp;&emsp;&emsp;├── *.vasp  
&emsp;├── poscars  
&emsp;&emsp;├── *.vasp  
&emsp;├── samples  
&emsp;&emsp;├── *.csv  
&emsp;├── *.py  
&emsp;├── samples.csv  
├── input  
&emsp;├── data  
&emsp;&emsp;├── *.py  
&emsp;├── model  
&emsp;&emsp;├── *.py  
&emsp;├── train_data  
&emsp;&emsp;├── nbrs_data.pkl  
&emsp;&emsp;├── id_prop.csv  
&emsp;├── predict_data  
&emsp;&emsp;├── *.vasp  
&emsp;&emsp;├── id_prop.csv  
&emsp;├── args.txt  
&emsp;├── atom_init.txt  
&emsp;├── bo*.py  
&emsp;├── main.py  
&emsp;├── tools.py  
├── output  
&emsp;├── train  
&emsp;&emsp;├── printlog.txt  
&emsp;&emsp;├── commandline_args.txt  
&emsp;&emsp;├── arguments.json  
&emsp;&emsp;├── split_data.json  
&emsp;&emsp;├── best_model.pth  
&emsp;&emsp;├── last_model.pth  
&emsp;&emsp;├── loss.csv  
&emsp;&emsp;├── rmse-loss.png  
&emsp;├── predict  
&emsp;&emsp;├── printlog.txt  
&emsp;&emsp;├── commandline_args.txt  
&emsp;&emsp;├── arguments.json  
&emsp;&emsp;├── samples-predict.csv  
&emsp;├── bo  
&emsp;&emsp;├── results  
&emsp;&emsp;&emsp;├── *.pkl  
&emsp;&emsp;├── convergence.png  
&emsp;&emsp;├── evaluations.png  
&emsp;&emsp;├── objective.png  
&emsp;&emsp;├── printlog.txt  
├── stack2d  
&emsp;├── 2d-poscars  
&emsp;&emsp;├── *.vasp  
&emsp;├── stack2d.py  

## Model naming
                        Distance embedding                Bond-type embedding                Structure embedding  
CGCNN                        cgcnn                             cgcnn_01                            cgcnn_cv  
MEGNET                       megnet                            megnet_01                           megnet_cv  
PAINN                        painn                             painn_01                            painn_cv  

# How to Use  
## Step 1: Use stack2d to build a dataset.

├── "stack2d" folder contains files related to the supercell modeling program for stacked 2D materials.
&emsp;├── "2d-poscars" folder provides vasp structure files of 104 2D materials selected in the article.
&emsp;├── "stack2d.py" is the modeling program that can construct the stacked supercell of any two layers of 2D materials after any rotation and displacement. See the comments in the file for specific usage methods.

Please build your own dataset according to your needs


## Step 2: Preprocessing the dataset for training.

├── "convert" folder contains files related to preprocessing. This operation can save a lot of time reading data during training.
&emsp;├── "input" folder provides the vasp structure files of the binding energy dataset and band gap dataset used in the article.
&emsp;├── "samples" folder provides sample-target files for the binding energy dataset and band gap dataset used in the article.

To preprocess, you need to add the vasp structure files to "poscars" folder, add the corresponding sample-target file to "covert" folder and rename it to "id_prop.csv", and then execute the python program of the corresponding model to generate the corresponding "nbrs_data.pkl".


## Step 3: Train, predict, and other operations.

&emsp;├── "input/atom_init.txt" provides the element features used in the article, and readers can also define them themselves.
&emsp;├── "input/args.txt" provides the input parameters of the main execution file "main.py". You can execute "python main.py -h" in the terminal to view the meaning of each parameter.

To train, you need to add corresponding "nbrs_data.pkl" and "id_prop.csv" to "input/train_data" folder, set up "args.txt", and then execute "python main.py @args.txt" in the terminal. The training results will be output to "output/train" folder

To predict, you need to add vasp structure files and corresponding "id_prop.csv" to "input/predict_data" folder, set up "args.txt" (Note that cutoff and max_num_nbrs need to be consistent with those during training), and then execute "python main.py @args.txt" in the terminal. The prediction results will be output to "output/predict" folder.

To optimize hyperparameters via random search, you can customize the hyperparameter types, search range, search times, etc. in the "bo_*.py" of each model, and then execute "python bo_*.py" in the terminal.The optimization results will be output to "output/bo" folder.
