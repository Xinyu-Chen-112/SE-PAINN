# This is the source code for paper [Structure embedding method for machine learning models accelerates research on stacked 2D materials].

## Some files were not uploaded due to their large size. Readers can use these codes to obtain them themselves or request them from the author.

## Complete file tree
root_dir  
├── convert  
&emsp;&emsp;├── input  
&emsp;&emsp;&emsp;&emsp;├── binding_energy  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── \*.vasp  
&emsp;&emsp;&emsp;&emsp;├── band_gap  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── \*.vasp  
&emsp;&emsp;├── poscars  
&emsp;&emsp;&emsp;&emsp;├── \*.vasp  
&emsp;&emsp;├── samples  
&emsp;&emsp;&emsp;&emsp;├── \*.csv  
&emsp;&emsp;├── \*.py  
&emsp;&emsp;├── id_prop.csv  
&emsp;&emsp;├── nbrs_data.pkl  
├── input  
&emsp;&emsp;├── data  
&emsp;&emsp;&emsp;&emsp;├── \*.py  
&emsp;&emsp;├── model  
&emsp;&emsp;&emsp;&emsp;├── \*.py  
&emsp;&emsp;├── train_data  
&emsp;&emsp;&emsp;&emsp;├── id_prop.csv  
&emsp;&emsp;&emsp;&emsp;├── nbrs_data.pkl  
&emsp;&emsp;├── predict_data  
&emsp;&emsp;&emsp;&emsp;├── id_prop.csv  
&emsp;&emsp;&emsp;&emsp;├── \*.vasp  
&emsp;&emsp;├── args.txt  
&emsp;&emsp;├── atom_init.txt  
&emsp;&emsp;├── bo_\*.py  
&emsp;&emsp;├── main.py  
&emsp;&emsp;├── tools.py  
├── output  
&emsp;&emsp;├── train  
&emsp;&emsp;&emsp;&emsp;├── printlog.txt  
&emsp;&emsp;&emsp;&emsp;├── commandline_args.txt  
&emsp;&emsp;&emsp;&emsp;├── arguments.json  
&emsp;&emsp;&emsp;&emsp;├── split_data.json  
&emsp;&emsp;&emsp;&emsp;├── best_model.pth  
&emsp;&emsp;&emsp;&emsp;├── last_model.pth  
&emsp;&emsp;&emsp;&emsp;├── loss.csv  
&emsp;&emsp;&emsp;&emsp;├── rmse-loss.png  
&emsp;&emsp;├── predict  
&emsp;&emsp;&emsp;&emsp;├── printlog.txt  
&emsp;&emsp;&emsp;&emsp;├── commandline_args.txt  
&emsp;&emsp;&emsp;&emsp;├── arguments.json  
&emsp;&emsp;&emsp;&emsp;├── samples-predict.csv  
&emsp;&emsp;├── bo  
&emsp;&emsp;&emsp;&emsp;├── results  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── \*.pkl  
&emsp;&emsp;&emsp;&emsp;├── convergence.png  
&emsp;&emsp;&emsp;&emsp;├── evaluations.png  
&emsp;&emsp;&emsp;&emsp;├── objective.png  
&emsp;&emsp;&emsp;&emsp;├── printlog.txt  
├── stack2d  
&emsp;&emsp;├── 2d-poscars  
&emsp;&emsp;&emsp;&emsp;├── \*.vasp  
&emsp;&emsp;├── stack2d.py  

## Model naming
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Distance embedding&emsp;&emsp;&emsp;&emsp;&emsp;Bond-type embedding&emsp;&emsp;&emsp;&emsp;&emsp;Structure embedding  
CGCNN&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;cgcnn&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;cgcnn_01&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;cgcnn_cv  
MEGNET&emsp;&emsp;&emsp;&emsp;&emsp;megnet&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;megnet_01&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&thinsp;megnet_cv  
PAINN&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;painn&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&thinsp;painn_01&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&thinsp;painn_cv  

# How to Use  
## Step 1: Use stack2d to build a dataset.

├── **stack2d** contains files related to the supercell modeling program for stacked 2D materials.  
&emsp;&emsp;├── **2d-poscars** provides vasp structure files of 104 2D materials selected in the article.  
&emsp;&emsp;├── **stack2d.py** is the modeling program that can construct the stacked supercell of any two layers of 2D materials after any rotation and displacement. See the comments in the file for specific usage methods.  

Please build your own dataset according to your needs


## Step 2: Preprocess the dataset for training.

├── **convert** contains files related to preprocessing. This operation can save a lot of time reading data during training.  
&emsp;&emsp;├── **input** provides the vasp structure files of the binding energy dataset and band gap dataset used in the article.  
&emsp;&emsp;├── **samples** provides sample-target files for the binding energy dataset and band gap dataset used in the article.  

To preprocess, you need to add the vasp structure files to **poscars**, add the corresponding sample-target file to **covert** and rename it to **id_prop.csv**, and then execute the python program of the corresponding model to generate the corresponding **nbrs_data.pkl**.


## Step 3: Train, predict, and other operations.

├── **input** contains a series of input files and executable files.  
&emsp;&emsp;├── **args.txt** provides the input parameters of the main execution program **main.py**. You can execute `python main.py -h` in the terminal to view the meaning of each parameter.  
&emsp;&emsp;├── **atom_init.txt** provides the element features used in the article, and readers can also define them themselves.  

To train, you need to add corresponding **nbrs_data.pkl** and **id_prop.csv** to **train_data**, set up **args.txt**, and then execute `python main.py @args.txt` in the terminal. The training results will be output to **output/train**.

To predict, you need to add vasp structure files and corresponding **id_prop.csv** to **input/predict_data**, set up **args.txt** (_Note that *cutoff and *max_num_nbrs must be consistent with those during training_), and then execute `python main.py @args.txt` in the terminal. The prediction results will be output to **output/predict**.

To optimize hyperparameters via random search, you can customize the hyperparameter types, search range, search times, etc. in the **bo_*.py** of each model, and then execute `python bo_\*.py` in the terminal.The optimization results will be output to **output/bo**.
