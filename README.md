# Image-Processing-Assignment
Image Processing Assignment of MSDSA Faculty

### Requirements:
- Python
- Anaconda/Miniconda[https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html]

### The environment is exported to requirements.yaml
Commands to run:
- Create environment in local using requirements.yaml
>conda env create -f requirements.yaml
- Run the conda virtual environment
>conda activate computer_vision
- To Run Jupyter Lab on the command line
>jupyter-lab

### Some Notes:
- The project tries to avoid built-in functions for image processing as much as it could.
- The project however uses built-in numpy matrix manipulation as much as possible.
- The broadcasting features is used very frequently.
- Any function that can be vectorized is vectorized for efficient computation.
- Images used for testing the functions is given directly in the repo in the Images folder.
- The functions are modularised and their applications are shown in jupyter Notebook.
- The functions works for all grayscale images but only works for some colour images.
