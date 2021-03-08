# Data Science Project Template

This repository is a template to get started with any data science project. This will be great for a beginner when using machine learning or data science as a starter project.

## Project Structure
```
.
├── data
│   ├── preprocessed
│   └── raw
├── models
│   └── __init__.py
├── notebook
│   └── exploratory_data_analysis.ipynb
├── result
├── src
│   ├── train.py
│   └── utils.py
└── README.md
```
Each directory has its own purpose. This is the explanation of the directories.
* `data` The data directory is a place to store your dataset. Usually we will deal with raw data and ready to use data. Therefore we separate them into `raw` to store the raw and `preprocessed` to store the train-test-dev dataset. It is always nice not to change the raw data as we might do any data engineering.
* `models` directory is __not__ the one which we store the output of the machine learning process, instead this is where we keep the model definition. Either it will be an ANN, CNN etc. This is good if we want to use deep learning frameworks such as TensorFlow or PyTorch.
* `notebooks` is where we store our Jupyter Notebooks. Mainly this is where we keep the exploratory or any feature engineering process. 
* `result` is where we want to store our output. We can save anything related to the output such as history from the training, or the evaluation result, or the weight of the model itself.
* `src` is the python scripts that we use. We can add utils function or training script or anything. Mostly we want to store any `*.py` files into this directory.
* `README.md` is the first thing that we will see in a project. It is always nice to have prologue and summary from the project we are going to reed before we jump to the code.

And that's it. We are set to start our project. 
Have fun!

