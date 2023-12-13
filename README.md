# XGBoost-Parallel-Datamining-Project
A Project for analyzing Amazon Customer Review Data using Parallel XGBoost Models

This repository contains the Source Code, and datasamples for the XGboost portion of our model.

In order to run and test the module, you will need to first preprocess the data, and then you can run the Master Trainer. In both files, the name of the Dataset you're testing with needs to be named in the "Dataset" Variable I.E. if you're you want to operate on the Gift_Cards_5.json review data, your dataset should be "Gift_Cards_5". Additionally in both files, you must fill in the desired "Critical Value" or K-score by which you need a minimum number of reviews to make your processing worthwhile. If you wish to test multiple K values, data preprocessing can first be done at a low K and all necessary files will be created for trianing at both low and high K values.

When you run the Data Preprocessor, the output csvs will be stored in the data Folder, and a minimal master csv will be created which speeds up the reading in process for the Master Trainer. Be advised that Data Preprocessing on the large dataset, may take multiple hours depending on your system, and will write an additional 8GB of space on your disk.

Running the Master Trainer will conduct its own training/testing and report results to the terminal. A png named "out.png" will be created which contains a confusion matrix of the results. 
