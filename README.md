# Image_classification_skincancer_review
Evaluate given dataset to classify skin cancer images as benign or malignant. 

To run :

python main.py


•main.py
o	This is file that you run to run training and testing
o	There is an args parser for the input user to setup specific arguments like what model, loss, optimizer, ect to use

•dataloader.py
o	custom data set class for controlling preprocessing steps 
o	we have specific preprocessing used in the training that does not need to be used on test examples


•model_selection.py
o	SkinCancerModel is our nn model class, choosing the backbone of the solver is mapped out here
o	train_model and evaluate_model functions are here and called in main

•visualization.py
o	visualization code is for plots and image examples are held here and called in main
o	some work is done in other files to make sure the needed information is passed to these functions 

•run_tests.sh
o	file that would be used to control running many main.py and arguments so that a person isn’t writing every command. Just a way to easily have training and testing continue automatically when setup
