import os
import shutil
import argparse


# Import visualization functions
from visualization import plot_training_curves, save_classified_examples

# Import model and training functions
from model_selection import SkinCancerModel, train_model, evaluate_model, save_model

# Import dataset and dataloader functions
from dataloader import CustomSkinCancerDataset, pick_dataloader
import datetime


# Create argument parser
parser = argparse.ArgumentParser(description='Skin Cancer Classification')

# Add arguments
parser.add_argument('--input_data_dir', type=str, default='..\\all_data', help='Model option ("..\\all_data", "..\\debug_data")')
parser.add_argument('--model', type=str, default='resnet18', help='Model option ("resnet18", "resnet34", "resnet50")')
parser.add_argument('--loss', type=str, default='BCELoss', help='Loss option ("BCELoss", "CrossEntropyLoss", "MSELoss", "NLLLoss", "SmoothL1Loss")')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer option ("Adam", "SGD", "RMSprop", "Adagrad", "AdamW")')
parser.add_argument('--model_save_path', type=str, default='/skin_cancer_model', help='Path to save the trained model')
parser.add_argument('--training_curve_path', type=str, default='/training_curve.png', help='Path to save the training curve plot')
parser.add_argument('--correctly_classified_path', type=str, default='/correctly_classified', help='Path to save the correctly classified examples')
parser.add_argument('--incorrectly_classified_path', type=str, default='/incorrectly_classified', help='Path to save the incorrectly classified examples')
parser.add_argument('--custom_name', type=str, default='TRAINING_1', help='Custom name for the results directory')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--save_period', type=int, default=10, help='Period to save the model')
parser.add_argument('--data_split', type=int, default=80, help='Percentage of data to use for training, give integer value between 0 and 100')

# Parse the arguments
args = parser.parse_args()

# Get the model, loss, and optimizer options from the arguments
input_data_dir = args.input_data_dir
model_option = args.model
loss_option = args.loss
optimizer_option = args.optimizer
model_save_path = args.model_save_path
training_curve_path = args.training_curve_path
correctly_classified_path = args.correctly_classified_path
incorrectly_classified_path = args.incorrectly_classified_path
custom_name = args.custom_name
batch_size = args.batch_size
epochs = args.epochs
save_period = args.save_period
data_split = args.data_split

if __name__ == '__main__':
    # Create a directory for the results
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    results_dir = f'results/{date}_{model_option}_{loss_option}_{optimizer_option}_{custom_name}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        user_input = input("Results directory already exists. Do you want to overwrite it? (Y/N): ")
        if user_input.lower() == "y":
            shutil.rmtree(results_dir)
            os.makedirs(results_dir)
        else:
            print("Stopping the run.")
            exit(1)



    # Load the dataset
    train_loader, test_loader = pick_dataloader(input_data_dir, batch_size=batch_size, data_split=data_split)

    # Load the model
    model = SkinCancerModel(model_option, loss_option, optimizer_option)

    # Train the model, Test the model, and Save the model
    train_losses, test_accuracies, correctly_classified, incorrectly_classified = train_model(model, train_loader, test_loader, epochs=epochs, save_period=save_period, model_save_path=results_dir+model_save_path)
    # accuracy, correctly_classified, incorrectly_classified = evaluate_model(model, test_loader)

    # Save the training curves and classified examples
    plot_training_curves(train_losses, test_accuracies, results_dir+training_curve_path)
    save_classified_examples(correctly_classified[:100], results_dir+correctly_classified_path)
    save_classified_examples(incorrectly_classified[:100], results_dir+incorrectly_classified_path)