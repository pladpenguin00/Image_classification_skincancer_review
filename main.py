import os
import shutil
import argparse


# Import visualization functions
from visualization import plot_training_curves, save_classified_examples

# Import model and training functions
from model_selection import SkinCancerModel, train_model, evaluate_model, save_model

# Import dataset and dataloader functions
from dataloader import CustomSkinCancerDataset, pick_dataloader


# Create argument parser
parser = argparse.ArgumentParser(description='Skin Cancer Classification')

# Add arguments
parser.add_argument('--model', type=str, default='resnet18', help='Model option ("resnet18", "resnet34", "resnet50")')
parser.add_argument('--loss', type=str, default='BCELoss', help='Loss option ("BCELoss", "CrossEntropyLoss", "MSELoss", "NLLLoss", "SmoothL1Loss")')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer option ("Adam", "SGD", "RMSprop", "Adagrad", "AdamW")')

# Parse the arguments
args = parser.parse_args()

# Get the model, loss, and optimizer options from the arguments
model_option = args.model
loss_option = args.loss
optimizer_option = args.optimizer

if __name__ == "__main__":
    # String names for results directory
    model_save_path = '/skin_cancer_model'
    training_curve_path = '/training_curve.png'
    correctly_classified_path = '/correctly_classified'
    incorrectly_classified_path = '/incorrectly_classified'
    custom_name = "TRAINING_2"

    # Define the model, loss, and optimizer options`
    # model_option = "resnet18"  # Choose the model option ("resnet18", "resnet34", "resnet50")
    # loss_option = "BCELoss"  # Choose the loss option ("BCELoss", "CrossEntropyLoss", "MSELoss", "NLLLoss", "SmoothL1Loss")
    # optimizer_option = "Adam"  # Choose the optimizer option ("Adam", "SGD", "RMSprop", "Adagrad", "AdamW")


    # Create a directory for the results
    results_dir = f'results/{model_option}_{loss_option}_{optimizer_option}_{custom_name}'
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
    batch_size=256
    train_loader, test_loader = pick_dataloader('..\\all_data', batch_size=batch_size)

    # Load the model
    model = SkinCancerModel(model_option, loss_option, optimizer_option)

    # Train the model, Test the model, and Save the model
    train_losses, test_accuracies, correctly_classified, incorrectly_classified = train_model(model, train_loader, test_loader, epochs=360, save_period=30, model_save_path=results_dir+model_save_path)
    # accuracy, correctly_classified, incorrectly_classified = evaluate_model(model, test_loader)

    # Save the training curves and classified examples
    plot_training_curves(train_losses, test_accuracies, results_dir+training_curve_path)
    save_classified_examples(correctly_classified[:100], results_dir+correctly_classified_path)
    save_classified_examples(incorrectly_classified[:100], results_dir+incorrectly_classified_path)