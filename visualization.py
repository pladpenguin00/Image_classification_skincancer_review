import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves(train_losses, test_accuracies, save_path):
    epochs = range(1, len(train_losses) + 1)
    # breakpoint()
    plt.figure(figsize=(10, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'r', label='Test accuracy')
    plt.title('Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Saved training curves to", save_path)
    print("Training loss:", train_losses)

def save_classified_examples(classified, output_path):
    # breakpoint()
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    benign_axes = []
    malignant_axes = []
    for i, ax in enumerate(axes.flat):
        if i < len(classified):
            img = classified[i].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            if classified[i] == "benign":
                benign_axes.append(ax)
            elif classified[i] == "malignant":
                malignant_axes.append(ax)
    
    plt.savefig(output_path)
    plt.close()
    print("Saved classified examples to", output_path)

def save_accuracy(accuracy, results_dir, input_data_dir):
    accuracy_file = os.path.join(results_dir, 'accuracy.txt')
    with open(accuracy_file, 'w') as f:
        f.write(f'Test_Accuracy on data {input_data_dir}: {accuracy}')

    print(f"Saved accuracy to {accuracy_file}")