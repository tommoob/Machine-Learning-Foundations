import os
from collections import defaultdict
from MNIST_data_intake import MNISTDataInjest
import numpy as np
from functools import partial
import argparse
from MNIST_pytorch_model import MNIST_hidden_layer, MnistCNN
import torch
import torch.nn as nn
import torch.optim as optim
from MNIST_image_dataset import ImageDataset
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.join(os.getcwd(), "utils"))
from utils.file_utils import file_utils
import cv2

file_tools = file_utils()

test_image_string, test_label_string = "t10k-images-idx3-ubyte", 't10k-labels-idx1-ubyte'
train_image_string, train_label_string = 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'

results_dir = os.path.join(os.getcwd(), "data/results")
save_errors = True


def main():
    load_parameters = False
    input_size = 784  
    hidden_size = 128
    output_size = 10  
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 1

    model_save_dir = os.path.join(
        results_dir, "models"
    )
    model_save_path = os.path.join(
        model_save_dir, file_utils.get_rand_filename(root="model"), ".pt"
        )
    os.makedirs(model_save_dir, exist_ok=True)

    
    model = MnistCNN()
    #model = MNIST_hidden_layer(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train
    train_dataset = ImageDataset(MNIST_data.images[train_image_string], MNIST_data.labels[train_label_string])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            labels = labels.to(torch.int64)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

    # Evaluate

    torch.save(model, model_save_path)
    test_dataset = ImageDataset(MNIST_data.images[test_image_string], MNIST_data.labels[test_label_string])
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model.eval()

    with torch.no_grad():
        correct = defaultdict(int)
        total = defaultdict(int)

        for images, labels in test_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for ix, lab in enumerate(labels):
                total[int(lab)] += 1
                if predicted[ix] == lab:
                    correct[int(lab)] += 1
                elif save_errors:
                    file_save_dir = os.path.join(results_dir, "data/results", 
                                                 file_tools.get_rand_filename(
                                                     f"falsely_predicted_as_{int(predicted[ix])}___"
                                                     ), ".jpg")
                    print(predicted[ix])
                    img = images[ix].numpy()
                    cv2.imwrite(file_save_dir, img)
        for key in total:
            print(f'Accuracy of the model to detect number: {key} in the test images is: {100 * correct[key] / total[key]}%')


def get_data():

    byte_num = 1
    ls = os.listdir(path_dir)
    for dir in ls:
        if "labels" in dir:
            MNIST_data.injest_labels(path_dir, byte_num, dir)
        elif "images" in dir:
            MNIST_data.injest_images(path_dir, dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/thomas/dev/src/Machine-Learning-Foundations/data/MNIST_data", help='what sort of augmentation')
    opt = parser.parse_args()
   
    path_dir = opt.path
    imgs, labs = defaultdict(partial(np.ndarray, 0)), defaultdict(partial(np.ndarray, 0))
   
    MNIST_data = MNISTDataInjest(imgs, labs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    get_data()

    main()

