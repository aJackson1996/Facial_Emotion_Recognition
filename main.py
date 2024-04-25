import argparse
import math
from copy import deepcopy

import cv2
import numpy as np
import torch.cuda
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import FER_Model, FER_Model_v2
import deeplake
import pandas as pd
from mtcnn import MTCNN
from matplotlib.patches import Rectangle


def train_model(model, train_set, train_labels, classification_loss, optimizer, device):
    losses = []
    correct = 0
    model.train()
    for i in range(len(train_set)):
        data_batch = train_set[i]
        label_batch = train_labels[i]
        data_batch, label_batch = data_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        output = model(data_batch)
        pred = output.argmax(dim=1, keepdim=True)
        loss = classification_loss(output, label_batch)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        correct += (pred == label_batch.reshape(label_batch.shape[0], -1)).sum().item()

    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / (len(train_set) * 19)
    outputstring = 'Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (len(train_set) * 19),
        100. * correct / (len(train_set) * 19))

    print(outputstring)
    return train_loss, train_acc


def test_model(model, test_set, test_labels, classification_loss, device):
    correct = 0
    losses = []
    best_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(len(test_set)):
            data_batch = test_set[i]

            label_batch = test_labels[i]
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)
            output = model(data_batch)
            pred = output.argmax(dim=1, keepdim=True)
            loss = classification_loss(output, label_batch)
            losses.append(loss.item())

            correct += (pred == label_batch.reshape(label_batch.shape[0], -1)).sum().item()
        test_loss = float(np.mean(losses))
        accuracy = 100. * correct / len(test_set)
        # if this accuracy is better than the highest one on record, save the weights that achieved this accuracy, and
        # overwrite the old best accuracy
        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            torch.save(deepcopy(model.state_dict()), 'C:/Users/antle/PycharmProjects/course_project/best_params.txt')
        outputstring = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_set), accuracy)
        print(outputstring)
        return test_loss, accuracy


# batch data into the designated number of samples per batch
# need to convert to tensor for faster training
def batch_data(dataset, batch_size):
    tensorize = transforms.ToTensor()
    batch_count = math.floor(len(dataset) / batch_size)
    batched_data = []
    for i in range(batch_count):
        sample = dataset[i * batch_size: batch_size * (i + 1)]
        batched_data.append(torch.stack(sample))

    return batched_data


def run_model(device):
    df = pd.read_csv('C:/Users/antle/OneDrive/Documents/fer2013/fer2013.csv')
    train_set, train_labels, test_set, test_labels = [], [], [], []
    transform = transforms.ToTensor()
    # manually splitting my data into training and test since I had issues with the pytorch and deeplake data loaders
    for idx, row in df.iterrows():
        pixels_as_array = transform(np.array(row['pixels'].split(' '), 'float32').reshape(48, -1))
        if row['Usage'] == 'Training':
            train_set.append(pixels_as_array)
            train_labels.append(torch.tensor(row['emotion']))
        if row['Usage'] == 'PublicTest':
            test_set.append(pixels_as_array)
            test_labels.append(torch.tensor(row['emotion']))
    classification_loss = nn.CrossEntropyLoss()
    FER = FER_Model_v2().to(device)
    optimizer = optim.SGD(FER.parameters(), lr=.01)
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # train/evaluate model on batched versions of datasets generated in the previous code
    for epoch in range(0, 50):
        train_loss, train_accuracy = train_model(FER, batch_data(train_set, 19), batch_data(train_labels, 19),
                                                 classification_loss,
                                                 optimizer, device)
        test_loss, test_accuracy = test_model(FER, batch_data(test_set, 1), batch_data(test_labels, 1),
                                              classification_loss,
                                              device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    # display graphs for accuracy and loss
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.plot(range(0, 50), train_accuracies, "y", label='Training curve')
    ax1.plot(range(0, 50), test_accuracies, "b", label='Testing curve')
    ax1.legend(loc='lower left')
    plt.show()
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.plot(range(0, 50), train_losses, "y", label='Model 1 Training curve')
    ax2.plot(range(0, 50), test_losses, "b", label='Model 1 Testing curve')
    ax2.legend(loc='lower left')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--needs_training',
                        type=bool, default=True,
                        help='Set to true if you need to generate parameters for the model')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load face detector
    detector = MTCNN()
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    # re-train model if flag is set to true; otherwise the model will be loaded using the best weights from a previous training session
    if FLAGS.needs_training:
        run_model(device)
    cap = cv2.VideoCapture(0)
    emotion_model = FER_Model_v2()
    emotion_model = emotion_model.to(device)
    # l oad parameters from previous session
    emotion_model.load_state_dict(torch.load('C:/Users/antle/PycharmProjects/course_project/best_params.txt'))
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    while True:
        captured, image = cap.read()
        detection_result = detector.detect_faces(image)
        # if a face is detected, process the region within the bounding box into a 48 x 48 grayscale image and feed it
        # into the model
        if len(detection_result) == 1:
            bb = detection_result[0]['box']
            image = cv2.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (255, 0, 0), 2)
            face = image[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
            emotion_capture = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
            emotion_capture = cv2.cvtColor(emotion_capture, cv2.COLOR_BGR2GRAY)
            emotion_capture = torch.FloatTensor(emotion_capture).to(device)
            emotion_capture = torch.reshape(emotion_capture, (1, 1, 48, 48))
            pred, indexes = torch.sort(emotion_model(emotion_capture), 1, descending=True)
            # Get the top 3 emotion predictions in descending order and add them to the video capture
            top_preds = pred[0][:3]
            pred_string = '/'.join(
                ['{} : {:.2f}'.format(emotions[indexes[0][x]], top_preds[x]) for x in range(len(top_preds))])
            image = cv2.putText(image, pred_string, (bb[0] + bb[2], bb[1] + bb[3]), cv2.FONT_HERSHEY_PLAIN, .8,
                                (0, 255, 0), 2)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break
    cap.release()
    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
