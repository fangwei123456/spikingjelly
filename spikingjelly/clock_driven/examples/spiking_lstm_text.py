import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven import rnn
# from torch.utils.tensorboard import SummaryWriter
# import sys
# if sys.platform != 'win32':
#     import readline
# import torchvision
# import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import glob
import unicodedata
import string
import random
import time
import math


####################
# prepare the data
####################
if __name__ == '__main__':

    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters)

    def findFiles(path):
        return glob.glob(path)

    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename).read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]

    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []

    # Ubuntu
    for filename in findFiles('./data/names/*.txt'):  # Windows findFiles('.\data\\names\*.txt')
        category = filename.split('/')[-1].split('.')[0]  # Windows filename.split('\\')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    # Windows
    # for filename in findFiles('.\data\\names\*.txt'):  # Windows findFiles('.\data\\names\*.txt')
    #     category = filename.split('\\')[-1].split('.')[0]  # Windows filename.split('\\')[-1].split('.')[0]
    #     all_categories.append(category)
    #     lines = readLines(filename)
    #     category_lines[category] = lines

    n_categories = len(all_categories)


    # split the data into training set and testing set
    numExamplesPerCategory = []
    category_lines_train = {}
    category_lines_test = {}
    testNumtot = 0
    for c, names in category_lines.items():
        category_lines_train[c] = names[:int(len(names)*0.8)]
        category_lines_test[c] = names[int(len(names)*0.8):]
        numExamplesPerCategory.append([len(category_lines[c]), len(category_lines_train[c]), len(category_lines_test[c])])
        testNumtot += len(category_lines_test[c])


    # Find letter index from all_letters, e.g. "a" = 0
    def letterToIndex(letter):
        return all_letters.find(letter)

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][letterToIndex(letter)] = 1
        return tensor

    def categoryFromOutput(output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0][0]
        return all_categories[category_i], category_i

    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    # Preparing [x, y] pair
    def randomPair(sampleSource):
        """
        Args:
            sampleSource:  'train', 'test', 'all'
        Returns:
            category, line, category_tensor, line_tensor
        """
        category = randomChoice(all_categories)
        if sampleSource == 'train':
            line = randomChoice(category_lines_train[category])
        elif sampleSource == 'test':
            line = randomChoice(category_lines_test[category])
        elif sampleSource == 'all':
            line = randomChoice(category_lines[category])
        category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.float)
        line_tensor = lineToTensor(line)
        return category, line, category_tensor, line_tensor

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    ####################
    # prepare the net
    ####################

    n_hidden = 256  # 256   # 128

    class Net(nn.Module):
        def __init__(self, n_letters, n_hidden, n_categories):
            super().__init__()
            self.n_input = n_letters
            self.n_hidden = n_hidden
            self.n_out = n_categories
            self.lstm = rnn.SpikingLSTM(self.n_input, self.n_hidden, 1)
            self.fc = nn.Linear(self.n_hidden, self.n_out)
            # self.softmax = nn.LogSoftmax()

        def forward(self, x):
            x, _ = self.lstm(x)
            output = self.fc(x[-1])
            output = F.softmax(output, dim=1)
            return output


    ####################
    # training and testing
    ####################

    IF_TRAIN = 0
    TRAIN_EPISODES = 1000000  # 100000
    # TEST_EPISODES = 50

    # print_every = 5000
    plot_every = 1000
    learning_rate = 1e-4  # 0.001  # 0.005 # If you set this too high, it might explode. If too low, it might not learn

    net = Net(n_letters, n_hidden, n_categories)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    if IF_TRAIN:
        print('Training...')
        current_loss = 0
        correct_num = 0
        avg_losses = []
        accuracy_rec = []
        # all_losses = []
        test_accu_rec = []
        start = time.time()
        for epoch in range(1, TRAIN_EPISODES+1):
            net.train()
            category, line, category_tensor, line_tensor = randomPair('train')
            label_one_hot = F.one_hot(category_tensor.to(int), n_categories).float()

            optimizer.zero_grad()
            out_prob_log = net(line_tensor)
            # loss = nn.NLLLoss(out_prob_log, category_tensor)
            loss = F.mse_loss(out_prob_log, label_one_hot)
            loss.backward()
            optimizer.step()

            current_loss += loss.data.item()
            # all_losses.append(loss.data.item())

            guess, _ = categoryFromOutput(out_prob_log.data)
            if guess == category:
                correct_num += 1

            # Add current loss avg to list of losses
            if epoch % plot_every == 0:
                avg_losses.append(current_loss / plot_every)
                accuracy_rec.append(correct_num / plot_every)
                current_loss = 0
                correct_num = 0

            # 每训练一定次数即进行一次测试
            if epoch % plot_every == 0:  # int(TRAIN_EPISODES/1000)
                net.eval()
                with torch.no_grad():
                    numCorrect = 0
                    for i in range(n_categories):
                        category = all_categories[i]
                        for tname in category_lines_test[category]:
                            output = net(lineToTensor(tname))
                            guess, _ = categoryFromOutput(output.data)
                            if guess == category:
                                numCorrect += 1
                    test_accu = numCorrect / testNumtot
                    test_accu_rec.append(test_accu)
                    print('Epoch %d %d%% (%s); Avg_loss %.4f; Train accuracy %.4f; Test accuracy %.4f' % (
                        epoch, epoch / TRAIN_EPISODES * 100, timeSince(start), avg_losses[-1], accuracy_rec[-1], test_accu))

        torch.save(net, 'char_rnn_classification.pth')
        np.save('avg_losses.npy', np.array(avg_losses))
        np.save('accuracy_rec.npy', np.array(accuracy_rec))
        np.save('test_accu_rec.npy', np.array(test_accu_rec))
        np.save('category_lines_train.npy', category_lines_train, allow_pickle=True)
        np.save('category_lines_test.npy', category_lines_test, allow_pickle=True)
        # x = np.load('category_lines_test.npy', allow_pickle=True)
        # xdict = x.item()

        plt.figure()
        plt.subplot(311)
        plt.plot(avg_losses)
        plt.title('Average loss')
        plt.subplot(312)
        plt.plot(accuracy_rec)
        plt.title('Train accuracy')
        plt.subplot(313)
        plt.plot(test_accu_rec)
        plt.title('Test accuracy')
        plt.xlabel('Epoch (*1000)')
        plt.subplots_adjust(hspace=0.6)
        plt.savefig('TrainingProcess.svg')
        plt.close()

    else:
        print('Testing...')

        net = torch.load('char_rnn_classification.pth')

        # 遍历测试集计算准确率
        print('Calculating testing accuracy...')
        numCorrect = 0
        for i in range(n_categories):
            category = all_categories[i]
            for tname in category_lines_test[category]:
                output = net(lineToTensor(tname))
                guess, _ = categoryFromOutput(output.data)
                if guess == category:
                    numCorrect += 1
        test_accu = numCorrect / testNumtot
        print('Test accuracy: {:.3f}, Random guess: {:.3f}'.format(test_accu, 1/n_categories))
        plt.figure()
        plt.bar(1, test_accu)
        plt.xlim(0, 5)
        plt.ylim(0, 1)
        plt.title('Test accuracy: {:.3f}, Random guess: {:.3f}'.format(test_accu, 1/n_categories))
        plt.show()
        plt.savefig('TestAccuracy.png')
        plt.close()

        # 让用户输入姓氏以判断其属于哪种语系
        n_predictions = 3
        for j in range(3):
            first_name = input('请输入一个姓氏以判断其属于哪种语系：')
            print('\n> %s' % first_name)
            output = net(lineToTensor(first_name))
            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])

        # 计算confusion矩阵
        print('Calculating confusion matrix...')
        confusion = torch.zeros(n_categories, n_categories)
        n_confusion = 10000

        # Keep track of correct guesses in a confusion matrix
        for i in range(n_confusion):
            category, line, category_tensor, line_tensor = randomPair('all')
            output = net(line_tensor)
            guess, guess_i = categoryFromOutput(output.data)
            category_i = all_categories.index(category)
            confusion[category_i][guess_i] += 1

        confusion = confusion / confusion.sum(1)
        np.save('confusion.npy', confusion)

        # Set up plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion.numpy())
        fig.colorbar(cax)
        # Set up axes
        ax.set_xticklabels([''] + all_categories, rotation=90)
        ax.set_yticklabels([''] + all_categories)
        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # sphinx_gallery_thumbnail_number = 2
        plt.show()
        plt.savefig('ConfusionMatrix.svg')
        plt.close()
