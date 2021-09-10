Classifying Names with a Character-level Spiking LSTM
==============================================================================
Authors: `LiutaoYu <https://github.com/LiutaoYu>`_, `fangwei123456 <https://github.com/fangwei123456>`_

This tutorial applies a Spiking LSTM to reproduce the PyTorch official tutorial `NLP From Scratch: Classifying Names with a Character-Level RNN <https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html>`_.
Please make sure that you have read the original tutorial and corresponding codes before proceeding.
Specifically, we will train a spiking LSTM to classify surnames into different languages according to their spelling, based on a dataset consisting of several thousands of surnames from 18 languages of origin.
The integrated script can be found here ( `clock_driven/examples/spiking_lstm_text.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/spiking_lstm_text.py>`_).

Preparing the data
----------------------------
First of all, we need to download and preprocess the data as the original tutorial, which produces a dictionary ``{language: [names ...]}`` .
Then, we split the dataset into a training set and a testing set (the ratio is 4:1), i.e.,  ``category_lines_train`` and ``category_lines_test`` .
Here, we emphasize several important variables: ``all_categories`` is the list of 18 languages, the length of which is ``n_categories=18``;
``n_letters=58`` is the number of all characters composing the surnames.

.. code-block:: python

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

In addition, we rephrase the function ``randomTrainingExample()``  to function ``randomPair(sampleSource)``  for different conditions.
Here we adopt function ``lineToTensor()`` and ``randomChoice()`` from the original tutorial.
``lineToTensor()`` converts a surname into a one-hot tensor, and ``randomChoice()`` randomly choose a sample from the dataset.

.. code-block:: python

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

Building a spiking LSTM network
--------------------------------------
We build a spiking LSTM based on the ``rnn`` module from  `spikingjelly <https://github.com/fangwei123456/spikingjelly>`_ .
The theory can be found in the paper  `Long Short-Term Memory Spiking Networks and Their Applications <https://arxiv.org/abs/2007.04779>`_ .
The amounts of neurons in the input layer, hidden layer and output layer are ``n_letters``, ``n_hidden`` and ``n_categories`` respectively.
We add a fully connected layer to the output layer, and use ``softmax`` function to obtain the classification probability.

.. code-block:: python

    from spikingjelly.clock_driven import rnn
    n_hidden = 256

    class Net(nn.Module):
        def __init__(self, n_letters, n_hidden, n_categories):
            super().__init__()
            self.n_input = n_letters
            self.n_hidden = n_hidden
            self.n_out = n_categories
            self.lstm = rnn.SpikingLSTM(self.n_input, self.n_hidden, 1)
            self.fc = nn.Linear(self.n_hidden, self.n_out)

        def forward(self, x):
            x, _ = self.lstm(x)
            output = self.fc(x[-1])
            output = F.softmax(output, dim=1)
            return output

Training the network
---------------------------------------
First of all, we initialize the ``net`` , and define parameters like ``TRAIN_EPISODES`` and ``learning_rate``.
Here we adopt ``mse_loss`` and ``Adam`` optimizer to train the network.
The process of one training epoch is as follows:
1) randomly choose a sample from the training set, and convert the input and label into tensors;
2) feed the input to the network, and obtain the classification probability through the forward process;
3) calculate the network loss through ``mse_loss``;
4) back-propagate the gradients, and update the training parameters;
5) judge whether the prediction is correct or not, and count the number of correct predictions to obtain the training accuracy every ``plot_every`` epochs;
6) evaluate the network on the testing set every ``plot_every`` epochs to obtain the testing accuracy.
During training, we record the history of network loss ``avg_losses`` , training accuracy ``accuracy_rec`` and testing accuracy ``test_accu_rec`` , to observe the training process.
After training, we will save the final state of the network for testing, and also some variables for later analyses.

.. code-block:: python

    # IF_TRAIN = 1
    TRAIN_EPISODES = 1000000
    plot_every = 1000
    learning_rate = 1e-4

    net = Net(n_letters, n_hidden, n_categories)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    print('Training...')
    current_loss = 0
    correct_num = 0
    avg_losses = []
    accuracy_rec = []
    test_accu_rec = []
    start = time.time()
    for epoch in range(1, TRAIN_EPISODES+1):
        net.train()
        category, line, category_tensor, line_tensor = randomPair('train')
        label_one_hot = F.one_hot(category_tensor.to(int), n_categories).float()

        optimizer.zero_grad()
        out_prob_log = net(line_tensor)
        loss = F.mse_loss(out_prob_log, label_one_hot)
        loss.backward()
        optimizer.step()

        current_loss += loss.data.item()

        guess, _ = categoryFromOutput(out_prob_log.data)
        if guess == category:
            correct_num += 1

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            avg_losses.append(current_loss / plot_every)
            accuracy_rec.append(correct_num / plot_every)
            current_loss = 0
            correct_num = 0

        # evaluate the network on the testing set every ``plot_every`` epochs to obtain the testing accuracy
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
    # x = np.load('category_lines_test.npy', allow_pickle=True)  # way to loading the data
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

We will observe the following results when executing ``%run ./spiking_lstm_text.py`` in Python Console with ``IF_TRAIN = 1`` .

.. code-block:: shell

    Backend Qt5Agg is interactive backend. Turning interactive mode on.
    Training...
    Epoch 1000 0% (0m 18s); Avg_loss 0.0525; Train accuracy 0.0830; Test accuracy 0.0806
    Epoch 2000 0% (0m 37s); Avg_loss 0.0514; Train accuracy 0.1470; Test accuracy 0.1930
    Epoch 3000 0% (0m 55s); Avg_loss 0.0503; Train accuracy 0.1650; Test accuracy 0.0537
    Epoch 4000 0% (1m 14s); Avg_loss 0.0494; Train accuracy 0.1920; Test accuracy 0.0938
    ...
    ...
    Epoch 998000 99% (318m 54s); Avg_loss 0.0063; Train accuracy 0.9300; Test accuracy 0.5036
    Epoch 999000 99% (319m 14s); Avg_loss 0.0056; Train accuracy 0.9380; Test accuracy 0.5004
    Epoch 1000000 100% (319m 33s); Avg_loss 0.0055; Train accuracy 0.9340; Test accuracy 0.5118

The following picture shows how average loss ``avg_losses`` , training accuracy ``accuracy_rec`` and testing accuracy ``test_accu_rec`` improve with training.

.. image:: ../_static/tutorials/clock_driven/\9_spikingLSTM_text/TrainingProcess.*
    :width: 100%

Testing the network
---------------------------
We first load the well-trained network, and then conduct the following tests:
1) calculate the testing accuracy of the final network;
2) predict the language origin of the surnames provided by the user;
3) calculate the confusion matrix, indicating for every actual language (rows) which language the network guesses (columns).

.. code-block:: python

    # IF_TRAIN = 0
    print('Testing...')

    net = torch.load('char_rnn_classification.pth')

    # calculate the testing accuracy of the final network
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

    # predict the language origin of the surnames provided by the user
    n_predictions = 3
    for j in range(3):
        first_name = input('Please input a surname to predict its language origin:')
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

    # calculate the confusion matrix
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

We will observe the following results when executing ``%run ./spiking_lstm_text.py`` in Python Console with ``IF_TRAIN = 0`` .

.. code-block:: shell

    Testing...
    Calculating testing accuracy...
    Test accuracy: 0.512, Random guess: 0.056
    Please input a surname to predict its language origin:> YU
    > YU
    (0.18) Scottish
    (0.12) English
    (0.11) Italian
    Please input a surname to predict its language origin:> Yu
    > Yu
    (0.63) Chinese
    (0.23) Korean
    (0.07) Vietnamese
    Please input a surname to predict its language origin:> Zou
    > Zou
    (1.00) Chinese
    (0.00) Arabic
    (0.00) Polish
    Calculating confusion matrix...

The following picture exhibits the confusion matrix, of which a brighter diagonal element indicates better prediction, and thus less confusion, such as Arabic and Greek.
However, some languages are prone to confusion, such as Korean and Chinese, English and Scottish.

.. image:: ../_static/tutorials/clock_driven/\9_spikingLSTM_text/ConfusionMatrix.*
    :width: 100%
