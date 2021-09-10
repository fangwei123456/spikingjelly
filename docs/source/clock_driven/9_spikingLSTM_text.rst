利用Spiking LSTM实现基于文本的姓氏分类任务
==============================================================================
本教程作者：`LiutaoYu <https://github.com/LiutaoYu>`_，`fangwei123456 <https://github.com/fangwei123456>`_

本节教程使用Spiking LSTM重新实现PyTorch的官方教程 `NLP From Scratch: Classifying Names with a Character-Level RNN <https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html>`_。
对应的中文版教程可参见 `使用字符级别特征的RNN网络进行名字分类 <https://pytorch.apachecn.org/docs/1.0/char_rnn_classification_tutorial.html>`_。
请确保你已经阅读了原版教程和代码，因为本教程是对原教程的扩展。本教程将构建和训练字符级的Spiking LSTM来对姓氏进行分类。
具体而言，本教程将在18种语言构成的几千个姓氏的数据集上训练Spiking LSTM模型，网络可根据一个姓氏的拼写预测其属于哪种语言。
完整代码可见于 `clock_driven/examples/spiking_lstm_text.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/spiking_lstm_text.py>`_。

准备数据
------------------------
首先，我们参照原教程下载数据，并进行预处理。预处理后，我们可以得到一个语言对应姓氏列表的字典，即 ``{language: [names ...]}`` 。
进一步地，我们将数据集按照4:1的比例划分为训练集和测试集，即 ``category_lines_train`` 和 ``category_lines_test`` 。
这里还需要留意几个后续会经常使用的变量： ``all_categories`` 是全部语言种类的列表， ``n_categories=18`` 则是语言种类的数量，
``n_letters=58`` 是组成 ``names`` 的所有字母和符号的集合的元素数量。

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

此外，我们改写了原教程中的 ``randomTrainingExample()`` 函数，以便在不同条件下进行使用。
注意此处利用了原教程中定义的 ``lineToTensor()`` 和 ``randomChoice()`` 两个函数。
前者用于将单词转化为one-hot张量，后者用于从数据集中随机抽取一个样本。

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

构造Spiking LSTM神经网络
---------------------------
我们利用 `spikingjelly <https://github.com/fangwei123456/spikingjelly>`_ 中的rnn模块( ``rnn.SpikingLSTM()`` )来搭建Spiking LSTM神经网络。
其工作原理可参见论文 `Long Short-Term Memory Spiking Networks and Their Applications <https://arxiv.org/abs/2007.04779>`_ 。
输入层神经元个数等于 ``n_letters`` ，隐藏层神经元个数 ``n_hidden`` 可自行定义，输出层神经元个数等于 ``n_categories`` 。
我们在LSTM的输出层之后接一个全连接层，并利用softmax函数对全连接层的数据进行处理以获取类别概率。

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

网络训练
---------------------------
首先，我们初始化网络 ``net`` ，并定义训练时长 ``TRAIN_EPISODES`` 、学习率 ``learning_rate`` 等。
这里我们采用 ``mse_loss`` 损失函数和 ``Adam`` 优化器来对训练网络。
单个epoch的训练流程大致如下：1）从训练集中随机抽取一个样本，获取输入和标签，并转化为tensor；2）网络接收输入，进行前向过程，获取各类别的预测概率；
3）利用 ``mse_loss`` 函数获取网络预测概率和真实标签one-hot编码之间的差距，即网络损失；4）梯度反传，并更新网络参数；
5）判断此次预测是否正确，并累计预测正确的数量，以获取模型在训练过程中针对训练集的准确率（每隔 ``plot_every`` 个epoch计算一次）；
6）每隔 ``plot_every`` 个epoch在测试集上测试一次，并统计测试准确率。
此外，在训练过程中，我们会记录网络损失 ``avg_losses`` 、训练集准确率 ``accuracy_rec`` 和测试集准确率 ``test_accu_rec`` ，以便于观察训练效果，并在训练完成后绘制图片。
在训练完成之后，我们会保存网络的最终状态以用于测试；同时，也可以保存相关变量，以便后续分析。

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

        # 优化一次参数后，需要重置网络的状态。是否需要？结果差别不明显！(2020.11.3)
        # functional.reset_net(net)

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

        # 每训练一定次数即进行一次测试
        if epoch % plot_every == 0:  # int(TRAIN_EPISODES/1000)
            net.eval()
            with torch.no_grad():
                numCorrect = 0
                for i in range(n_categories):
                    category = all_categories[i]
                    for tname in category_lines_test[category]:
                        output = net(lineToTensor(tname))
                        # 运行一次后，需要重置网络的状态。是否需要？
                        # functional.reset_net(net)
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
    # x = np.load('category_lines_test.npy', allow_pickle=True)  # 读取数据的方法
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

设定 ``IF_TRAIN = 1`` ，在Python Console中运行 ``%run ./spiking_lstm_text.py`` ，输出如下：

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

下图展示了训练过程中损失函数、测试集准确率、测试集准确率随时间的变化。
值得注意的一点是，测试表明，在当前Spiking LSTM网络中是否在一次运行完成后重置网络 ``functional.reset_net(net)`` 对于结果没有显著的影响。
我们猜测是因为当前网络输入是随时间变化的，而且网络自身需要运行一段时间后才会输出分类结果，因此网络初始状态影响不显著。

.. image:: ../_static/tutorials/clock_driven/\9_spikingLSTM_text/TrainingProcess.*
    :width: 100%

网络测试
---------------------------
在测试过程中，我们首先需要导入训练完成后存储的网络，随后进行三方面的测试：（1）计算最终的测试集准确率；（2）让用户输入姓氏拼写以预测其属于哪种语言；
（3）计算Confusion matrix，每一行表示当样本源于某一个类别时，网络预测其属于各类别的概率，即对角线表示预测正确的概率。

.. code-block:: python

    # IF_TRAIN = 0
    print('Testing...')

    net = torch.load('char_rnn_classification.pth')

    # 遍历测试集计算准确率
    print('Calculating testing accuracy...')
    numCorrect = 0
    for i in range(n_categories):
        category = all_categories[i]
        for tname in category_lines_test[category]:
            output = net(lineToTensor(tname))
            # 运行一次后，需要重置网络的状态。是否需要？
            # functional.reset_net(net)
            guess, _ = categoryFromOutput(output.data)
            if guess == category:
                numCorrect += 1
    test_accu = numCorrect / testNumtot
    print('Test accuracy: {:.3f}, Random guess: {:.3f}'.format(test_accu, 1/n_categories))

    # 让用户输入姓氏以判断其属于哪种语系
    n_predictions = 3
    for j in range(3):
        first_name = input('请输入一个姓氏以判断其属于哪种语系：')
        print('\n> %s' % first_name)
        output = net(lineToTensor(first_name))
        # 运行一次后，需要重置网络的状态。是否需要？
        # functional.reset_net(net)
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
        # 运行一次后，需要重置网络的状态。是否需要？
        # functional.reset_net(net)
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

设定 ``IF_TRAIN = 0``，在Python Console中运行 ``%run ./spiking_lstm_text.py``，输出如下：

.. code-block:: shell

    Testing...
    Calculating testing accuracy...
    Test accuracy: 0.512, Random guess: 0.056
    请输入一个姓氏以判断其属于哪种语系:> YU
    > YU
    (0.18) Scottish
    (0.12) English
    (0.11) Italian
    请输入一个姓氏以判断其属于哪种语系:> Yu
    > Yu
    (0.63) Chinese
    (0.23) Korean
    (0.07) Vietnamese
    请输入一个姓氏以判断其属于哪种语系:> Zou
    > Zou
    (1.00) Chinese
    (0.00) Arabic
    (0.00) Polish
    Calculating confusion matrix...

下图展示了Confusion matrix。对角线越亮，表示模型对某一类别预测最好，很少产生混淆，如Arabic和Greek。
而有的语言则较容易产生混淆，如Korean和Chinese，Spanish和Portuguese，English和Scottish。

.. image:: ../_static/tutorials/clock_driven/\9_spikingLSTM_text/ConfusionMatrix.*
    :width: 100%
