神经形态数据集处理
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

``spikingjelly.datasets`` 中集成了常用的神经形态数据集，包括 N-MNIST [#NMNIST]_, CIFAR10-DVS [#CIFAR10DVS]_, DVS128 Gesture [#DVS128Gesture]_, NAV Gesture [#NAVGesture]_, ASLDVS [#ASLDVS]_ 等。在本节教程中，我们将以 DVS128 Gesture 为例，展示如何使用惊蜇框架处理神经形态数据集。

下载DVS128 Gesture
-----------------------
DVS128 Gesture数据集可以从 https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794 进行下载。box网站不支持在不登陆的情况下使用代码直接下载，因此用户需要手动从网站上下载。将数据集下载到了 ``E:/datasets/DVS128Gesture``，下载完成后这个文件夹的目录结构为

.. code:: bash

    .
    |-- DvsGesture.tar.gz
    |-- LICENSE.txt
    |-- README.txt
    `-- gesture_mapping.csv


获取Event数据
-----------------------
导入惊蜇框架的DVS128 Gesture的模块，创建训练集和测试集，其中参数 ``use_frame=False`` 表示我们不使用帧数据，而是使用Event数据。

.. code:: python

    from spikingjelly.datasets import DVS128Gesture

    root_dir = 'E:/datasets/DVS128Gesture'
    train_set = DVS128Gesture(root_dir, train=True, use_frame=False)
    test_set = DVS128Gesture(root_dir, train=True, use_frame=False)

运行这段代码，惊蜇框架将会完成以下工作：

#. 检测数据集是否存在，如果存在，则进行MD5校验，确认数据集无误后，开始进行解压。将原始数据解压到同级目录下的 ``extracted`` 文件夹
#. DVS128 Gesture中的每个样本，是在不同光照环境下，对不同表演者进行录制的手势视频。一个AER文件中包含了多个手势，对应的会有一个csv文件来标注
整个视频内各个时间段内都是哪种手势。因此，单个的视频文件并不是一个类别，而是多个类别的集合。惊蜇框架会启动多线程进行划分，将每个视频中的每个手势类别文件单独提取出来

下面是运行过程中的命令行输出：

.. code:: bash

    DvsGesture.tar.gz already exists, check md5
    C:/Users/fw/anaconda3/envs/pytorch-env/lib/site-packages/torchaudio/backend/utils.py:88: UserWarning: No audio backend is available.
      warnings.warn('No audio backend is available.')
    md5 checked, extracting...
    mkdir E:/datasets/DVS128Gesture/events_npy
    mkdir E:/datasets/DVS128Gesture/events_npy/train
    mkdir E:/datasets/DVS128Gesture/events_npy/test
    read events data from *.aedat and save to *.npy...
    convert events data from aedat to numpy format.
    thread 0 start
    thread 1 start
    thread 2 start
    thread 3 start
    thread 4 start
    thread 5 start
    thread 6 start
    thread 7 start
      0%|          | 0/122 [00:00<?, ?it/s]wroking thread: [0, 1, 2, 3, 4, 5, 6, 7]
    finished thread: []

提取各个手势类别的速度较慢，需要耐心等待。运行完成后，同级目录下会多出一个 ``events_npy`` 文件夹，其中包含训练集和测试集：

.. code:: bash

    |-- events_npy
    |   |-- test
    |   `-- train

打印一个数据：

.. code:: python

    x, y = train_set[0]
    print('event', x)
    print('label', y)

得到输出为：

.. code:: bash

    event {'t': array([172843814, 172843824, 172843829, ..., 179442748, 179442763,
           179442789]), 'x': array([ 54,  59,  53, ...,  36, 118, 118]), 'y': array([116, 113,  92, ..., 102,  80,  83]), 'p': array([0, 1, 1, ..., 0, 1, 1])}
    label 9

其中 ``x`` 使用字典格式存储Events数据，键为 ``['t', 'x', 'y', 'p']``；``y`` 是数据的标签，DVS128 Gesture共有11类。

获取Frame数据
-----------------------
将原始的Event流积分成Frame数据，是常用的处理方法，我们采用 [#PLIF]_ 的实现方式。。我们将原始的Event数据记为 :math:`E(x_{i}, y_{i}, t_{i}, p_{i}), 0 \leq i \le N`；设置 ``split_by='number'`` 表示从Event数量 :math:`N` 上进行划分，接近均匀地划分为 ``frames_num=20``， 也就是 :math:`T` 段。记积分后的Frame数据中的某一帧
为 :math:`F(j)`，在 :math`(p, x, y)` 位置的像素值为 :math:`F(j, p, x, y)`；math:`F(j)` 是从Event流中索引介于 :math:`j_{l}` 和 :math:`j_{r}` 的Event
积分而来：

.. math::

    j_{l} & = \left\lfloor \frac{N}{T}\right \rfloor \cdot j \\
	j_{r} & = \begin{cases} \left \lfloor \frac{N}{T} \right \rfloor \cdot (j + 1), & \text{if}~~ j <  T - 1 \cr N, &  \text{if} ~~j = T - 1 \end{cases}\\
    F(j, p, x, y) &= \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})

其中 :math:`\lfloor \cdot \rfloor` 是向下取整，:math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` 是示性函数，当且仅当 :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})` 时取值为1，否则为0。

运行下列代码，惊蜇框架就会开始进行积分，创建Frame数据集：

.. code:: python

    train_set = DVS128Gesture(root_dir, train=True, use_frame=True, frames_num=20, split_by='number', normalization=None)
    test_set = DVS128Gesture(root_dir, train=True, use_frame=True, frames_num=20, split_by='number', normalization=None)

命令行的输出为：

.. code:: bash

    npy format events data root E:/datasets/DVS128Gesture/events_npy/train, E:/datasets/DVS128Gesture/events_npy/test already exists
    mkdir E:/datasets/DVS128Gesture/frames_num_20_split_by_number_normalization_None, E:/datasets/DVS128Gesture/frames_num_20_split_by_number_normalization_None/train, E:/datasets/DVS128Gesture/frames_num_20_split_by_number_normalization_None/test.
    creating frames data..
    thread 0 start, processing files index: 0 : 294.
    thread 1 start, processing files index: 294 : 588.
    thread 2 start, processing files index: 588 : 882.
    thread 4 start, processing files index: 882 : 1176.
    thread 0 finished.
    thread 1 finished.
    thread 2 finished.
    thread 3 finished.
    thread 0 start, processing files index: 0 : 72.
    thread 1 start, processing files index: 72 : 144.
    thread 2 start, processing files index: 144 : 216.
    thread 4 start, processing files index: 216 : 288.
    thread 0 finished.
    thread 1 finished.
    thread 2 finished.
    thread 3 finished.

运行后，同级目录下会出现 ``frames_num_20_split_by_number_normalization_None`` 文件夹，这里存放了积分生成的Frame数据。

打印一个数据：

.. code:: python

    x, y = train_set[0]
    x, y = train_set[0]
    print('frame shape', x.shape)
    print('label', y)

得到输出为：

.. code:: bash

    frame shape torch.Size([20, 2, 128, 128])
    label 9

查看1个积分好的Frame数据：

.. code:: python

    from torchvision import transforms
    from matplotlib import pyplot as plt

    x, y = train_set[5]
    to_img = transforms.ToPILImage()

    img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    img_tensor[:, 1] = x[:, 0]
    img_tensor[:, 2] = x[:, 1]


    for t in range(img_tensor.shape[0]):
        print(t)
        plt.imshow(to_img(img_tensor[t]))
        plt.pause(0.01)

显示效果如下图所示：

.. image:: ../_static/tutorials/clock_driven/13_neuromorphic_datasets/dvsg.gif
    :width: 100%

.. [#NMNIST] Orchard, Garrick, et al. “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades.” Frontiers in Neuroscience, vol. 9, 2015, pp. 437–437.


.. [#CIFAR10DVS] Li, Hongmin, et al. “CIFAR10-DVS: An Event-Stream Dataset for Object Classification.” Frontiers in Neuroscience, vol. 11, 2017, pp. 309–309.


.. [#DVS128Gesture] Amir, Arnon, et al. “A Low Power, Fully Event-Based Gesture Recognition System.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 7388–7397.

.. [#NAVGesture] Maro, Jean-Matthieu, et al. “Event-Based Visual Gesture Recognition with Background Suppression Running on a Smart-Phone.” 2019 14th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2019), 2019, p. 1.

.. [#ASLDVS] Bi, Yin, et al. “Graph-Based Object Classification for Neuromorphic Vision Sensing.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 491–501.

.. [#PLIF] Fang, Wei, et al. “Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks.” ArXiv: Neural and Evolutionary Computing, 2020.