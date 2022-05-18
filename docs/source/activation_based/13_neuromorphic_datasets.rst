神经形态数据集处理
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

``spikingjelly.datasets`` 中集成了常用的神经形态数据集，包括 N-MNIST [#NMNIST]_, CIFAR10-DVS [#CIFAR10DVS]_, DVS128 Gesture [#DVS128Gesture]_, N-Caltech101 [#NMNIST]_, ASLDVS [#ASLDVS]_ 等。所有数据集的处理都遵循类似的步骤，开发人员也可以很轻松的添加新数据集代码。在本节教程中，我
们将以 DVS128 Gesture 为例，展示如何使用惊蜇框架处理神经形态数据集。

自动下载和手动下载
-----------------------
CIFAR10-DVS等数据集支持自动下载。支持自动下载的数据集，在首次运行时原始数据集将会被下载到数据集根目录下的 ``download`` 文件夹。每个数据集的 ``downloadable()``
函数定义了该数据集是否能够自动下载，而 ``resource_url_md5()`` 函数定义了各个文件的下载链接和MD5。示例：

.. code:: python

    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    print('CIFAR10-DVS downloadable', CIFAR10DVS.downloadable())
    print('resource, url, md5/n', CIFAR10DVS.resource_url_md5())

    print('DVS128Gesture downloadable', DVS128Gesture.downloadable())
    print('resource, url, md5/n', DVS128Gesture.resource_url_md5())

输出为：

.. code:: bash

    CIFAR10-DVS downloadable True
    resource, url, md5
     [('airplane.zip', 'https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'), ('automobile.zip', 'https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'), ('bird.zip', 'https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'), ('cat.zip', 'https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'), ('deer.zip', 'https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'), ('dog.zip', 'https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'), ('frog.zip', 'https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'), ('horse.zip', 'https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'), ('ship.zip', 'https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'), ('truck.zip', 'https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')]
    DVS128Gesture downloadable False
    resource, url, md5
     [('DvsGesture.tar.gz', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '8a5c71fb11e24e5ca5b11866ca6c00a1'), ('gesture_mapping.csv', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '109b2ae64a0e1f3ef535b18ad7367fd1'), ('LICENSE.txt', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '065e10099753156f18f51941e6e44b66'), ('README.txt', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', 'a0663d3b1d8307c329a43d949ee32d19')]

DVS128 Gesture数据集不支持自动下载，但它的 ``resource_url_md5()`` 函数会打印出获取下载地址的网址。DVS128 Gesture数据集可以从 https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794 进行下载。box网站不支持在不登陆的情况下使用代码直接下载，因此用户需要手动从网站上下载。将数据集下载到 ``E:/datasets/DVS128Gesture/download``，下载完成后这个文件夹的目录结构为

.. code:: bash

    .
    |-- DvsGesture.tar.gz
    |-- LICENSE.txt
    |-- README.txt
    `-- gesture_mapping.csv


获取Event数据
-----------------------
创建训练集和测试集，其中参数 ``data_type='event'`` 表示我们使用Event数据。

.. code:: python

    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    root_dir = 'D:/datasets/DVS128Gesture'
    train_set = DVS128Gesture(root_dir, train=True, data_type='event')

运行这段代码，惊蜇框架将会完成以下工作：

#. 检测数据集是否存在，如果存在，则进行MD5校验，确认数据集无误后，开始进行解压。将原始数据解压到同级目录下的 ``extract`` 文件夹
#. DVS128 Gesture中的每个样本，是在不同光照环境下，对不同表演者进行录制的手势视频。一个AER文件中包含了多个手势，对应的会有一个csv文件来标注整个视频内各个时间段内都是哪种手势。因此，单个的视频文件并不是一个类别，而是多个类别的集合。惊蜇框架会启动多线程进行划分，将每个视频中的每个手势类别文件单独提取出来

下面是运行过程中的命令行输出：

.. code:: bash

    The [D:/datasets/DVS128Gesture/download] directory for saving downloaed files already exists, check files...
    Mkdir [D:/datasets/DVS128Gesture/extract].
    Extract [D:/datasets/DVS128Gesture/download/DvsGesture.tar.gz] to [D:/datasets/DVS128Gesture/extract].
    Mkdir [D:/datasets/DVS128Gesture/events_np].
    Start to convert the origin data from [D:/datasets/DVS128Gesture/extract] to [D:/datasets/DVS128Gesture/events_np] in np.ndarray format.
    Mkdir [('D:/datasets/DVS128Gesture//events_np//train', 'D:/datasets/DVS128Gesture//events_np//test').
    Mkdir ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9'] in [D:/datasets/DVS128Gesture/events_np/train] and ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9'] in [D:/datasets/DVS128Gesture/events_np/test].
    Start the ThreadPoolExecutor with max workers = [8].
    Start to split [D:/datasets/DVS128Gesture/extract/DvsGesture/user02_fluorescent.aedat] to samples.
    [D:/datasets/DVS128Gesture/events_np/train/0/user02_fluorescent_0.npz] saved.
    [D:/datasets/DVS128Gesture/events_np/train/1/user02_fluorescent_0.npz] saved.

    ......

    [D:/datasets/DVS128Gesture/events_np/test/8/user29_lab_0.npz] saved.
    [D:/datasets/DVS128Gesture/events_np/test/9/user29_lab_0.npz] saved.
    [D:/datasets/DVS128Gesture/events_np/test/10/user29_lab_0.npz] saved.
    Used time = [1017.27s].
    All aedat files have been split to samples and saved into [('D:/datasets/DVS128Gesture//events_np//train', 'D:/datasets/DVS128Gesture//events_np//test')].

提取各个手势类别的速度较慢，需要耐心等待。运行完成后，同级目录下会多出一个 ``events_np`` 文件夹，其中包含训练集和测试集：

.. code:: bash

    |-- events_np
    |   |-- test
    |   `-- train

打印一个数据：

.. code:: python

    event, label = train_set[0]
    for k in event.keys():
        print(k, event[k])
    print('label', label)

得到输出为：

.. code:: bash

    t [80048267 80048277 80048278 ... 85092406 85092538 85092700]
    x [49 55 55 ... 60 85 45]
    y [82 92 92 ... 96 86 90]
    p [1 0 0 ... 1 0 0]
    label 0

其中 ``event`` 使用字典格式存储Events数据，键为 ``['t', 'x', 'y', 'p']``；``label`` 是数据的标签，DVS128 Gesture共有11类。

获取Frame数据
-----------------------
将原始的Event流积分成Frame数据，是常用的处理方法，我们采用 [#PLIF]_ 的实现方式。。我们将原始的Event数据记为 :math:`E(x_{i}, y_{i}, t_{i}, p_{i}), 0 \leq i < N`；设置 ``split_by='number'`` 表示从Event数量 :math:`N` 上进行划分，接近均匀地划分为 ``frames_num=20``， 也就是 :math:`T` 段。记积分后的Frame数据中的某一帧
为 :math:`F(j)`，在 :math:`(p, x, y)` 位置的像素值为 :math:`F(j, p, x, y)`；:math:`F(j)` 是从Event流中索引介于 :math:`j_{l}` 和 :math:`j_{r}` 的Event
积分而来：

.. math::

    j_{l} & = \left\lfloor \frac{N}{T}\right \rfloor \cdot j \\
	j_{r} & = \begin{cases} \left \lfloor \frac{N}{T} \right \rfloor \cdot (j + 1), & \text{if}~~ j <  T - 1 \cr N, &  \text{if} ~~j = T - 1 \end{cases} \\
    F(j, p, x, y) &= \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})

其中 :math:`\lfloor \cdot \rfloor` 是向下取整，:math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` 是示性函数，当且仅当 :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})` 时取值为1，否则为0。

运行下列代码，惊蜇框架就会开始进行积分，创建Frame数据集：

.. code:: python

    train_set = DVS128Gesture(root_dir, train=True, data_type='frame', frames_number=20, split_by='number')

命令行的输出为：

.. code:: bash

    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/0].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/1].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/10].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/2].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/3].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/4].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/5].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/6].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/7].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/8].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/9].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/0].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/1].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/10].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/2].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/3].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/4].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/5].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/6].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/7].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/8].
    Mkdir [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/9].
    Start ThreadPoolExecutor with max workers = [8].
    Start to integrate [D:/datasets/DVS128Gesture/events_np/test/0/user24_fluorescent_0.npz] to frames and save to [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/0].
    Start to integrate [D:/datasets/DVS128Gesture/events_np/test/0/user24_fluorescent_led_0.npz] to frames and save to [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/test/0].

    ......

    Frames [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/9/user23_lab_0.npz] saved.Frames [D:/datasets/DVS128Gesture/frames_number_20_split_by_number/train/9/user23_led_0.npz] saved.

    Used time = [102.11s].

运行后，同级目录下会出现 ``frames_number_20_split_by_number`` 文件夹，这里存放了积分生成的Frame数据。

打印一个数据：

.. code:: python

    frame, label = train_set[0]
    print(frame.shape)

得到输出为：

.. code:: bash

    (20, 2, 128, 128)

查看1个积分好的Frame数据：

.. code:: python

    from spikingjelly.datasets import play_frame
    frame, label = train_set[500]
    play_frame(frame)

显示效果如下图所示：

.. image:: ../_static/tutorials/clock_driven/13_neuromorphic_datasets/dvsg.*
    :width: 100%

固定时间间隔积分
----------------------------
使用固定时间间隔积分，更符合实际物理系统。例如每 ``10 ms`` 积分一次，则长度为 ``L ms`` 的数据，可以得到  ``math.floor(L / 10)`` 帧。但
神经形态数据集中每个样本的长度往往不相同，因此会得到不同长度的帧数据。使用惊蜇框架提供的 :class:`spikingjelly.datasets.pad_sequence_collate`
和 :class:`spikingjelly.datasets.padded_sequence_mask` 可以很方便的对不等长数据进行对齐和还原。

示例代码：

.. code:: python

    import torch
    from torch.utils.data import DataLoader
    from spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask, dvs128_gesture
    root='D:/datasets/DVS128Gesture'
    train_set = dvs128_gesture.DVS128Gesture(root, data_type='frame', duration=1000000, train=True)
    for i in range(5):
        x, y = train_set[i]
        print(f'x[{i}].shape=[T, C, H, W]={x.shape}')
    train_data_loader = DataLoader(train_set, collate_fn=pad_sequence_collate, batch_size=5)
    for x, y, x_len in train_data_loader:
        print(f'x.shape=[N, T, C, H, W]={tuple(x.shape)}')
        print(f'x_len={x_len}')
        mask = padded_sequence_mask(x_len)  # mask.shape = [T, N]
        print(f'mask=\n{mask.t().int()}')
        break

输出为：

.. code:: bash

    The directory [D:/datasets/DVS128Gesture\duration_1000000] already exists.
    x[0].shape=[T, C, H, W]=(6, 2, 128, 128)
    x[1].shape=[T, C, H, W]=(6, 2, 128, 128)
    x[2].shape=[T, C, H, W]=(5, 2, 128, 128)
    x[3].shape=[T, C, H, W]=(5, 2, 128, 128)
    x[4].shape=[T, C, H, W]=(7, 2, 128, 128)
    x.shape=[N, T, C, H, W]=(5, 7, 2, 128, 128)
    x_len=tensor([6, 6, 5, 5, 7])
    mask=
    tensor([[1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)


自定义积分方法
-----------------------
惊蜇框架支持用户自定义积分方法。用户只需要提供积分函数 ``custom_integrate_function`` 以及保存frames的文件夹名 ``custom_integrated_frames_dir_name``。
``custom_integrate_function`` 是用户定义的函数，输入是 ``events, H, W``，其中 ``events`` 是一个pythono字典，键为
``['t', 'x', 'y', 'p']`` 值为 ``numpy.ndarray`` 类型。``H`` 是数据高度，``W`` 是数据宽度。例如，对于DVS手势数据集，H=128, W=128。
这个函数的返回值应该是frames。

``custom_integrated_frames_dir_name`` 可以为 ``None``，在这种情况下，保存frames的文件夹名会被设置成 ``custom_integrate_function.__name__``。


例如，我们定义这样一种积分方式：随机将全部events一分为二，然后积分成2帧。我们可定义如下函数：

.. code:: python

    import spikingjelly.datasets as sjds
    def integrate_events_to_2_frames_randomly(events: Dict, H: int, W: int):
        index_split = np.random.randint(low=0, high=events['t'].__len__())
        frames = np.zeros([2, 2, H, W])
        t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
        frames[0] = sjds.integrate_events_segment_to_frame(x, y, p, H, W, 0, index_split)
        frames[1] = sjds.integrate_events_segment_to_frame(x, y, p, H, W, index_split, events['t'].__len__())
        return frames

接下来创建数据集：

.. code:: python

    train_set = DVS128Gesture(root_dir, train=True, data_type='frame', custom_integrate_function=integrate_events_to_2_frames_randomly)

运行完毕后，在 ``root_dir`` 目录下出现了 ``integrate_events_to_2_frames_randomly`` 文件夹，保存了我们的frame数据。

查看一下我们积分得到的数据：

.. code:: python

    from spikingjelly.datasets import play_frame
    frame, label = train_set[500]
    play_frame(frame)

.. image:: ../_static/tutorials/clock_driven/13_neuromorphic_datasets/dvsg2.*
    :width: 100%

惊蜇框架还支持其他的积分方式，阅读API文档以获取更多信息。

.. [#NMNIST] Orchard, Garrick, et al. “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades.” Frontiers in Neuroscience, vol. 9, 2015, pp. 437–437.

.. [#CIFAR10DVS] Li, Hongmin, et al. “CIFAR10-DVS: An Event-Stream Dataset for Object Classification.” Frontiers in Neuroscience, vol. 11, 2017, pp. 309–309.

.. [#DVS128Gesture] Amir, Arnon, et al. “A Low Power, Fully Event-Based Gesture Recognition System.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 7388–7397.

.. [#ASLDVS] Bi, Yin, et al. “Graph-Based Object Classification for Neuromorphic Vision Sensing.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 491–501.

.. [#PLIF] Fang, Wei, et al. “Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks.” ArXiv: Neural and Evolutionary Computing, 2020.