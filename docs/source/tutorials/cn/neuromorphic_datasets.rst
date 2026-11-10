神经形态数据集处理
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/neuromorphic_datasets`

``spikingjelly.datasets`` 提供 N-MNIST [#NMNIST]_、CIFAR10-DVS [#CIFAR10DVS]_、DVS128 Gesture [#DVS128Gesture]_、N-Caltech101 和 ASLDVS [#ASLDVS]_ 等神经形态数据集。你将以 DVS128 Gesture 为例，依次准备原始文件、读取事件、积分为帧，并处理变长序列。

自动下载和手动下载
-----------------------

CIFAR10-DVS等数据集支持自动下载。支持自动下载的数据集，在首次运行时原始数据集将会被下载到数据集根目录下的 ``download`` 文件夹。每个数据集的 ``downloadable()``
函数定义了该数据集是否能够自动下载，而 ``resource_url_md5()`` 函数定义了各个文件的下载链接和MD5。示例：

.. code:: python

    from spikingjelly.datasets import CIFAR10DVS
    from spikingjelly.datasets import DVS128Gesture

    print('CIFAR10-DVS downloadable', CIFAR10DVS.downloadable())
    print('resource, url, md5\n', CIFAR10DVS.resource_url_md5())

    print('DVS128Gesture downloadable', DVS128Gesture.downloadable())
    print('resource, url, md5\n', DVS128Gesture.resource_url_md5())

输出为：

.. code:: bash

    CIFAR10-DVS downloadable True
    resource, url, md5
     [('airplane.zip', 'https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'), ('automobile.zip', 'https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'), ('bird.zip', 'https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'), ('cat.zip', 'https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'), ('deer.zip', 'https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'), ('dog.zip', 'https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'), ('frog.zip', 'https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'), ('horse.zip', 'https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'), ('ship.zip', 'https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'), ('truck.zip', 'https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')]
    DVS128Gesture downloadable False
    resource, url, md5
     [('DvsGesture.tar.gz', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '8a5c71fb11e24e5ca5b11866ca6c00a1'), ('gesture_mapping.csv', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '109b2ae64a0e1f3ef535b18ad7367fd1'), ('LICENSE.txt', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '065e10099753156f18f51941e6e44b66'), ('README.txt', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', 'a0663d3b1d8307c329a43d949ee32d19')]

DVS128 Gesture 不支持自动下载。请从 https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794 下载 4 个文件，并放入 ``D:/datasets/DVS128Gesture/download``。目录结构应为：

.. code:: bash

    .
    |-- DvsGesture.tar.gz
    |-- LICENSE.txt
    |-- README.txt
    `-- gesture_mapping.csv


.. Note::

    不同框架的预处理方式会产生不同的训练集和测试集样本数。SpikingJelly 的划分规则和样本数见 :class:`spikingjelly.datasets.dvs128_gesture.DVS128Gesture`。


获取事件数据
-----------------------

先创建训练集。``data_type='event'`` 表示每个样本返回原始事件字典：

.. code:: python

    from spikingjelly.datasets import DVS128Gesture

    root_dir = 'D:/datasets/DVS128Gesture'
    train_set = DVS128Gesture(root_dir, train=True, data_type='event')

首次运行时，SpikingJelly 会完成以下准备工作：

#. 若处理后的数据目录不存在，SpikingJelly 校验（或下载）原始文件的 MD5，再解压到 ``extract`` 目录；已存在的 ``extract`` 会直接复用，不做完整性检查。若准备过程中断，需按报错提示清理目录和 ``.building`` 标记后重试。
#. 每个 AER 文件记录一位表演者在一种光照环境下完成的多个手势，相邻的 CSV 文件给出各手势的时间段和类别。SpikingJelly 使用多线程按这些标注切分文件，每个切分结果只包含一个手势。

提取各个手势类别需要一定时间。成功后，同级目录下会生成包含训练集和测试集的
``events_np`` 文件夹：

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

``event`` 是键为 ``['t', 'x', 'y', 'p']`` 的字典，每个值都是独立的 NumPy 数组；``t`` 的单位为微秒。``label`` 是 0 到 10 之间的类别标签。

获取帧数据
-----------------------

把事件流积分为帧是常用的预处理方法，这里采用 [#PLIF]_ 的实现。将事件记为 :math:`E(x_{i}, y_{i}, t_{i}, p_{i}), 0 \leq i < N`。设置 ``split_by='number'`` 和 ``frames_number=20`` 后，程序按事件数量将序列近似均分为 :math:`T=20` 段。记第 :math:`j` 帧为 :math:`F(j)`，位置 :math:`(p, x, y)` 的像素值为 :math:`F(j, p, x, y)`。该帧由索引在 :math:`j_{l}` 与 :math:`j_{r}` 之间的事件积分得到：

.. math::

    j_{l} & = \left\lfloor \frac{N}{T}\right \rfloor \cdot j \\
	j_{r} & = \begin{cases} \left \lfloor \frac{N}{T} \right \rfloor \cdot (j + 1), & \text{if}~~ j <  T - 1 \cr N, &  \text{if} ~~j = T - 1 \end{cases} \\
    F(j, p, x, y) &= \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})

其中 :math:`\lfloor \cdot \rfloor` 是向下取整，:math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` 是示性函数，当且仅当 :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})` 时取值为1，否则为0。

创建帧数据集时，SpikingJelly 会执行积分：

.. code:: python

    train_set = DVS128Gesture(root_dir, train=True, data_type='frame', frames_number=20, split_by='number')

处理完成后，``root_dir/frames_number_20_split_by_number`` 保存积分生成的帧。

打印一个数据：

.. code:: python

    frame, label = train_set[0]
    print(frame.shape)

得到输出为：

.. code:: bash

    (20, 2, 128, 128)

播放一个帧序列：

.. code:: python

    from spikingjelly.datasets.utils import play_frame
    frame, label = train_set[500]
    play_frame(frame)

显示效果如下图所示：

.. image:: ../../_static/tutorials/neuromorphic_datasets/dvsg.*
    :width: 100%

固定时间间隔积分
----------------------------

固定时长积分让每一帧覆盖相同的物理时间。若每 ``10 ms`` 积分一次，首末事件相差 ``L ms`` 的样本会生成 :math:`\lceil L / 10 \rceil` 帧。不同样本会得到不同的序列长度。:class:`spikingjelly.datasets.utils.pad_sequence_collate` 可以在批次内补齐序列，:class:`spikingjelly.datasets.utils.padded_sequence_mask` 则标出有效时间步。

DVS128 Gesture 的时间单位为微秒，因此 ``duration=1000000`` 表示每帧 1 秒：

.. code:: python

    import torch
    from torch.utils.data import DataLoader
    from spikingjelly.datasets.utils import pad_sequence_collate, padded_sequence_mask
    from spikingjelly.datasets import DVS128Gesture

    root='D:/datasets/DVS128Gesture'
    train_set = DVS128Gesture(root, data_type='frame', duration=1000000, train=True)
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

你也可以传入 ``custom_integrate_function`` 自定义积分方法。函数接收 ``events, H, W``：``events`` 是键为 ``['t', 'x', 'y', 'p']`` 的字典，值为 ``numpy.ndarray``；``H`` 和 ``W`` 是帧的高度和宽度。DVS128 Gesture 的 ``H`` 与 ``W`` 都是 128。函数需要返回帧数组。

``custom_integrated_frames_dir_name`` 用于指定缓存目录。传入 ``None`` 时，目录名取自 ``custom_integrate_function.__name__``。

以下函数随机选择一个切分点，把事件积分为 2 帧：

.. code:: python

    from typing import Dict
    import numpy as np
    from spikingjelly.datasets.utils import integrate_events_segment_to_frame

    def integrate_events_to_2_frames_randomly(events: Dict, H: int, W: int):
        index_split = np.random.randint(low=0, high=len(events['t']))
        frames = np.zeros([2, 2, H, W])
        t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
        frames[0] = integrate_events_segment_to_frame(x, y, p, H, W, 0, index_split)
        frames[1] = integrate_events_segment_to_frame(x, y, p, H, W, index_split, len(events['t']))
        return frames

把函数传给数据集构造器：

.. code:: python

    train_set = DVS128Gesture(root_dir, train=True, data_type='frame', custom_integrate_function=integrate_events_to_2_frames_randomly)

处理完成后，``root_dir/integrate_events_to_2_frames_randomly`` 会保存生成的帧。读取一个样本进行播放：

.. code:: python

    from spikingjelly.datasets.utils import play_frame
    frame, label = train_set[500]
    play_frame(frame)

.. image:: ../../_static/tutorials/neuromorphic_datasets/dvsg2.*
    :width: 100%

其他积分方式和参数见神经形态数据集 API 文档。

.. [#NMNIST] Orchard, Garrick, et al. “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades.” Frontiers in Neuroscience, vol. 9, 2015, pp. 437–437.

.. [#CIFAR10DVS] Li, Hongmin, et al. “CIFAR10-DVS: An Event-Stream Dataset for Object Classification.” Frontiers in Neuroscience, vol. 11, 2017, pp. 309–309.

.. [#DVS128Gesture] Amir, Arnon, et al. “A Low Power, Fully Event-Based Gesture Recognition System.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 7388–7397.

.. [#ASLDVS] Bi, Yin, et al. “Graph-Based Object Classification for Neuromorphic Vision Sensing.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 491–501.

.. [#PLIF] Fang, Wei, et al. “Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks.” ArXiv: Neural and Evolutionary Computing, 2020.
