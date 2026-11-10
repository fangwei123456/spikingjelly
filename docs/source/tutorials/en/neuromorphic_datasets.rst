Process neuromorphic datasets
======================================

Authors: `fangwei123456 <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/neuromorphic_datasets`

``spikingjelly.datasets`` provides neuromorphic datasets including N-MNIST [#NMNIST]_, CIFAR10-DVS [#CIFAR10DVS]_, DVS128 Gesture [#DVS128Gesture]_, N-Caltech101, and ASLDVS [#ASLDVS]_. Using DVS128 Gesture, you will prepare the raw files, read events, integrate them into frames, and batch variable-length sequences.

Download datasets
-------------------------------------------------------

SpikingJelly can download some datasets (e.g., CIFAR10-DVS) automatically. When we use these datasets for the first time, SpikingJelly
will download the dataset to ``download`` in the root directory. The ``downloadable()`` function of each dataset defines
whether this dataset can be downloaded automatically, and the ``resource_url_md5()`` function defines the download url and
MD5 of each file. Here is an example:

.. code:: python

    from spikingjelly.datasets import CIFAR10DVS
    from spikingjelly.datasets import DVS128Gesture

    print('CIFAR10-DVS downloadable', CIFAR10DVS.downloadable())
    print('resource, url, md5\n', CIFAR10DVS.resource_url_md5())

    print('DVS128Gesture downloadable', DVS128Gesture.downloadable())
    print('resource, url, md5\n', DVS128Gesture.resource_url_md5())

The outputs are:

.. code:: bash

    CIFAR10-DVS downloadable True
    resource, url, md5
     [('airplane.zip', 'https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'), ('automobile.zip', 'https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'), ('bird.zip', 'https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'), ('cat.zip', 'https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'), ('deer.zip', 'https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'), ('dog.zip', 'https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'), ('frog.zip', 'https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'), ('horse.zip', 'https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'), ('ship.zip', 'https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'), ('truck.zip', 'https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')]
    DVS128Gesture downloadable False
    resource, url, md5
     [('DvsGesture.tar.gz', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '8a5c71fb11e24e5ca5b11866ca6c00a1'), ('gesture_mapping.csv', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '109b2ae64a0e1f3ef535b18ad7367fd1'), ('LICENSE.txt', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '065e10099753156f18f51941e6e44b66'), ('README.txt', 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', 'a0663d3b1d8307c329a43d949ee32d19')]


DVS128 Gesture does not support automatic download. Download its four files from https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794 and place them in ``D:/datasets/DVS128Gesture/download``. The directory should contain:

.. code:: bash

    .
    |-- DvsGesture.tar.gz
    |-- LICENSE.txt
    |-- README.txt
    `-- gesture_mapping.csv

.. admonition:: Note
    :class: note

    Different preprocessing methods produce different train and test sample counts. See :class:`spikingjelly.datasets.dvs128_gesture.DVS128Gesture` for SpikingJelly's split rules and counts.


Get event data
-----------------------

Create the training set first. With ``data_type='event'``, each sample is returned as an event dictionary:

.. code:: python

    from spikingjelly.datasets import DVS128Gesture

    root_dir = 'D:/datasets/DVS128Gesture'
    train_set = DVS128Gesture(root_dir, train=True, data_type='event')

On the first run, SpikingJelly prepares the data in these steps:

#. If the processed data directory is missing, SpikingJelly checks the original files against their MD5 hashes (or downloads them), then extracts them to ``extract``. An existing ``extract`` directory is reused without an integrity check. If preparation is interrupted, follow the error message to clean the partial directory and ``.building`` marker before retrying.
#. Each AER file records one actor performing several gestures under one lighting condition. Its CSV file gives the time range and class for each gesture. SpikingJelly uses multiple threads to split the recording so that each output sample contains one gesture.

Cutting and extracting can take some time. On success, an ``events_np`` folder
contains the train/test set:

.. code:: bash

    |-- events_np
    |   |-- test
    |   `-- train

Print a sample:

.. code:: python

    event, label = train_set[0]
    for k in event.keys():
        print(k, event[k])
    print('label', label)

The output is:

.. code:: bash

    t [80048267 80048277 80048278 ... 85092406 85092538 85092700]
    x [49 55 55 ... 60 85 45]
    y [82 92 92 ... 96 86 90]
    p [1 0 0 ... 1 0 0]
    label 0

``event`` is a dictionary with keys ``['t', 'x', 'y', 'p']``. Each value is an independent NumPy array, and ``t`` is measured in microseconds. ``label`` is an integer from 0 to 10.

Get frame data
-----------------------

The event-to-frame integrating method for pre-processing neuromorphic datasets is widely used. We use the same method from [#PLIF]_ in SpikingJelly. Data in neuromorphic datasets are in the formulation of :math:`E(x_{i}, y_{i}, t_{i}, p_{i})` that represent the event's coordinate, time and polarity. We split the event's number :math:`N` into :math:`T` slices with nearly the same number of events in each slice and integrate events to frames. With ``split_by='number'`` and ``frames_number=20``, :math:`T` is the number of frames, so feeding one frame per simulation step gives 20 steps. Denote a two channels frame as :math:`F(j)` and a pixel at :math:`(p, x, y)` as :math:`F(j, p, x, y)`, the pixel value is integrated from the events data whose indices are between :math:`j_{l}` and :math:`j_{r}`:

.. math::

    j_{l} & = \left\lfloor \frac{N}{T}\right \rfloor \cdot j \\
    j_{r} & = \begin{cases} \left \lfloor \frac{N}{T} \right \rfloor \cdot (j + 1), & \text{if}~~ j <  T - 1 \cr N, &  \text{if} ~~j = T - 1 \end{cases} \\
    F(j, p, x, y) &= \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})

where :math:`\lfloor \cdot \rfloor` is the floor operation, :math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` is an indicator function and it equals 1 only when :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})`.

SpikingJelly will integrate events to frames when running the following code:

.. code:: python

    train_set = DVS128Gesture(root_dir, train=True, data_type='frame', frames_number=20, split_by='number')

A ``frames_number_20_split_by_number`` folder will be created and contain the Frame data.

Print a sample:

.. code:: python

    frame, label = train_set[0]
    print(frame.shape)

The outputs are:

.. code:: bash

    (20, 2, 128, 128)

Visualize one sample:

.. code:: python

    from spikingjelly.datasets.utils import play_frame
    frame, label = train_set[500]
    play_frame(frame)

We will get the images like:

.. image:: ../../_static/tutorials/neuromorphic_datasets/dvsg.*
    :width: 100%

Fixed-duration integration
--------------------------------------

Fixed-duration integration gives every frame the same physical time span. With a ``10 ms`` duration, a sample whose first and last events are ``L ms`` apart produces :math:`\lceil L / 10 \rceil` frames. Sample lengths therefore vary. :class:`spikingjelly.datasets.utils.pad_sequence_collate` pads sequences within a batch, and :class:`spikingjelly.datasets.utils.padded_sequence_mask` marks the valid time steps.

DVS128 Gesture timestamps are measured in microseconds, so ``duration=1000000`` means one second per frame:

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

The outputs are:

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

Custom integration
----------------------------

You can pass ``custom_integrate_function`` to define another integration method. The function receives ``events, H, W``. ``events`` is a dictionary whose keys are ``['t', 'x', 'y', 'p']`` and whose values are ``numpy.ndarray`` objects. ``H`` and ``W`` are the frame height and width, both 128 for DVS128 Gesture. The function must return the frame array.

``custom_integrated_frames_dir_name`` selects the cache directory. When it is ``None``, the directory name comes from ``custom_integrate_function.__name__``.

This function chooses a random split point and integrates the events into two frames:

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

Pass the function to the dataset constructor:

.. code:: python

    train_set = DVS128Gesture(root_dir, train=True, data_type='frame', custom_integrate_function=integrate_events_to_2_frames_randomly)

After preprocessing, ``root_dir/integrate_events_to_2_frames_randomly`` contains the generated frames. Load and play one sample:

.. code:: python

    from spikingjelly.datasets.utils import play_frame
    frame, label = train_set[500]
    play_frame(frame)

.. image:: ../../_static/tutorials/neuromorphic_datasets/dvsg2.*
    :width: 100%

SpikingJelly provides more methods to integrate events to frames. Read the API doc for more details.

.. [#NMNIST] Orchard, Garrick, et al. “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades.” Frontiers in Neuroscience, vol. 9, 2015, pp. 437–437.

.. [#CIFAR10DVS] Li, Hongmin, et al. “CIFAR10-DVS: An Event-Stream Dataset for Object Classification.” Frontiers in Neuroscience, vol. 11, 2017, pp. 309–309.

.. [#DVS128Gesture] Amir, Arnon, et al. “A Low Power, Fully Event-Based Gesture Recognition System.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 7388–7397.

.. [#ASLDVS] Bi, Yin, et al. “Graph-Based Object Classification for Neuromorphic Vision Sensing.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 491–501.

.. [#PLIF] Fang, Wei, et al. “Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks.” ArXiv: Neural and Evolutionary Computing, 2020.
