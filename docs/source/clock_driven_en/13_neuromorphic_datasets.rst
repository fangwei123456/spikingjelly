Neuromorphic Datasets Processing
======================================

Authors: `fangwei123456 <https://github.com/fangwei123456>`_

``spikingjelly.datasets`` provides frequently-used neuromorphic datasets, including N-MNIST [#NMNIST]_, CIFAR10-DVS [#CIFAR10DVS]_, DVS128 Gesture [#DVS128Gesture]_, NAV Gesture [#NAVGesture]_, ASLDVS [#ASLDVS]_, etc. In this tutorial, we will take DVS 128 Gesture dataset as an example to show how to use SpikingJelly to process neuromorphic datasets.

Download DVS128 Gesture
-----------------------
The DVS128 Gesture dataset can be downloaded from https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794. The box website does not allow us to download data by python codes without login. Thus, the user have to download manually. Suppose we have downloaded the dataset into ``E:/datasets/DVS128Gesture``, then the directory structure is

.. code:: bash

    .
    |-- DvsGesture.tar.gz
    |-- LICENSE.txt
    |-- README.txt
    `-- gesture_mapping.csv


Get Events Data
-----------------------
Let us import DVS128 Gesture from SpikingJelly and create train/test set. We set ``use_frame=False`` to use Event data rather than frame data.

.. code:: python

    from spikingjelly.datasets import DVS128Gesture

    root_dir = 'E:/datasets/DVS128Gesture'
    train_set = DVS128Gesture(root_dir, train=True, use_frame=False)
    test_set = DVS128Gesture(root_dir, train=True, use_frame=False)

SpikingJelly will do the followed work when running these codes:

#. Check whether the dataset exists. If the dataset exists, check MD5 to ensure the dataset is complete. Then SpikingJelly will extract the origin data into the ``extracted`` folder
#. The sample in DVS128 Gesture is the video which records one actor displayed different gestures under different illumination conditions. Hence, an AER sample contains many gestures and there is also a adjoint csv file to label the time stamp of each gesture. Hence, an AER sample is not a sample with one class but multi-classes. SpikingJelly will use multi-threads to cut and extract each gesture from these files.

Here are the terminal outputs:

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

We have to wait for a moment because the cutting and extracting is very slow. A ``events_npy`` folder will be created and contain the train/test set:

.. code:: bash

    |-- events_npy
    |   |-- test
    |   `-- train

Print a sample:

.. code:: python

    x, y = train_set[0]
    print('event', x)
    print('label', y)

The output is:

.. code:: bash

    event {'t': array([172843814, 172843824, 172843829, ..., 179442748, 179442763,
           179442789]), 'x': array([ 54,  59,  53, ...,  36, 118, 118]), 'y': array([116, 113,  92, ..., 102,  80,  83]), 'p': array([0, 1, 1, ..., 0, 1, 1])}
    label 9

where ``x`` is a dictionary with keys ``['t', 'x', 'y', 'p']``;``y`` is the label of the sample. Note that the classes number of DVS128 Gesture is 11.

Get Frames Data
-----------------------
The event-to-frame integrating method for pre-processing neuromorphic datasets is widely used. We use the same method from [#PLIF]_ in SpikingJelly. Data in neuromorphic datasets are in the formulation of :math:`E(x_{i}, y_{i}, t_{i}, p_{i})` that represent the event's coordinate, time and polarity. We split the event's number :math:`N` into :math:`T` slices with nearly the same number of events in each slice and integrate events to frames. Note that :math:`T` is also the simulating time-step. Denote a two channels frame as :math:`F(j)` and a pixel at :math:`(p, x, y)` as :math:`F(j, p, x, y)`, the pixel value is integrated from the events data whose indices are between :math:`j_{l}` and :math:`j_{r}`:

.. math::

    j_{l} & = \left\lfloor \frac{N}{T}\right \rfloor \cdot j \\
	j_{r} & = \begin{cases} \left \lfloor \frac{N}{T} \right \rfloor \cdot (j + 1), & \text{if}~~ j <  T - 1 \cr N, &  \text{if} ~~j = T - 1 \end{cases}\\
    F(j, p, x, y) &= \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})

where :math:`\lfloor \cdot \rfloor` is the floor operation, :math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` is an indicator function and it equals 1 only when :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})`.

SpikingJelly will integrate events to frames when running the followed codes:

.. code:: python

    train_set = DVS128Gesture(root_dir, train=True, use_frame=True, frames_num=20, split_by='number', normalization=None)
    test_set = DVS128Gesture(root_dir, train=True, use_frame=True, frames_num=20, split_by='number', normalization=None)

The outputs from the terminal are:

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

A ``frames_num_20_split_by_number_normalization_None`` folder will be created and contain the Frame data.

Print a sample:

.. code:: python

    x, y = train_set[0]
    x, y = train_set[0]
    print('frame shape', x.shape)
    print('label', y)

The output is:

.. code:: bash

    frame shape torch.Size([20, 2, 128, 128])
    label 9

Let us visualize a sample:

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

We will get the images like:

.. image:: ../_static/tutorials/clock_driven/13_neuromorphic_datasets/dvsg.gif
    :width: 100%


.. [#NMNIST] Orchard, Garrick, et al. “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades.” Frontiers in Neuroscience, vol. 9, 2015, pp. 437–437.


.. [#CIFAR10DVS] Li, Hongmin, et al. “CIFAR10-DVS: An Event-Stream Dataset for Object Classification.” Frontiers in Neuroscience, vol. 11, 2017, pp. 309–309.


.. [#DVS128Gesture] Amir, Arnon, et al. “A Low Power, Fully Event-Based Gesture Recognition System.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 7388–7397.

.. [#NAVGesture] Maro, Jean-Matthieu, et al. “Event-Based Visual Gesture Recognition with Background Suppression Running on a Smart-Phone.” 2019 14th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2019), 2019, p. 1.

.. [#ASLDVS] Bi, Yin, et al. “Graph-Based Object Classification for Neuromorphic Vision Sensing.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 491–501.

.. [#PLIF] Fang, Wei, et al. “Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks.” ArXiv: Neural and Evolutionary Computing, 2020.
