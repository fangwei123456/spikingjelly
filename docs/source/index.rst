.. _index:

欢迎来到惊蜇(SpikingJelly)的文档
###################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ 是一个基于 `PyTorch <https://pytorch.org/>`_ ，使用脉冲神经\
网络(Spiking Neural Network, SNN)进行深度学习的框架。

* :ref:`Homepage in English <index_en>`

安装
----------------

注意，SpikingJelly是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装spikingjelly。

奇数版本是开发版，随着GitHub/OpenI不断更新。偶数版本是稳定版，可以从PyPI获取。

从 `PyPI <https://pypi.org/project/spikingjelly/>`_ 安装最新的稳定版本：

.. code-block:: bash

    pip install spikingjelly

从源代码安装最新的开发版：

通过 `GitHub <https://github.com/fangwei123456/spikingjelly>`_：

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    python setup.py install

通过 `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_ ：

.. code-block:: bash

    git clone https://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    python setup.py install

.. toctree::
    :maxdepth: 1
    :caption: 上手教程

    tutorial.clock_driven
    tutorial.event_driven
    /clock_driven/0_neuron
    /clock_driven/2_encoding
    /clock_driven/3_fc_mnist
    /clock_driven/4_conv_fashion_mnist
    /clock_driven/5_ann2snn
    /clock_driven/6_dqn_cart_pole
    /clock_driven/7_a2c_cart_pole
    /clock_driven/8_ppo_cart_pole
    /clock_driven/9_spikingLSTM_text
    /clock_driven/10_propagation_pattern
    /clock_driven/11_cext_neuron_with_lbl
    /clock_driven/12_monitor
    /clock_driven/13_neuromorphic_datasets
    /clock_driven/14_classify_dvsg
    /clock_driven/15_recurrent_connection_and_stateful_synapse
    /clock_driven/16_train_large_scale_snn


模块文档
-------------------------

* :ref:`APIs`

文档索引
-------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


引用和出版物
-------------------------
如果您在自己的工作中用到了惊蜇(SpikingJelly)，您可以按照下列格式进行引用：

.. code-block::

    @misc{SpikingJelly,
        title = {SpikingJelly},
        author = {Fang, Wei and Chen, Yanqi and Ding, Jianhao and Chen, Ding and Yu, Zhaofei and Zhou, Huihui and Tian, Yonghong and other contributors},
        year = {2020},
        howpublished = {\url{https://github.com/fangwei123456/spikingjelly}},
        note = {Accessed: YYYY-MM-DD},
    }

其中的 `YYYY-MM-DD` 需要更改为您的工作使用的惊蜇(SpikingJelly)版本对应的最后一次代码修改日期。

使用惊蜇(SpikingJelly)的出版物可见于 `Publications using SpikingJelly <https://github.com/fangwei123456/spikingjelly/blob/master/publications.md>`_。

项目信息
-------------------------
北京大学信息科学技术学院数字媒体所媒体学习组 `Multimedia Learning Group <https://pkuml.org/>`_ 和 `鹏城实验室 <https://www.pcl.ac.cn/>`_ 是SpikingJelly的主要开发者。

.. image:: ./_static/logo/pku.png
    :width: 20%

.. image:: ./_static/logo/pcl.png
    :width: 20%

开发人员名单可见于 `贡献者 <https://github.com/fangwei123456/spikingjelly/graphs/contributors>`_ 。

友情链接
-------------------------
* `脉冲神经网络相关博客 <https://www.cnblogs.com/lucifer1997/tag/SNN/>`_
* `脉冲强化学习相关博客 <https://www.cnblogs.com/lucifer1997/tag/SNN-RL/>`_

.. _index_en:

Welcome to SpikingJelly's documentation
############################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ is an open-source deep learning framework for Spiking Neural Network (SNN) based on `PyTorch <https://pytorch.org/>`_.

* :ref:`中文首页 <index>`

Installation
----------------

Note that SpikingJelly is based on PyTorch. Please make sure that you have installed PyTorch before you install SpikingJelly.

The odd version number is the developing version, which is updated with GitHub/OpenI repository. The even version number is the stable version and available at PyPI.

Install the last stable version from `PyPI <https://pypi.org/project/spikingjelly/>`_：

.. code-block:: bash

    pip install spikingjelly

Install the latest developing version from the source codes:

From `GitHub <https://github.com/fangwei123456/spikingjelly>`_:

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    python setup.py install

From `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_：

.. code-block:: bash

    git clone https://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    python setup.py install


.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorial_en.clock_driven
    /clock_driven_en/0_neuron
    /clock_driven_en/2_encoding
    /clock_driven_en/3_fc_mnist
    /clock_driven_en/4_conv_fashion_mnist
    /clock_driven_en/5_ann2snn
    /clock_driven_en/6_dqn_cart_pole
    /clock_driven_en/7_a2c_cart_pole
    /clock_driven_en/8_ppo_cart_pole
    /clock_driven_en/9_spikingLSTM_text
    /clock_driven_en/10_propagation_pattern
    /clock_driven_en/11_cext_neuron_with_lbl
    /clock_driven_en/13_neuromorphic_datasets
    /clock_driven_en/14_classify_dvsg
    /clock_driven_en/15_recurrent_connection_and_stateful_synapse
    /clock_driven_en/16_train_large_scale_snn



Modules Docs
-------------------------

* :ref:`APIs`

Indices and tables
-------------------------

* :ref:`Index <genindex>`
* :ref:`Module Index <modindex>`
* :ref:`Search Page <search>`

Citation
-------------------------

If you use SpikingJelly in your work, please cite it as follows:

.. code-block::

    @misc{SpikingJelly,
        title = {SpikingJelly},
        author = {Fang, Wei and Chen, Yanqi and Ding, Jianhao and Chen, Ding and Yu, Zhaofei and Zhou, Huihui and Tian, Yonghong and other contributors},
        year = {2020},
        howpublished = {\url{https://github.com/fangwei123456/spikingjelly}},
        note = {Accessed: YYYY-MM-DD},
    }

Note: To specify the version of framework you are using, the default value YYYY-MM-DD in the note field should be replaced with the date of the last change of the framework you are using, i.e. the date of the latest commit.

Publications using SpikingJelly are recorded in `Publications using SpikingJelly <https://github.com/fangwei123456/spikingjelly/blob/master/publications.md>`_. If you use SpikingJelly in your paper, you can also add it to this table by pull request.

About
-------------------------
`Multimedia Learning Group, Institute of Digital Media (NELVT), Peking University <https://pkuml.org/>`_ and `Peng Cheng Laboratory <http://www.szpclab.com/>`_ are the main developers of SpikingJelly.

.. image:: ./_static/logo/pku.png
    :width: 20%

.. image:: ./_static/logo/pcl.png
    :width: 20%

The list of developers can be found at `contributors <https://github.com/fangwei123456/spikingjelly/graphs/contributors>`_.

.. _APIs:
.. toctree::
   :maxdepth: 4
   :caption: APIs

   spikingjelly.clock_driven
   spikingjelly.datasets
   spikingjelly.event_driven
   spikingjelly.visualizing
   spikingjelly.cext
