**Language:**
:ref:`中文 <index>` | :ref:`English <index_en>`

.. _index:

欢迎来到惊蜇(SpikingJelly)的文档
###################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ 是一个基于 `PyTorch <https://pytorch.org/>`_ ，使用脉冲神经网络(Spiking Neural Network, SNN)进行深度学习的框架。

版本说明
----------------
自 ``0.0.0.0.14`` 版本开始，包括 ``clock_driven`` 和 ``event_driven`` 在内的模块被重命名了，请参考教程 :doc:`./tutorials/cn/migrate_from_legacy`。

不同版本文档的地址（其中 `latest` 是开发版）：

- `zero <https://spikingjelly.readthedocs.io/zh_CN/zero/>`_

- `0.0.0.0.4 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/>`_

- `0.0.0.0.6 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.6/>`_

- `0.0.0.0.8 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.8/>`_

- `0.0.0.0.10 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.10/>`_

- `0.0.0.0.12 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.12/>`_

- `0.0.0.0.14 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/>`_

- `latest <https://spikingjelly.readthedocs.io/zh_CN/latest/>`_

安装
----------------

注意，SpikingJelly是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装SpikingJelly。

奇数版本是开发版，随着GitHub/OpenI不断更新。偶数版本是稳定版，可以从PyPI获取。

**从 PyPI 安装最新的稳定版本：**

.. code-block:: bash

    pip install spikingjelly

**从源代码安装最新的开发版：**

通过 `GitHub <https://github.com/fangwei123456/spikingjelly>`_：

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    pip install .

通过 `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_ ：

.. code-block:: bash

    git clone https://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    pip install .

**可选依赖**

若想使用 CuPy 后端，需安装 `CuPy <https://docs.cupy.dev/en/stable/install.html#installing-cupy>`_ 。

.. code:: bash

    pip install cupy-cuda12x # for CUDA 12.x
    pip install cupy-cuda11x # for CUDA 11.x

若想使用 Triton 后端，请确保安装了 `Triton <https://github.com/triton-lang/triton>`_。

.. code:: bash

    pip install triton==3.3.1 # spikingjelly is tested with triton==3.3.1

若想使用 ``nir_exchange`` 功能，请安装 `NIR <https://github.com/neuromorphs/NIR>`_ 和 `NIRTorch <https://github.com/neuromorphs/NIRTorch>`_ 。

.. code:: bash

    pip install nir nirtorch

上手教程
----------------------

.. toctree::
    :maxdepth: 2

    /tutorials/cn/index

引用和出版物
-------------------------
如果您在自己的工作中用到了惊蜇(SpikingJelly)，您可以按照下列格式进行引用：

.. code-block::

    @article{
    doi:10.1126/sciadv.adi1480,
    author = {Wei Fang  and Yanqi Chen  and Jianhao Ding  and Zhaofei Yu  and Timothée Masquelier  and Ding Chen  and Liwei Huang  and Huihui Zhou  and Guoqi Li  and Yonghong Tian },
    title = {SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence},
    journal = {Science Advances},
    volume = {9},
    number = {40},
    pages = {eadi1480},
    year = {2023},
    doi = {10.1126/sciadv.adi1480},
    URL = {https://www.science.org/doi/abs/10.1126/sciadv.adi1480},
    eprint = {https://www.science.org/doi/pdf/10.1126/sciadv.adi1480},
    abstract = {Spiking neural networks (SNNs) aim to realize brain-inspired intelligence on neuromorphic chips with high energy efficiency by introducing neural dynamics and spike properties. As the emerging spiking deep learning paradigm attracts increasing interest, traditional programming frameworks cannot meet the demands of the automatic differentiation, parallel computation acceleration, and high integration of processing neuromorphic datasets and deployment. In this work, we present the SpikingJelly framework to address the aforementioned dilemma. We contribute a full-stack toolkit for preprocessing neuromorphic datasets, building deep SNNs, optimizing their parameters, and deploying SNNs on neuromorphic chips. Compared to existing methods, the training of deep SNNs can be accelerated 11×, and the superior extensibility and flexibility of SpikingJelly enable users to accelerate custom models at low costs through multilevel inheritance and semiautomatic code generation. SpikingJelly paves the way for synthesizing truly energy-efficient SNN-based machine intelligence systems, which will enrich the ecology of neuromorphic computing. Motivation and introduction of the software framework SpikingJelly for spiking deep learning.}}

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
* `神经形态计算软件框架LAVA相关博客 <https://www.cnblogs.com/lucifer1997/p/16286303.html>`_

.. _index_en:

Welcome to SpikingJelly's documentation
############################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ is an open-source deep learning framework for Spiking Neural Network (SNN) based on `PyTorch <https://pytorch.org/>`_.

Notification
----------------
From the version ``0.0.0.0.14``, modules including ``clock_driven`` and ``event_driven`` are renamed. \
Please refer to the tutorial :doc:`./tutorials/en/migrate_from_legacy`.

Docs for different versions (`latest` is the developing version):

- `zero <https://spikingjelly.readthedocs.io/zh_CN/zero/>`_

- `0.0.0.0.4 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/#index-en>`_

- `0.0.0.0.6 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.6/#index-en>`_

- `0.0.0.0.8 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.8/#index-en>`_

- `0.0.0.0.10 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.10/#index-en>`_

- `0.0.0.0.12 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.12/#index-en>`_

- `0.0.0.0.14 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/#index-en>`_

- `latest <https://spikingjelly.readthedocs.io/zh_CN/latest/#index-en>`_

Installation
----------------

Note that SpikingJelly is based on PyTorch. Please make sure that you have installed PyTorch before you install SpikingJelly.

The odd version number is the developing version, which is updated with GitHub/OpenI repository. The even version number is the stable version and available at PyPI.

**Install the last stable version from PyPI:**

.. code-block:: bash

    pip install spikingjelly

**Install the latest developing version from the source codes:**

From `GitHub <https://github.com/fangwei123456/spikingjelly>`_:

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    pip install .

From `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_：

.. code-block:: bash

    git clone https://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    pip install .

**Optional Dependencies**

To enable CuPy backend, install `CuPy <https://docs.cupy.dev/en/stable/install.html#installing-cupy>`_ .

.. code:: bash

    pip install cupy-cuda12x # for CUDA 12.x
    pip install cupy-cuda11x # for CUDA 11.x

To enable Triton backend, make sure that `Triton <https://github.com/triton-lang/triton>`_ is installed.

.. code:: bash

    pip install triton==3.3.1 # spikingjelly is tested with triton==3.3.1

To enable ``nir_exchange`` , install `NIR <https://github.com/neuromorphs/NIR>`_ and `NIRTorch <https://github.com/neuromorphs/NIRTorch>`_ .

.. code:: bash

    pip install nir nirtorch

Tutorials
------------------------

.. toctree::
    :maxdepth: 2

    /tutorials/en/index

Citation
-------------------------

If you use SpikingJelly in your work, please cite it as follows:

.. code-block::

    @article{
    doi:10.1126/sciadv.adi1480,
    author = {Wei Fang  and Yanqi Chen  and Jianhao Ding  and Zhaofei Yu  and Timothée Masquelier  and Ding Chen  and Liwei Huang  and Huihui Zhou  and Guoqi Li  and Yonghong Tian },
    title = {SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence},
    journal = {Science Advances},
    volume = {9},
    number = {40},
    pages = {eadi1480},
    year = {2023},
    doi = {10.1126/sciadv.adi1480},
    URL = {https://www.science.org/doi/abs/10.1126/sciadv.adi1480},
    eprint = {https://www.science.org/doi/pdf/10.1126/sciadv.adi1480},
    abstract = {Spiking neural networks (SNNs) aim to realize brain-inspired intelligence on neuromorphic chips with high energy efficiency by introducing neural dynamics and spike properties. As the emerging spiking deep learning paradigm attracts increasing interest, traditional programming frameworks cannot meet the demands of the automatic differentiation, parallel computation acceleration, and high integration of processing neuromorphic datasets and deployment. In this work, we present the SpikingJelly framework to address the aforementioned dilemma. We contribute a full-stack toolkit for preprocessing neuromorphic datasets, building deep SNNs, optimizing their parameters, and deploying SNNs on neuromorphic chips. Compared to existing methods, the training of deep SNNs can be accelerated 11×, and the superior extensibility and flexibility of SpikingJelly enable users to accelerate custom models at low costs through multilevel inheritance and semiautomatic code generation. SpikingJelly paves the way for synthesizing truly energy-efficient SNN-based machine intelligence systems, which will enrich the ecology of neuromorphic computing. Motivation and introduction of the software framework SpikingJelly for spiking deep learning.}}


Publications using SpikingJelly are recorded in `Publications using SpikingJelly <https://github.com/fangwei123456/spikingjelly/blob/master/publications.md>`_. If you use SpikingJelly in your paper, you can also add it to this table by pull request.

About
-------------------------
`Multimedia Learning Group, Institute of Digital Media (NELVT), Peking University <https://pkuml.org/>`_ and `Peng Cheng Laboratory <http://www.szpclab.com/>`_ are the main developers of SpikingJelly.

.. image:: ./_static/logo/pku.png
    :width: 20%

.. image:: ./_static/logo/pcl.png
    :width: 20%

The list of developers can be found at `contributors <https://github.com/fangwei123456/spikingjelly/graphs/contributors>`_.

API 文档 | API Docs
##########################

.. toctree::
   :maxdepth: 2

   /APIs/spikingjelly

.. toctree::
    :hidden:

    出版物 | Publications <https://github.com/fangwei123456/spikingjelly/blob/master/publications.md>
