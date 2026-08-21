spikingjelly.datasets.shd module
=====================================

For :class:`SpikingHeidelbergDigits <spikingjelly.datasets.shd.SpikingHeidelbergDigits>` and :class:`SpikingSpeechCommands <spikingjelly.datasets.shd.SpikingSpeechCommands>`, ``custom_integrate_function`` should have the following form:

.. code:: python

   import os
   import h5py
   import numpy as np
   from spikingjelly.datasets import utils


   def custom_integrate_function_example(h5_file: h5py.File, i: int, output_dir: str, W: int):
      events = {'t': h5_file['spikes']['times'][i], 'x': h5_file['spikes']['units'][i]}
      label = h5_file['labels'][i]
      frames = np.zeros([2, W])
      index_split = np.random.randint(low=0, high=len(events['t']))
      frames[0] = np.bincount(events['x'][:index_split].astype(int), minlength=W)
      frames[1] = np.bincount(events['x'][index_split:].astype(int), minlength=W)
      fname = os.path.join(output_dir, str(label), str(i))
      utils.np_savez(fname, frames=frames)

.. automodule:: spikingjelly.datasets.shd
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members:
