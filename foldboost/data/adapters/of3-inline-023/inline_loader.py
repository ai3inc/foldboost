"""Single-item prediction: run worker-0 featurization inline with isolated RNG.
Keep native sampler/base seed, worker initializer, collator and pinning; no numerical kernels changed.
Only a one-item, one-worker map-style loader is eligible. Other loaders remain original.
"""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader,IterableDataset
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from torch.utils.data._utils import worker
AUDIT={'adapter':'single-item-isolated-worker-023','candidate_calls':0,'fallback_calls':0,'eligible_loaders':0,'modules':[]}
class InlineIterator(_SingleProcessDataLoaderIter):
 def __init__(self,loader):
  self.original_loader=loader
  local=DataLoader(dataset=loader.dataset,batch_sampler=loader.batch_sampler,
   num_workers=0,collate_fn=loader.collate_fn,pin_memory=loader.pin_memory,
   timeout=loader.timeout,worker_init_fn=loader.worker_init_fn,
   generator=loader.generator,persistent_workers=False,pin_memory_device=loader.pin_memory_device)
  super().__init__(local)
 def _next_data(self):
  # StopIteration is checked before worker seed/state mutation.
  index=self._next_index()
  py_state=random.getstate();np_state=np.random.get_state();cpu_state=torch.get_rng_state();threads=torch.get_num_threads();info=worker._worker_info
  cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
  try:
   torch.set_num_threads(1)
   random.seed(self._base_seed);torch.manual_seed(self._base_seed)
   np.random.seed(worker._generate_state(self._base_seed,0))
   worker._worker_info=worker.WorkerInfo(id=0,num_workers=1,seed=self._base_seed,dataset=self._dataset)
   if self.original_loader.worker_init_fn is not None:self.original_loader.worker_init_fn(0)
   data=self._dataset_fetcher.fetch(index)
   if self._pin_memory:
    from torch.utils.data._utils.pin_memory import pin_memory
    data=pin_memory(data,self._pin_memory_device)
   AUDIT['candidate_calls']+=1
   return data
  finally:
   worker._worker_info=info;random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(cpu_state)
   if cuda_states is not None:torch.cuda.set_rng_state_all(cuda_states)
   torch.set_num_threads(threads)
class InlineLoader(DataLoader):
 def __iter__(self):return InlineIterator(self)
def adapt(loader):
 eligible=(type(loader) is DataLoader and loader.num_workers==1 and not isinstance(loader.dataset,IterableDataset) and len(loader.dataset)==1 and loader.batch_size==1 and not loader.drop_last and loader.timeout==0 and loader.generator is not None)
 if eligible:
  loader.__class__=InlineLoader;AUDIT['eligible_loaders']+=1
 else:AUDIT['fallback_calls']+=1
 return loader
