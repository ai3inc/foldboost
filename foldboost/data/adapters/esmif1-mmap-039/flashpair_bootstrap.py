"""Only checkpoint loading changes; GPU providers and constructor RNG unchanged."""
import os,sys,json,atexit,importlib.machinery
AUDIT={'adapter':'esm-if1-checkpoint-private-mmap039','candidate_calls':0,'fallback_calls':0,'modules':[],'counter_scope':'actual verified ESM-IF1 checkpoint load calls; no GPU kernel claim'}
def install(module):
 assert module.__version__=='2.4.0+cu121'
 from mmap_loader import make_loader
 module.load=make_loader(module,module.load,'/weights/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt',AUDIT)
 AUDIT['modules'].append({'name':module.__name__,'path':str(module.__file__),'version':module.__version__})
def enable():
 base=importlib.machinery.SourceFileLoader.exec_module
 def execute(loader,module):
  result=base(loader,module)
  if module.__name__=='torch':install(module)
  return result
 importlib.machinery.SourceFileLoader.exec_module=execute
 atexit.register(lambda:sys.stderr.write('[flashpair-model-audit] '+json.dumps(AUDIT,sort_keys=True)+'\n'))
if os.environ.get('FLASHP_ESMIF1_MMAP')=='1':enable()
