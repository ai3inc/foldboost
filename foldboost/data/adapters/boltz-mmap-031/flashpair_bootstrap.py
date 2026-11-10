"""Only checkpoint loading changes; GPU providers and constructor RNG unchanged."""
import os,sys,json,atexit,importlib.machinery
AUDIT={'adapter':'boltz-checkpoint-private-mmap031','candidate_calls':0,'fallback_calls':0,'modules':[],'counter_scope':'actual verified confidence checkpoint load calls; no GPU kernel claim'}
def install(module):
 assert module.__version__=='2.12.0+cu130'
 from mmap_loader import make_loader
 module.load=make_loader(module,module.load,'/weights/boltz2/boltz2_conf.ckpt',AUDIT)
 AUDIT['modules'].append({'name':module.__name__,'path':str(module.__file__),'version':module.__version__})
def enable():
 base=importlib.machinery.SourceFileLoader.exec_module
 def execute(loader,module):
  result=base(loader,module)
  if module.__name__=='torch':install(module)
  return result
 importlib.machinery.SourceFileLoader.exec_module=execute
 atexit.register(lambda:sys.stderr.write('[flashpair-model-audit] '+json.dumps(AUDIT,sort_keys=True)+'\n'))
if os.environ.get('FLASHP_BOLTZ_MMAP')=='1':enable()
