"""Only checkpoint loading changes; GPU providers and constructor RNG unchanged."""
import os,sys,json,atexit,importlib.machinery
AUDIT={'adapter':'atlasfold-two-checkpoint-private-mmap037','candidate_calls':0,'fallback_calls':0,'modules':[],'counter_scope':'actual verified AtlasFold/AtlasLM checkpoint load calls; no GPU kernel claim'}
def install(module):
 assert module.__version__=='2.7.1+cu128'
 from mmap_loader import make_loader
 original=module.load
 wrappers={path:make_loader(module,original,path,AUDIT) for path in ['/weights/atlasfold-260703/weights/atlasfold-260703.pth', '/weights/atlaslm-3b-base/weights/atlaslm_3b_base.pth']}
 def load(f,*args,**kw):
  import pathlib
  name=f if isinstance(f,(str,os.PathLike)) else getattr(f,'name',None)
  if isinstance(name,(str,os.PathLike)):
   path=str(pathlib.Path(name).resolve())
   if path in wrappers:return wrappers[path](f,*args,**kw)
  AUDIT['fallback_calls']+=1
  return original(f,*args,**kw)
 module.load=load
 AUDIT['modules'].append({'name':module.__name__,'path':str(module.__file__),'version':module.__version__})
def enable():
 base=importlib.machinery.SourceFileLoader.exec_module
 def execute(loader,module):
  result=base(loader,module)
  if module.__name__=='torch':install(module)
  return result
 importlib.machinery.SourceFileLoader.exec_module=execute
 atexit.register(lambda:sys.stderr.write('[flashpair-model-audit] '+json.dumps(AUDIT,sort_keys=True)+'\n'))
if os.environ.get('FLASHP_ATLAS_MMAP')=='1':enable()
