import os
if os.environ.get('FLASHP_INLINE_WORKER')=='1':
 import importlib.machinery,hashlib,pathlib,atexit,sys,json
 # Import torch only after the original data module has been imported by the model.
 AUDIT={'adapter':'single-item-isolated-worker-021','candidate_calls':0,'fallback_calls':0,'eligible_loaders':0,'modules':[]}
 base=importlib.machinery.SourceFileLoader.exec_module
 def execute(loader,module):
  result=base(loader,module);path=getattr(module,'__file__','') or ''
  if path.endswith('/openfold3/core/data/framework/data_module.py'):
   assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()=='3837e0e666a11421b0fe824ee0b815d22d6c9689022fdd85f824ad9bfa92d45d'
   from inline_loader import adapt,AUDIT as live
   global AUDIT
   AUDIT=live;AUDIT['modules'].append(module.__name__);original=module.DataModule.generate_dataloader
   def generate(self,mode,*args,**kwargs):
    result=original(self,mode,*args,**kwargs)
    if getattr(mode,'name',str(mode)).split('.')[-1].lower()=='prediction':return adapt(result)
    return result
   module.DataModule.generate_dataloader=generate
  return result
 importlib.machinery.SourceFileLoader.exec_module=execute
 def report():sys.stderr.write('[flashpair-model-audit] '+json.dumps(AUDIT)+'\n')
 atexit.register(report)
