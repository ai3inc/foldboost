import os,sys,json,atexit,importlib.machinery
AUDIT={'adapter':'genie3-restricted-unpickler-memo-string-list-dispatch050','candidate_calls':0,'restricted_calls':0,'fallback_calls':0,'modules':[]}
def install(module):
 assert module.__version__=='2.7.1+cu126'
 from dispatch_loader import install as connect,SOURCE_SHA
 module.load=connect(module,module.load,['/weights/genie3/pretrained/v1/checkpoints/step=600000.ckpt'],AUDIT)
 AUDIT['modules'].append({'name':module.__name__,'path':module.__file__,'version':module.__version__,'restricted_source_sha':SOURCE_SHA})
def enable():
 base=importlib.machinery.SourceFileLoader.exec_module
 def execute(loader,module):
  result=base(loader,module)
  if loader.name=='torch':install(module)
  return result
 importlib.machinery.SourceFileLoader.exec_module=execute
 atexit.register(lambda:sys.stderr.write('[flashpair-model-audit] '+json.dumps(AUDIT,sort_keys=True)+'\n'))
if os.environ.get('FLASHP_GENIE3_DISPATCH')=='1':enable()
