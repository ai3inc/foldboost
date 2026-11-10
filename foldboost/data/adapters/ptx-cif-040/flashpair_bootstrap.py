import os,sys,json,atexit,importlib.machinery
AUDIT={'adapter':'biotite-cif-prefix-index040','candidate_calls':0,'indexed_lines':0,'modules':[],'counter_scope':'actual CIFFile.deserialize calls; no GPU kernel or checkpoint mmap change'}
def install(module):
 from cif_index import install as patch
 patch(module,AUDIT)
 AUDIT['modules'].append({'name':module.__name__,'path':str(module.__file__)})
def enable():
 base=importlib.machinery.SourceFileLoader.exec_module
 def execute(loader,module):
  result=base(loader,module)
  if loader.name=='biotite.structure.io.pdbx.cif':install(module)
  return result
 importlib.machinery.SourceFileLoader.exec_module=execute
 atexit.register(lambda:sys.stderr.write('[flashpair-model-audit] '+json.dumps(AUDIT,sort_keys=True)+'\n'))
if os.environ.get('FLASHP_PTX_CIF')=='1':enable()
