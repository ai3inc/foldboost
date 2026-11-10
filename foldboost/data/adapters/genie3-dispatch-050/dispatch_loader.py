"""Reorder unchanged memo/string/list opcode branches of pinned PyTorch restricted loader."""
import ast,hashlib,pathlib,contextvars,functools,os
SOURCE_SHA='bc2f0fdcb5657109d066951d58864ec4227c8f65ce4d1f144015062ef097e4b2'

def transformed(source):
 if hashlib.sha256(source).hexdigest()!=SOURCE_SHA:raise AssertionError('Unpinned restricted-unpickler source')
 tree=ast.parse(source);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Unpickler');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='load')
 loop=next(x for x in fn.body if isinstance(x,ast.While));head=next(x for x in loop.body if isinstance(x,ast.If) and isinstance(x.test,ast.Compare))
 branches=[];node=head
 while True:
  branches.append(node)
  if len(node.orelse)==1 and isinstance(node.orelse[0],ast.If):node=node.orelse[0]
  else:tail=node.orelse;break
 tests=["key[0] in [BINPUT[0], LONG_BINPUT[0]]", "key[0] == BINUNICODE[0]", "key[0] in [BINGET[0], LONG_BINGET[0]]", "key[0] == EMPTY_LIST[0]", "key[0] == APPEND[0]"]
 moved=[next(x for x in branches if ast.dump(x.test)==ast.dump(ast.parse(t,mode='eval').body)) for t in tests]
 rest=[x for x in branches if x not in moved];ordered=moved+rest
 # Original byte-valued opcode tests are disjoint. Keep EVERY handler body and final else intact.
 for i,node in enumerate(ordered):node.orelse=[ordered[i+1]] if i+1<len(ordered) else tail
 loop.body[loop.body.index(head)]=ordered[0]
 ast.fix_missing_locations(fn)
 return fn

def install(torch,original,allowed,audit):
 mod=torch._weights_only_unpickler
 source=pathlib.Path(mod.__file__).read_bytes();fn=transformed(source)
 ns=dict(mod.__dict__);exec(compile(ast.Module(body=[fn],type_ignores=[]),'<flashpair-reordered-restricted-loader050>','exec'),ns)
 faster=ns['load'];base=mod.Unpickler.load;enabled=contextvars.ContextVar('flashpair_genie_restricted_load050',default=False)
 def restricted(self):
  if enabled.get():audit['restricted_calls']=audit.get('restricted_calls',0)+1;return faster(self)
  return base(self)
 mod.Unpickler.load=restricted
 paths={pathlib.Path(p).resolve() for p in allowed}
 @functools.wraps(original)
 def load(f,*args,**kwargs):
  # Public arguments, weights_only defaults and security checks are unchanged.
  selected=isinstance(f,(str,os.PathLike)) and pathlib.Path(f).resolve() in paths
  if not selected:audit['fallback_calls']+=1;return original(f,*args,**kwargs)
  token=enabled.set(True)
  try:result=original(f,*args,**kwargs)
  finally:enabled.reset(token)
  audit['candidate_calls']+=1;audit.setdefault('paths',[]).append(str(pathlib.Path(f).resolve()))
  return result
 return load
