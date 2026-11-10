"""Local, private mmap of an already hash-verified immutable checkpoint only."""
import os,pathlib,functools,mmap

def make_loader(torch,original,allowed,audit):
 allowed=pathlib.Path(allowed).resolve()
 @functools.wraps(original)
 def load(f,*args,**kw):
  name=f if isinstance(f,(str,os.PathLike)) else getattr(f,'name',None)
  selected=False
  if isinstance(name,(str,os.PathLike)) and kw.get('mmap') is None:
   try:
    selected=(pathlib.Path(name).resolve()==allowed and allowed.is_file()
              and (isinstance(f,(str,os.PathLike)) or f.tell()==0)
              and torch.serialization.get_default_mmap_options()==mmap.MAP_PRIVATE)
   except (OSError,ValueError,AttributeError):selected=False
  if not selected:
   audit['fallback_calls']+=1
   return original(f,*args,**kw)
  # Keep map_location, pickle module, weights_only, dtype and RNG behavior intact.
  result=original(str(allowed),*args,**dict(kw,mmap=True))
  audit['candidate_calls']+=1
  audit.setdefault('paths',[]).append(str(allowed))
  return result
 return load
