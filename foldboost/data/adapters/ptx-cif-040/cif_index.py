"""Index data blocks without per-line empty/comment checks already implied by prefix."""
import hashlib,pathlib
SOURCE_SHA='d60c8c0c0351e75c8d5ec5746852002b8552518977da70480bde11c107f9ecf1'
def install(module,audit):
 assert hashlib.file_digest(pathlib.Path(module.__file__).open('rb'),'sha256').hexdigest()==SOURCE_SHA,'Unrecognized Biotite CIF source'
 def deserialize(text):
  lines=text.splitlines();starts=[];names=[]
  for i,line in enumerate(lines):
   # A data_ prefix cannot be blank or a comment; preserve original line and name.
   if line.startswith('data_'):
    starts.append(i);names.append(line[5:])
  result=module.CIFFile(module._create_element_dict(lines,names,starts))
  audit['candidate_calls']+=1
  audit['indexed_lines']=audit.get('indexed_lines',0)+len(lines)
  return result
 module.CIFFile.deserialize=staticmethod(deserialize)
