"""Pinned-library fingerprint reuse across CLI processes in an owner-private cache.
First use and any identity/metadata change rehash every byte against the wheel pin.
This assumes the cache-owning user and local filesystem metadata are trusted, like Triton's JIT cache.
No runtime source or GPU arithmetic is modified. Not full rehash on cache hits.
"""
import ast,atexit,functools,hashlib,json,os,stat,tempfile,fcntl
from pathlib import Path
SOURCE_SHA = 'dc767c8fadcf23ea82d79e257c37d44077eae7f681cf967565fd43e9c017937b'
LIB_SHA = 'a22254a46935cd91d93139f3b30d4c2e11cfabddd9bbd7a38540a07d5cd4dbfe'
COUNTS = {'candidate_calls':0,'fallback_calls':0,'cache_hits':0,'cache_misses':0,'hashed_bytes':0,'rejected_receipts':0}

def signature(st):
    return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_uid,st.st_gid,st.st_mode]

class Digest:
    def __init__(self,value):self.value=value
    def hexdigest(self):return self.value

def private_directory(path):
    path.mkdir(mode=0o700,parents=True,exist_ok=True)
    st=path.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid!=os.getuid() or st.st_mode & 0o077:
        raise RuntimeError('Fingerprint cache is not owner-private')

def pinned_digest(path,cache,expected):
    path=Path(path).resolve(strict=True);cache=Path(cache);private_directory(cache)
    key=hashlib.sha256(str(path).encode()).hexdigest();receipt=cache/(key+'.json')
    lock_fd=os.open(str(cache/(key+'.lock')),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(lock_fd,'a+') as lock:
        lockstat=os.fstat(lock.fileno())
        if lockstat.st_uid!=os.getuid() or lockstat.st_mode & 0o077 or not stat.S_ISREG(lockstat.st_mode):raise RuntimeError('Unsafe cache lock')
        fcntl.flock(lock,fcntl.LOCK_EX)
        with path.open('rb') as f:
            before=os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode):raise RuntimeError('Library must be a regular file')
            record={'schema':1,'source_pin':SOURCE_SHA,'path':str(path),'signature':signature(before),'sha256':expected}
            cached=None
            try:
                fd=os.open(str(receipt),os.O_RDONLY|os.O_NOFOLLOW)
                with os.fdopen(fd,'r') as rf:
                    rs=os.fstat(rf.fileno())
                    if rs.st_uid==os.getuid() and not rs.st_mode & 0o077 and stat.S_ISREG(rs.st_mode) and rs.st_size<8192:cached=json.load(rf)
            except (OSError,ValueError):pass
            if cached==record:
                if signature(path.stat())!=signature(before) or signature(os.fstat(f.fileno()))!=signature(before):raise RuntimeError('Library changed during fingerprint check')
                COUNTS['cache_hits']+=1;COUNTS['candidate_calls']+=1
                return Digest(expected)
            if cached is not None:COUNTS['rejected_receipts']+=1
            digest=hashlib.file_digest(f,'sha256').hexdigest()
            if signature(path.stat())!=signature(before) or signature(os.fstat(f.fileno()))!=signature(before):raise RuntimeError('Library changed during full hash')
            if digest!=expected:raise RuntimeError('Library does not match original pinned wheel')
            fd,tmp=tempfile.mkstemp(prefix='.receipt-',dir=cache)
            try:
                with os.fdopen(fd,'w') as wf:json.dump(record,wf,sort_keys=True);wf.flush();os.fsync(wf.fileno())
                os.replace(tmp,receipt)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
            COUNTS['cache_misses']+=1;COUNTS['candidate_calls']+=1;COUNTS['hashed_bytes']+=before.st_size
            return Digest(expected)

def verified_hash(path):
    cache=os.environ.get('FLASHP_FINGERPRINT_CACHE')
    if not cache:raise RuntimeError('Explicit private fingerprint cache required')
    return pinned_digest(path,cache,LIB_SHA)

def replacement(module):
    source = Path(module.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise RuntimeError('Unsupported or modified Triton compiler source')
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'triton_key')
    # Only replace the exact backend read loop; all file discovery/order and other digests stay original.
    index = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'libtriton_hash')
    block = fn.body[index:index+2]
    expected = ast.parse('''libtriton_hash = hashlib.sha256()
with open(os.path.join(TRITON_PATH, "_C/libtriton.so"), "rb") as f:
    while True:
        chunk = f.read(1024**2)
        if not chunk:
            break
        libtriton_hash.update(chunk)
''').body
    assert [ast.dump(x, include_attributes=False) for x in block] == [ast.dump(x, include_attributes=False) for x in expected]
    fn.body[index:index+2] = ast.parse('libtriton_hash = _flashpair_verified_hash(os.path.join(TRITON_PATH, "_C/libtriton.so"))').body
    namespace = dict(vars(module), _flashpair_verified_hash=verified_hash)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<flashpair036-identical-triton-key>', 'exec'), namespace)
    return namespace['triton_key']

def install(module):
    if getattr(module, '_flashpair036', False):
        return
    module.triton_key = replacement(module)
    module._flashpair036 = True
    def report():
        if COUNTS['candidate_calls'] or COUNTS['fallback_calls']:
            print('FLASHP_MPNN036 ' + json.dumps(COUNTS, sort_keys=True), flush=True)
    atexit.register(report)
