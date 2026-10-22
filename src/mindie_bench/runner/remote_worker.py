# [Role] Own the remote slot's PID/network/IPC namespace and stream process sessions.
# [Invariants] Requests are bounded (96 MiB); owner EOF exits the namespace leader so
#   bwrap reaps detached descendants; each stream has one writer and one identity.
# [Timeouts & Locks] Inner command timeout is bounded by the caller; the outer worker
#   applies caller timeout + 20s so post-timeout inventory and JSON flush complete.
# [Side-Effect Policy] Executes only framed slot commands inside the isolated container
#   namespace; writes no host files.

import base64
import json
import os
import queue
import select
import subprocess
import sys
import threading
import uuid

MAX_STREAM_BYTES=8*1024*1024
MAX_REQUEST_BYTES=96*1024*1024
requests=queue.Queue()
streams={}
def read_owner():
    while True:
        line=sys.stdin.buffer.readline(MAX_REQUEST_BYTES+1)
        if not line:
            break
        if len(line)>MAX_REQUEST_BYTES:
            requests.put({"frame_error":"remote request exceeds evidence bound"})
            while line and not line.endswith(b"\n"):
                line=sys.stdin.buffer.readline(MAX_REQUEST_BYTES+1)
            continue
        requests.put(line)
    # EOF is owner loss, including while a command is running. Exiting the
    # PID-namespace leader lets bubblewrap kill all its detached descendants.
    os._exit(0)
def write_stream(state):
    try:
        while True:
            data=state["input"].get()
            if data is None:
                state["process"].stdin.close()
                return
            try:
                remaining=memoryview(data)
                while remaining:
                    size=os.write(state["process"].stdin.fileno(),remaining[:65536])
                    if not size:
                        raise BrokenPipeError("remote stream accepted no input bytes")
                    remaining=remaining[size:]
            finally:
                with state["lock"]:
                    state["pending"]-=len(data)
    except Exception as exc:
        with state["lock"]:
            state["write_error"]=type(exc).__name__+": "+str(exc)
def open_stream(command):
    if not isinstance(command,str) or len(command.encode())>1024*1024:
        raise ValueError("invalid remote stream command")
    if len(streams)>=32:
        raise ValueError("remote stream process bound exceeded")
    process=subprocess.Popen(["/bin/bash","-c",command],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    state=dict(process=process,input=queue.Queue(maxsize=32),lock=threading.Lock(),
        pending=0,eof=False,write_error=None,pipes={})
    for name in ["stdout","stderr"]:
        pipe=getattr(process,name)
        os.set_blocking(pipe.fileno(),False)
        state["pipes"][pipe.fileno()]=(name,pipe)
    identity=uuid.uuid4().hex
    streams[identity]=state
    threading.Thread(target=write_stream,args=(state,),daemon=True).start()
    return {"stream_id":identity}
def exchange_stream(request):
    identity=request["stream_id"]
    state=streams[identity]
    data=base64.b64decode(request.get("stdin_b64",""),validate=True)
    eof=request.get("eof",False)
    if type(eof) is not bool or len(data)>MAX_STREAM_BYTES:
        raise ValueError("invalid or oversized remote stream input")
    with state["lock"]:
        if state["write_error"]:
            raise RuntimeError(state["write_error"])
        if state["eof"] and (data or eof):
            raise ValueError("remote stream input is already closed")
        needed=int(bool(data))+int(eof)
        if (state["pending"]+len(data)>MAX_STREAM_BYTES
                or state["input"].qsize()+needed>32):
            raise ValueError("remote stream pending input bound exceeded")
        state["pending"]+=len(data)
        if data:
            state["input"].put_nowait(data)
        if eof:
            state["eof"]=True
            state["input"].put_nowait(None)
    output={"stdout":[],"stderr":[]}
    size=0
    while state["pipes"] and size<MAX_STREAM_BYTES:
        ready,_,_=select.select(list(state["pipes"]),[],[],.05 if not size else 0)
        if not ready:
            break
        for fd in ready:
            try:
                chunk=os.read(fd,min(65536,MAX_STREAM_BYTES-size))
            except BlockingIOError:
                continue
            if chunk:
                output[state["pipes"][fd][0]].append(chunk)
                size+=len(chunk)
            else:
                state["pipes"].pop(fd)[1].close()
            if size>=MAX_STREAM_BYTES:
                break
    with state["lock"]:
        if state["write_error"]:
            raise RuntimeError(state["write_error"])
    code=state["process"].poll()
    if state["pipes"]:
        code=None
    elif code is not None:
        with state["lock"]:
            if not state["eof"] and not state["pending"]:
                state["eof"]=True
                state["input"].put_nowait(None)
        streams.pop(identity)
    return dict(exit_code=code,
        stdout_b64=base64.b64encode(b"".join(output["stdout"])).decode(),
        stderr_b64=base64.b64encode(b"".join(output["stderr"])).decode())
threading.Thread(target=read_owner, daemon=True).start()
ready={"ready":True}
marker=os.environ.get("REMOTE_DEV_JOB_TOKEN")
if os.environ.get("MINDIE_BENCH_COMPONENT_GUARD")=="1":
    boot_id=open("/proc/sys/kernel/random/boot_id").read().strip()
    fields=open("/proc/self/stat").read().rsplit(") ",1)[1].split()
    guard=dict(marker=marker,boot_id=boot_id,retain_until_release=True)
    ready["owner_receipt"]=dict(pid=os.getpid(),start_ticks=fields[19],
        boot_id=boot_id,process_guard=guard,pid_namespace=os.readlink("/proc/self/ns/pid"))
print(json.dumps(ready), flush=True)
while True:
    line=requests.get()
    try:
        if isinstance(line,dict):
            raise ValueError(line["frame_error"])
        request = json.loads(line)
        if request.get("control")=="quiesce":
            for item in os.listdir("/proc"):
                if item.isdigit() and int(item) not in {os.getpid(),os.getppid()}:
                    try:
                        os.kill(int(item),9)
                    except ProcessLookupError:
                        pass
            print(json.dumps({"quiesced":True}),flush=True)
            continue
        if request.get("control")=="open_stream":
            value=open_stream(request["command"])
        elif request.get("control")=="exchange_stream":
            value=exchange_stream(request)
        else:
            outer_timeout = float(request["timeout"]) + 20.0 if request.get("timeout") is not None else None
            result = subprocess.run(["/bin/bash", "-c", request["command"]],
                input=base64.b64decode(request["stdin_b64"]), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=outer_timeout)
            value = dict(exit_code=result.returncode, stdout_b64=base64.b64encode(result.stdout).decode(),
                stderr_b64=base64.b64encode(result.stderr).decode())
    except Exception as exc:
        value = dict(error=type(exc).__name__, message=str(exc))
    print(json.dumps(value), flush=True)
