import socket, time
NODELAY = True
for size in (100, 57000):           # 小包 vs 57KB
    s = socket.socket(); s.connect(("127.0.0.1", 9999))
    if NODELAY: s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    p = b"x" * size; ts = []
    for _ in range(20):
        t = time.perf_counter(); s.sendall(p)
        n = 0
        while n < size: n += len(s.recv(65536))
        ts.append((time.perf_counter() - t) * 1000)
    s.close()
    ts = sorted(ts)[2:-2]           # 去掉首尾抖动
    print(f"size={size:6d}  mean={sum(ts)/len(ts):.1f}ms")
