import resource, subprocess, sys
rc = subprocess.call(sys.argv[1:])
print(f"[rss] child max RSS {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20:.2f} GiB, rc {rc}", flush=True)
sys.exit(rc)
