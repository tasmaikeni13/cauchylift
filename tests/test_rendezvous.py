import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

def _test_rank(index):
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    if rank == 0:
        print(f"Cluster size: {world_size} ranks. Testing rendezvous...")
    xm.rendezvous("test_sync_1")
    if rank == 0:
        print("Rendezvous 1 passed!")
    xm.rendezvous("test_sync_2")
    if rank == 0:
        print("Rendezvous 2 passed!")

if __name__ == "__main__":
    xmp.spawn(_test_rank, args=())
