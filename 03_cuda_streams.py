import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Usando dispositivo: {DEVICE}")
import torchvision.models as models
from torch.utils.data import DataLoader
from data_utils import FakeDataset
import time

DEVICE = "cuda"
model = models.resnet18().to(DEVICE).train()
loader = DataLoader(FakeDataset(1024), batch_size=32, num_workers=4, pin_memory=True)

# Creamos dos streams diferentes
compute_stream = torch.cuda.Stream()
copy_stream = torch.cuda.Stream()

start = time.perf_counter()

# Lógica de Pipelining
data_iter = iter(loader)
batch = next(data_iter)

# Primer copiado
with torch.cuda.stream(copy_stream):
    inputs, labels = batch[0].to(DEVICE, non_blocking=True), batch[1].to(DEVICE, non_blocking=True)

for i in range(len(loader) - 1):
    copy_stream.synchronize() # Asegura que la copia terminó antes de computar
    
    with torch.cuda.stream(compute_stream):
        outputs = model(inputs) # Computa lote N
        
    batch = next(data_iter)
    with torch.cuda.stream(copy_stream):
        inputs, labels = batch[0].to(DEVICE, non_blocking=True), batch[1].to(DEVICE, non_blocking=True) # Copia lote N+1

torch.cuda.synchronize()
end = time.perf_counter()
print(f"Streams Throughput: {1000/(end-start):.2f} images/sec")