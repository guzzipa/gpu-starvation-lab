import time
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from data_utils import FakeDataset

# Selección automática de dispositivo
if torch.backends.mps.is_available(): DEVICE = "mps"
elif torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

model = models.resnet18().to(DEVICE).train()
dataset = FakeDataset(size=1024, samples=200)
loader = DataLoader(dataset, batch_size=32) # Sin optimizaciones

start = time.perf_counter()
for inputs, labels in loader:
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()

end = time.perf_counter()
print(f"Throughput: {200/(end-start):.2f}")