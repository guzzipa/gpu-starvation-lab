
# Valores originales que hacían "morir" a la Air:
# BATCH_SIZE = 32, IMG_SIZE = 1024, SAMPLES = 500


import time
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from data_utils import FakeDataset

# Detección de dispositivo
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# CONFIGURACIÓN SEGURA PARA MACBOOK AIR M4
BATCH_SIZE = 16
SAMPLES = 100 
IMG_SIZE = 256

def run_test(name, num_workers=0):
    print(f"Probando: {name}...")
    model = models.resnet18().to(DEVICE).train()
    # Cargamos el dataset
    dataset = FakeDataset(size=IMG_SIZE, samples=SAMPLES)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers,
        # En Mac, a veces 'fork' es más estable que 'spawn' para benchmarks simples
        multiprocessing_context='fork' if num_workers > 0 else None
    )

    start_time = time.perf_counter()
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()

    throughput = SAMPLES / (time.perf_counter() - start_time)
    return throughput

# PROTECCIÓN NECESARIA PARA MULTIPROCESSING
if __name__ == '__main__':
    print(f"Iniciando benchmark en {DEVICE}...")
    
    # Prueba 1: Baseline
    val_baseline = run_test("Baseline (0 workers)", num_workers=0)
    
    # Respiro térmico para la MacBook Air
    print("Enfriando chip...")
    time.sleep(5) 
    
    # Prueba 2: Optimized
    val_optimized = run_test("Optimized (4 workers)", num_workers=4)

    print("\n" + "="*30)
    print(f"RESULTADOS FINALES")
    print("="*30)
    print(f"Baseline:  {val_baseline:.2f} img/s")
    print(f"Optimized: {val_optimized:.2f} img/s")
    
    mejora = ((val_optimized - val_baseline) / val_baseline) * 100
    print(f"Mejora:    {mejora:.2f}%")
    print("="*30)