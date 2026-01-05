import subprocess
import time

scripts = [
    ("Baseline", "01_baseline.py"),
    ("Optimized (Workers + Pin)", "02_optimized.py"),
    ("Expert (CUDA Streams)", "03_cuda_streams.py")
]

print(f"{'Metodo':<30} | {'Throughput (img/s)':<20}")
print("-" * 55)

for nombre, script in scripts:
    try:
        # Ejecutamos el script y capturamos la salida
        result = subprocess.run(["python", script], capture_output=True, text=True)
        
        # Buscamos la línea que tiene el resultado (suponiendo que imprimimos 'Throughput: X')
        output = result.stdout.strip().split('\n')[-1]
        print(f"{nombre:<30} | {output}")
    except Exception as e:
        print(f"Error corriendo {script}: {e}")

print("-" * 55)