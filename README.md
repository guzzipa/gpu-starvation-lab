# 🚀 PyTorch GPU Optimization Lab: CUDA vs. Apple Silicon M4

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio es un laboratorio experimental para medir el **GPU Starvation** (inanición de la GPU) y el impacto de la arquitectura de memoria en el rendimiento de modelos de Deep Learning. Inspirado en la metodología de **NVIDIA Nsight Systems**, adaptada para un entorno multiplataforma.

---

## 🧐 El Experimento: ¿Por qué optimizar el Data Loading?

En el entrenamiento de modelos de IA, la GPU suele ser el recurso más caro. El objetivo es que la GPU nunca esté ociosa. Tradicionalmente, esto se logra usando `multiprocessing` para que la CPU prepare el lote `N+1` mientras la GPU computa el lote `N`.

Este laboratorio compara un pipeline **Baseline** (secuencial) contra uno **Optimizado** (paralelo) en dos arquitecturas radicalmente distintas:
1.  **NVIDIA (CUDA):** Arquitectura de memoria discreta (comunicación vía bus PCIe).
2.  **Apple Silicon (MPS):** Arquitectura de **Memoria Unificada** (CPU y GPU comparten el mismo chip y silicio).

---

## 💻 Resultados del Benchmark (MacBook Air M4)



En las pruebas realizadas localmente en una **MacBook Air M4**, obtuvimos un resultado contraintuitivo que demuestra la eficiencia de la memoria unificada:

| Estrategia | Configuración | Throughput (img/s) | Resultado |
| :--- | :--- | :--- | :--- |
| **Baseline** | 0 Workers (Single Process) | **91.63** | 🟢 Ganador |
| **Optimized** | 4 Workers (Multiprocessing) | **54.85** | 🔴 40.14% más lento |

### 🔍 Análisis Técnico del M4:
* **Overhead de Multiprocesamiento:** En Apple Silicon, la latencia de memoria es tan baja que el tiempo necesario para que Python gestione procesos hijos (`spawn/fork`) es mayor que la ganancia de velocidad.
* **Memoria Unificada:** Al no haber un bus PCIe de por medio, un solo proceso es capaz de saturar el ancho de banda necesario para un modelo como ResNet-18 con imágenes de 256px.
* **Lección:** La "sobre-optimización" puede penalizar el rendimiento en hardware de consumo eficiente.

---

## 🛠️ Cómo ejecutar este Laboratorio

### 1. Requisitos
- Python 3.11+
- MacBook con chip M1/M2/M3/M4 o PC con GPU NVIDIA.

### 2. Instalación
```
bash
git clone [https://github.com/guzzipa/gpu-starvation-lab.git](https://github.com/guzzipa/gpu-starvation-lab.git)
cd gpu-starvation-lab
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision
```
