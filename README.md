# TemporalCNN para el concurso de GDG on Campus UAM

La entrega clasifica series temporales de cuatro canales y 256 pasos en ocho clases. El motor C aplica el preprocesamiento V024 y ejecuta la TemporalCNN con los pesos incluidos. Usa AVX2/FMA cuando está disponible y una implementación escalar en los demás casos.

## Evaluar un clon limpio

El concurso carga `Model()` desde [model.py](model.py). La inferencia requiere Python 3.10+, NumPy y Linux con un compilador C (`cc`, o el ejecutable indicado en `CC`). Durante la inicialización se compila el motor en un directorio temporal y se cargan los pesos de `model/weights.bin`. El modelo no descarga archivos durante la evaluación.

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
curl -fL -o data/test_demo.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/test_demo.npz
python harness.py evaluate --model model.py --source file:data/test_demo.npz
```

El comando usa las 400 muestras de demostración. Para probar el límite de tiempo con el conjunto de entrenamiento, descárgalo según [data/README.md](data/README.md) y ejecuta `python harness.py evaluate --model model.py`.

[Instrucciones oficiales de entrega](https://github.com/GDG-UAM/ml-contest-materials#your-submission). Una vez publicado el repositorio de GitHub, comprueba que los organizadores tienen acceso al clon y envíales el enlace por correo según esas instrucciones.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `model.py`, `requirements.txt` | Punto de entrada y dependencia de inferencia |
| `native/` | Código C, preprocesamiento y carga desde Python |
| `model/` | Pesos nativos, checkpoint PyTorch, grafo ONNX y metadatos |
| `src/` | Arquitectura PyTorch, preparación de datos y entrenamiento |
| `tools/` | Exportación de pesos y comparación con ONNX Runtime |
| `harness.py`, `competition_api.py`, `colab.ipynb` | Material oficial de evaluación |
| `data/`, `examples/` | Instrucciones de descarga y ejemplos |

## Reproducir el modelo

Instala las dependencias de desarrollo con `python -m pip install -r src/requirements.txt`. Para regenerar los pesos nativos del grafo incluido, ejecuta `python -m tools.export_c`. Para regenerar el grafo desde el checkpoint PyTorch, ejecuta primero `python -m tools.export_onnx` y luego vuelve a exportar los pesos nativos. Ambos comandos actualizan los archivos en `model/`.

```bash
python -m tools.validate_native --dataset data/test_demo.npz
```

Para comparar todas las muestras de entrenamiento y 10.000 entradas sintéticas, descarga los datos de entrenamiento y ejecuta `python -m tools.validate_native --full`. Consulta [src/README.md](src/README.md) para entrenar y exportar otra versión, y [model/README.md](model/README.md) para consultar la procedencia de los pesos incluidos.
