# Clasificador de Residuos con YOLOv8, OpenCV y Tkinter

Este proyecto implementa un **sistema de clasificación de residuos en tiempo real** utilizando **visión por computadora e inteligencia artificial**.  
A través de una cámara y una interfaz gráfica con **Tkinter**, el modelo identifica si un objeto corresponde a un **metal, plástico, cartón o vidrio**.

---

## Tecnologías utilizadas

- **Python**  
- **OpenCV** – Para el procesamiento de imágenes y captura de la cámara  
- **GroundingDINO** – Para el etiquetado automático del dataset  
- **Roboflow** – Para el aumento de datos y robustecimiento del dataset  
- **YOLOv8 (Ultralytics)** – Para realizar el entrenamiento y obtener el modelo entrenado  
- **Tkinter** – interfaz gráfica simple para visualización  
- **PIP** – gestor de dependencias

---

## Estructura del proyecto

```
PROYECTO-IA
│
├── Modelos                # Modelo YOLOv8 entrenado
├── setUp                  # Archivos de configuración para la ventana de Tkinter
├── main.py                # Archivo principal, lanza la aplicación
├── requirements.txt       # Dependencias para ejecución en CPU
├── requirements-gpu.txt   # Dependencias para ejecución con GPU (CUDA)
├── .gitignore
└── README.md
```
## Instalación

1. Clonar o descargar el proyecto:

```
git clone https://github.com/usuario/proyecto-ia.git
cd PROYECTO-IA
```

2. (Opcional) Crear un entorno virtual:

```
python -m venv venv
```

En Linux/Mac:

```
source venv/bin/activate
```

En Windows:

```
venv\Scripts\activate
```

3. Instalar dependencias según tu entorno:  

- Para CPU (si no cuentas con GPU NVIDIA o CUDA):  

```
pip install -r requirements.txt
```

- Para GPU con CUDA (ejecución más rápida en tarjetas NVIDIA):  

```
pip install -r requirements-gpu.txt
```

---

## Ejecución

En la carpeta principal del proyecto, ejecutar:

```
python main.py
```

Esto abrirá una ventana Tkinter y activará la cámara.  
Cuando se coloque un objeto frente a la cámara, el sistema mostrará en pantalla si es metal, plástico, cartón o vidrio.

---

## Ejemplo de uso

Al iniciar el programa se abre una interfaz gráfica con la cámara en tiempo real.  
Si se coloca un objeto frente a la cámara, el sistema lo clasifica automáticamente.  

Se recomienda incluir capturas de pantalla o gifs de la ejecución en esta sección.

---

## Entrenamiento del modelo

1. Dataset:  
   - Descargado de Internet.  
   - Etiquetado con GroundingDINO.  
   - Aumentado y procesado con Roboflow.

2. Entrenamiento:  
   - Modelo: YOLOv8.  
   - Clases: metal, plástico, cartón, vidrio.  
   - Resultados satisfactorios para clasificación en tiempo real.

---

## Documentación adicional

Para más detalles sobre la instalación y ejecución, revisar el archivo:  
Manual de ejecución disponible en la carpeta docs.

---

## Autor

Proyecto desarrollado por [Oscar Briones] – Estudiante de la Escuela Superior de Cómputo (ESCOM-IPN).
