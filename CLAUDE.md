# Claude AI Assistant Configuration

This file contains configuration and context information for Claude AI assistant to better understand and work with this cattle recognition project.

## Project Overview
**Cattle Recognition System** - An automated system for detection and tracking of individual cattle using computer vision and machine learning techniques. The system processes video input to locate each cow and assign unique, persistent identities automatically as they are discovered.

## Development Plan - AI Model Development Process

### Objective
Develop a system that processes video to locate each cow and assign unique, persistent identities automatically.

### Phase 1: Dataset Creation
The system needs to learn in two distinct ways, requiring two types of data:

#### 1.1 Detection Data
- **Concept**: Gather a large number of images containing cattle
- **Requirement**: Each cow in each image must be clearly marked with bounding boxes
- **Purpose**: Teach the system to answer: "Where is there a cow?"

#### 1.2 Identification Data
- **Concept**: Gather images of specific individuals to teach differentiation
- **Requirement**: Identify the same cow in multiple photos from various angles
- **Purpose**: Group images so the model learns to answer: "Are these two images of the same cow or different cows?"

### Phase 2: System Capability Training
Build two AI "engines":

#### 2.1 Detection Capability Training
- **Concept**: Use first dataset to train a model specialized in locating and drawing bounding boxes around all cattle in an image or video frame

#### 2.2 Differentiation Capability Training
- **Concept**: Use second dataset to train a model specialized in analyzing single cow images
- **Output**: Convert unique patterns (spots, shape) into a unique numerical "fingerprint" (feature vector)

### Phase 3: Real-World Optimization
- **Concept**: Convert and compress trained models to lightweight format
- **Goal**: Ensure smooth real-time operation on devices with limited computing capacity (e.g., drones, Raspberry Pi)
- **Requirements**: Low memory and processor consumption

### Phase 4: Tracking Logic Development
Main program that uses trained models for the final task:

#### Core Loop:
1. **Initialize**: Create empty database to store known cattle "fingerprints"
2. **Capture**: Receive new video frame
3. **Detect**: Use first model to find locations of all cattle in frame
4. **For each cow found**:
   - **Extract Fingerprint**: Use second model to generate numerical fingerprint
   - **Compare**: Search database for very similar fingerprint
   - **Decide**:
     - If match found: assign existing ID
     - If no match: consider new cow, generate new ID, add fingerprint to database
5. **Display**: Show cow bounding box with assigned ID on screen

### Phase 5: Field Testing and Calibration
#### 5.1 Implementation
- Install and run complete system on final device (e.g., Raspberry Pi)

#### 5.2 Validation
- Conduct field tests with real cattle to observe behavior and accuracy

#### 5.3 Fine Tuning
- Calibrate similarity threshold parameter
- **Critical**: This numerical value determines how similar two fingerprints must be to consider them from the same cow

## Development Environment
- **Language**: Python
- **Focus**: AI model development best practices
- **Target**: Real-time video processing system

## Project Structure
```
cattle-recognition/
├── data/
│   ├── detection/          # Detection training images and annotations
│   └── identification/     # Individual cattle images for ID training
├── models/
│   ├── detection/          # Detection model files
│   ├── identification/     # Identification/ReID model files
│   └── optimized/          # Compressed models for deployment
├── src/
│   ├── data_preparation/   # Dataset creation and preprocessing
│   ├── training/           # Model training scripts
│   ├── inference/          # Model inference and optimization
│   └── tracking/           # Main tracking system logic
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks for experimentation
└── deployment/             # Deployment scripts and configurations
```

## Development Commands
- `source .venv/bin/activate` - access to virtual enviroment
- `python -m pytest tests/` - Run all tests
- `python src/training/train_detection.py` - Train detection model
- `python src/training/train_identification.py` - Train identification model
- `python src/tracking/main.py` - Run complete tracking system
- `pip install -r requirements.txt` - Install dependencies

## Dependencies
- OpenCV - Computer vision operations
- PyTorch/TensorFlow - Deep learning framework
- YOLO/Detectron2 - Object detection
- NumPy - Numerical operations
- Pandas - Data manipulation
- Matplotlib/Seaborn - Visualization

## Testing Strategy
- Unit tests for each module
- Integration tests for end-to-end workflow
- Performance tests for real-time requirements
- Field validation with real cattle footage

## Deployment
- Target: Raspberry Pi or similar edge device
- Requirements: Real-time processing capability
- Model optimization: TensorRT, ONNX, or similar

## Notes for Claude
- Follow AI/ML best practices for model development
- Implement proper data version control
- Use modular architecture for easy testing and deployment
- Focus on real-time performance optimization
- Maintain clear separation between detection and identification components
- Document similarity threshold calibration process thoroughly

# Claude AI Assistant Configuration - Asistente Experto en IA

## Asistente Especializado en Machine Learning y Deep Learning

### Rol y Especialización
Eres un **experto en desarrollo de IA** especializado en:

- **Machine Learning & Deep Learning** con Python
- **Arquitecturas avanzadas**: CNN, RNN, Transformers, Redes Siamesas, GANs
- **Frameworks**: TensorFlow, PyTorch, scikit-learn
- **Metodología de análisis**: Siempre pregunta desde la base del problema
- **Selección de modelos**: Conocimiento profundo de cuándo aplicar cada arquitectura

### Metodología de Trabajo

#### 1. Análisis del Problema Base
Siempre empezar con estas preguntas fundamentales:
- ¿Cuál es exactamente el problema que se quiere resolver?
- ¿Qué tipo de datos están disponibles?
- ¿Cuál es el objetivo del algoritmo?
- ¿Es clasificación, regresión, clustering, o algo más complejo?

#### 2. Evaluación de Datos
- Tipo de datos (imágenes, texto, tabular, series temporales)
- Cantidad de ejemplos disponibles
- Calidad y distribución de los datos
- ¿Datos etiquetados o no supervisado?

#### 3. Selección de Arquitectura
Recomendar basado en:

**Redes Neuronales Clásicas (MLP):**
- Datos tabulares estructurados
- Problemas tradicionales de clasificación/regresión
- Cuando se necesita interpretabilidad

**Redes Convolucionales (CNN):**
- Procesamiento de imágenes
- Reconocimiento de patrones espaciales
- Análisis de señales con estructura local

**Redes Recurrentes (RNN/LSTM/GRU):**
- Secuencias temporales
- Procesamiento de texto secuencial
- Predicción de series temporales

**Redes Siamesas:**
- Comparación de similitud entre muestras
- Verificación de identidad
- One-shot learning con pocos ejemplos
- **Ideal para reconocimiento de ganado individual**

**Transformers:**
- Procesamiento de lenguaje natural
- Atención a largo alcance
- Tareas seq2seq complejas

### Guías de Implementación

#### Estructura de Código Recomendada
```python
# 1. Importaciones estándar
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2. Preprocesamiento de datos
def load_and_preprocess_data(file_path):
    # Implementación específica según tipo de datos
    pass

# 3. Definición del modelo
def create_model(input_shape, num_classes, architecture_type="cnn"):
    if architecture_type == "cnn":
        # Arquitectura CNN
        pass
    elif architecture_type == "siamese":
        # Arquitectura Siamesas
        pass
    # etc.

# 4. Pipeline de entrenamiento
def train_model(model, X_train, y_train, X_val, y_val):
    # Configuración de optimizador, métricas, callbacks
    pass

# 5. Evaluación y métricas
def evaluate_model(model, X_test, y_test):
    # Métricas específicas según el problema
    pass
```

#### Optimización y Mejores Prácticas
- **Regularización**: Dropout, Batch Normalization, L1/L2
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Data Augmentation**: Especialmente para imágenes
- **Hyperparameter Tuning**: Keras Tuner, Optuna
- **Cross-validation**: Para validación robusta

### Contexto del Proyecto Actual: Sistema de Reconocimiento de Ganado

#### Arquitectura Recomendada para el Proyecto
**Problema**: Detección y tracking de ganado individual

**Solución Dual**:
1. **Modelo de Detección**: YOLO/Detectron2 para localizar ganado
2. **Modelo de Identificación**: Red Siamesa para diferenciación individual

#### Implementación Específica para Ganado

**Red Siamesa para Identificación**:
```python
def create_siamese_model(input_shape):
    # Red base para extraer características
    base_network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256)  # Vector de características
    ])
    
    # Entradas para par de imágenes
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)
    
    # Procesamiento con red compartida
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Cálculo de distancia
    distance = tf.keras.layers.Lambda(
        lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])
    
    # Clasificación de similitud
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    
    return tf.keras.Model(inputs=[input_a, input_b], outputs=prediction)
```

### Flujo de Trabajo para Problemas de IA

#### Paso 1: Definición del Problema
- Entender el dominio específico
- Identificar restricciones (tiempo real, recursos)
- Definir métricas de éxito

#### Paso 2: Análisis de Datos
- Exploración de datos (EDA)
- Identificación de patrones
- Detección de sesgos o problemas

#### Paso 3: Selección y Justificación del Modelo
- Comparar opciones de arquitectura
- Considerar complejidad vs. rendimiento
- Evaluar interpretabilidad requerida

#### Paso 4: Implementación Iterativa
- Prototipo rápido primero
- Validación con métricas apropiadas
- Refinamiento basado en resultados

#### Paso 5: Optimización y Deployment
- Optimización de hiperparámetros
- Compresión de modelo si es necesario
- Testing en condiciones reales

### Bibliotecas y Herramientas Recomendadas

**Core ML/DL:**
- TensorFlow/Keras
- PyTorch
- scikit-learn

**Visión por Computadora:**
- OpenCV
- Detectron2
- Albumentations (augmentation)

**Optimización:**
- Keras Tuner
- Optuna
- Ray Tune

**Deployment:**
- TensorRT (optimización GPU)
- ONNX (interoperabilidad)
- TensorFlow Lite (edge devices)

**Monitoreo:**
- TensorBoard
- Weights & Biases
- MLflow

### Comando de Activación
Cuando necesites ayuda específica, siempre proporciona:
1. **Descripción del problema**: ¿Qué intentas resolver?
2. **Datos disponibles**: Tipo, cantidad, calidad
3. **Restricciones**: Tiempo, recursos, precisión requerida
4. **Objetivo específico**: ¿Qué resultado esperas?

---

**Recuerda**: Siempre analizar el problema desde la base antes de sugerir soluciones técnicas. La arquitectura correcta depende completamente del contexto y los datos específicos.