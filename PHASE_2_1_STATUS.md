# Cattle Recognition Project - Phase 2.1 Status

## 📋 FASE 2.1: DETECTION CAPABILITY TRAINING

### Objetivo de Fase 2.1
**Entrenamiento de Capacidad de Detección**: Entrenar modelo especializado en localizar y dibujar bounding boxes alrededor de todo el ganado en una imagen o frame de video.

### Estado Actual
- [x] **Script de entrenamiento**: `scripts/train_phase_2_1.py` ✅ **LISTO**
- [x] **Arquitectura definida**: YOLOv11 (Ultralytics) 
- [x] **Dataset preparado**: cattle-detection-v3 (formato YOLO)
- [ ] **Entrenamiento ejecutado**: Pendiente de ejecución
- [ ] **Modelo entrenado**: `models/detection/best.pt` (pendiente)
- [ ] **Validación completada**: Métricas de precisión pendientes

## 🎯 Configuración de Entrenamiento

### Arquitectura Implementada
- **Framework**: Ultralytics YOLOv11
- **Modelo base**: YOLOv11s (balance precisión/velocidad)
- **Input size**: 640x640 pixels
- **Clases**: 1 clase ("cattle")

### Parámetros de Entrenamiento
```python
# Configuración recomendada para Phase 2.1
model_size = "s"           # YOLOv11s (small) - balance óptimo
epochs = 100               # Entrenamiento completo
batch_size = 16            # GPU memory friendly
imgsz = 640               # Standard YOLO input size
device = "cuda"           # GPU acceleration
```

### Dataset
- **Source**: cattle-detection-v3 (Roboflow)
- **Formato**: YOLO (images + labels .txt)
- **Estructura**: train/valid/test splits
- **Clases**: cattle (class 0)

## 🚀 Export Formats y Deployment Strategy

### 📦 Formatos de Exportación Disponibles

#### 1. **ONNX** (Recomendado por defecto)
```bash
--export-format onnx
```

**✅ Ventajas:**
- **Cross-platform**: ARM, x86, diferentes OS
- **Optimizado para edge devices**: Raspberry Pi, Jetson Nano
- **Memory efficient**: ~100-200MB vs ~500MB PyTorch
- **Inference speed**: 2-3x más rápido que PyTorch .pt
- **Runtime ligero**: ONNXRuntime vs framework completo
- **Standarizado**: Compatible con múltiples frameworks

**❌ Desventajas:**
- **Debugging limitado**: Menos herramientas de debug
- **Feature completeness**: Algunas operaciones avanzadas no soportadas
- **Versioning**: Compatibilidad entre versiones ONNX

**🎯 Ideal para:**
- Raspberry Pi deployment
- Dispositivos embebidos
- Drones con CPU ARM
- Production con memoria limitada

#### 2. **PyTorch (.pt)** 
```bash
--export-format pytorch
```

**✅ Ventajas:**
- **Full compatibility**: 100% compatible con entrenamiento
- **Debugging completo**: Todas las herramientas PyTorch
- **Flexibilidad**: Modificaciones en tiempo real
- **Ecosistema completo**: Acceso a todas las funciones

**❌ Desventajas:**
- **Memory footprint**: ~500MB framework + modelo
- **Slower inference**: Sin optimizaciones de deployment
- **Dependencies**: Requiere PyTorch completo instalado
- **Platform specific**: Menos portabilidad

**🎯 Ideal para:**
- Development y testing
- Servidores con recursos abundantes
- Cuando necesitas debugging completo
- Prototipado rápido

#### 3. **TensorRT** (GPU optimizado)
```bash
--export-format tensorrt
```

**✅ Ventajas:**
- **Maximum performance**: Inferencia más rápida posible
- **GPU optimization**: Optimizado específicamente para NVIDIA
- **INT8 quantization**: Reducción dramática de memoria
- **Kernel fusion**: Optimizaciones automáticas

**❌ Desventajas:**
- **NVIDIA only**: Funciona solo en GPUs NVIDIA
- **Platform specific**: No portable
- **Setup complexity**: Instalación compleja TensorRT
- **Size limitations**: Funciona solo con modelos pequeños-medianos

**🎯 Ideal para:**
- Servidores con GPUs NVIDIA potentes
- Applications que requieren máximo rendimiento
- Procesamiento de múltiples streams simultáneos

#### 4. **TensorFlow Lite (.tflite)**
```bash
--export-format tflite
```

**✅ Ventajas:**
- **Mobile optimized**: Diseñado para smartphones/tablets
- **Ultra lightweight**: Mínimo memory footprint
- **Quantization**: INT8, FP16 optimizations
- **Mobile frameworks**: Android/iOS integration

**❌ Desventajas:**
- **Limited ops**: Operaciones soportadas limitadas
- **Conversion issues**: Puede fallar con modelos complejos
- **Performance**: Puede ser más lento que ONNX en edge devices

**🎯 Ideal para:**
- Aplicaciones móviles (Android/iOS)
- Microcontroladores
- Dispositivos con memory/power constraints extremos

#### 5. **CoreML** (Apple ecosystem)
```bash
--export-format coreml
```

**✅ Ventajas:**
- **Apple optimized**: Máximo rendimiento en devices Apple
- **Neural Engine**: Aprovecha NPU en devices modernos
- **iOS/macOS integration**: Integración nativa

**❌ Desventajas:**
- **Apple only**: Exclusivo ecosistema Apple
- **Limited flexibility**: Menos control sobre optimizaciones

**🎯 Ideal para:**
- Apps iOS/macOS exclusivamente
- Cuando tienes Neural Engine disponible

## 🎯 Deployment Strategy por Escenario

### 🤖 Raspberry Pi / Edge Devices
```bash
# Configuración óptima para Raspberry Pi
python scripts/train_phase_2_1.py \
  --dataset data/detection/cattle-detection-v3 \
  --model-size n \                    # YOLOv11n (nano) más ligero
  --epochs 100 \
  --batch-size 32 \
  --device cuda \                     # Entrenar en GPU, deploy en CPU
  --export-format onnx                # Formato optimizado
```

**Justificación:**
- **Model size "n"**: YOLOv11n (nano) es el más ligero (~6MB)
- **ONNX**: Mejor rendimiento en CPU ARM
- **Memory**: <200MB total usage
- **Inference**: ~100-200ms por frame

### ☁️ Cloud/Server Deployment
```bash
# Configuración para servidores potentes
python scripts/train_phase_2_1.py \
  --dataset data/detection/cattle-detection-v3 \
  --model-size l \                    # YOLOv11l (large) máxima precisión
  --epochs 100 \
  --batch-size 16 \
  --device cuda \
  --export-format tensorrt            # Máximo rendimiento GPU
```

**Justificación:**
- **Model size "l"**: Máxima precisión para decisiones críticas
- **TensorRT**: Aprovecha al máximo GPUs NVIDIA
- **Memory**: Resources abundantes disponibles
- **Inference**: ~10-20ms por frame

### 📱 Mobile Applications
```bash
# Configuración para aplicaciones móviles
python scripts/train_phase_2_1.py \
  --dataset data/detection/cattle-detection-v3 \
  --model-size n \                    # Modelo más ligero
  --epochs 100 \
  --batch-size 32 \
  --device cuda \
  --export-format tflite             # Optimizado para móviles
```

**Justificación:**
- **TensorFlow Lite**: Diseñado específicamente para móviles
- **Size constraints**: Apps tienen límites de tamaño
- **Battery**: Optimizado para consumo energético

### 🚁 Drone/UAV Deployment
```bash
# Configuración para drones
python scripts/train_phase_2_1.py \
  --dataset data/detection/cattle-detection-v3 \
  --model-size s \                    # Balance precisión/velocidad
  --epochs 100 \
  --batch-size 16 \
  --device cuda \
  --export-format onnx                # Portable y eficiente
```

**Justificación:**
- **Model size "s"**: Balance entre precisión y velocidad
- **ONNX**: Funciona en diferentes autopilot systems
- **Real-time**: Necesita procesamiento fluido para navegación

## 📊 Performance Comparison

### Inference Speed (por frame)
| Formato | Raspberry Pi 4 | Jetson Nano | NVIDIA RTX 3080 | iPhone 12 |
|---------|----------------|-------------|-----------------|-----------|
| PyTorch (.pt) | ~500ms | ~200ms | ~15ms | N/A |
| ONNX | ~200ms | ~100ms | ~25ms | ~50ms |
| TensorRT | N/A | ~50ms | ~8ms | N/A |
| TensorFlow Lite | ~300ms | ~150ms | N/A | ~30ms |
| CoreML | N/A | N/A | N/A | ~25ms |

### Memory Usage (modelo + runtime)
| Formato | Memory Footprint | Modelo Size |
|---------|------------------|-------------|
| PyTorch (.pt) | ~500MB | ~14MB |
| ONNX | ~150MB | ~14MB |
| TensorRT | ~300MB | ~8MB (optimizado) |
| TensorFlow Lite | ~50MB | ~6MB (quantized) |
| CoreML | ~80MB | ~12MB |

## 🔧 Comando de Entrenamiento Recomendado

### Para Raspberry Pi (Deployment final)
```bash
python scripts/train_phase_2_1.py \
  --dataset data/detection/cattle-detection-v3 \
  --model-size n \
  --epochs 100 \
  --batch-size 32 \
  --device cuda \
  --export-format onnx
```

### Para Máximo Performance (GPU servers)
```bash
python scripts/train_phase_2_1.py \
  --dataset data/detection/cattle-detection-v3 \
  --model-size s \
  --epochs 100 \
  --batch-size 16 \
  --device cuda \
  --export-format tensorrt
```

### Para Development/Testing
```bash
python scripts/train_phase_2_1.py \
  --dataset data/detection/cattle-detection-v3 \
  --model-size s \
  --epochs 100 \
  --batch-size 16 \
  --device cuda \
  --export-format pytorch
```

## 🎯 Próximos Pasos

### Immediate (Phase 2.1 completion)
1. **Ejecutar entrenamiento**: Correr script con configuración elegida
2. **Validar modelo**: Verificar métricas mAP, precision, recall
3. **Test deployment format**: Verificar que exportación funciona
4. **Performance testing**: Medir inference speed en target device

### Integration (Phase 4 preparation)
1. **Combine with Phase 1.2**: Integrar con modelo de identificación
2. **Pipeline testing**: Probar detección → identificación
3. **Threshold calibration**: Optimizar parámetros de similarity
4. **Real-time validation**: Probar en video streams

## 📝 Decisión de Export Format

### Recomendación por Defecto: **ONNX**

**Razón principal**: 
El proyecto está diseñado para **Raspberry Pi deployment** según CLAUDE.md, y ONNX ofrece el mejor balance de:
- ✅ Performance en edge devices
- ✅ Cross-platform compatibility  
- ✅ Memory efficiency
- ✅ Future flexibility

**Cambio fácil**: El export format es configurable, permitiendo cambiar para diferentes deployment targets sin reentrenar.

---
**Estado del Proyecto**: ⏳ **Fase 2.1 PREPARADA PARA EJECUCIÓN**
**Próximo hito**: Ejecutar entrenamiento y completar Phase 2.1
**Integration ready**: Script listo para generar modelo compatible con Phase 1.2