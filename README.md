
---

# Ejercicio 1

[![TP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/C4570/AA2-Menescaldi-Britos/blob/main/Ejercicio_1.ipynb)

---

# Ejercicio 2: Sistema de Clasificación de Gestos (Piedra, Papel o Tijeras)

Este ejercicio tiene como objetivo implementar un sistema para reconocer gestos de **piedra**, **papel** y **tijeras** utilizando **MediaPipe** y un modelo de red neuronal. El sistema consta de tres pasos: grabación del dataset, entrenamiento del modelo, y ejecución del clasificador en tiempo real.

### 🔥 Instrucciones para ejecutar el proyecto:

### 1. **Grabar el Dataset** (`record-dataset.py`)

En esta etapa, grabarás gestos de la mano para crear un dataset de entrenamiento.

1. **Ejecuta** el script `record-dataset.py`:
    ```bash
    python record-dataset.py
    ```

2. Aparecerá la cámara y deberás realizar los gestos correspondientes a:
    - **🪨 Piedra**: Presiona la tecla `1`.
    - **✂️ Tijeras**: Presiona la tecla `2`.
    - **📄 Papel**: Presiona la tecla `3`.

3. Cada vez que presiones uno de estos números, el sistema capturará la posición de tu mano y etiquetará los datos correctamente. 

4. **Instrucciones flotantes** en pantalla te recordarán qué tecla presionar para cada gesto. Si deseas detener la grabación, simplemente presiona la tecla **`q`**.

5. Al finalizar, los datos se guardarán automáticamente en la carpeta **Ejercicio 2** en archivos `.npy`:
   - `rps_dataset.npy`: contiene los landmarks de los gestos grabados.
   - `rps_labels.npy`: contiene las etiquetas correspondientes (0 para piedra, 1 para papel, 2 para tijeras).

### 2. **Entrenar el Modelo** (`train-gesture-classifier.py`)

Con los datos capturados, entrenarás un modelo de red neuronal para clasificar los gestos.

1. **Ejecuta** el script `train-gesture-classifier.py`:
    ```bash
    python train-gesture-classifier.py
    ```

2. El script cargará el dataset generado en la etapa anterior (`rps_dataset.npy` y `rps_labels.npy`), y entrenará una red neuronal con estos datos.

3. El modelo entrenado se guardará automáticamente en un archivo llamado `rps_model.h5`.

### 3. **Probar el Sistema Completo** (`rock-paper-scissors.py`)

Finalmente, pondrás a prueba el modelo entrenado para que clasifique gestos en tiempo real usando la cámara.

1. **Ejecuta** el script `rock-paper-scissors.py`:
    ```bash
    python rock-paper-scissors.py
    ```

2. La cámara se abrirá nuevamente. Al realizar un gesto con la mano frente a la cámara, el sistema utilizará **MediaPipe** para detectar la mano y el modelo entrenado para predecir si estás haciendo **🪨 piedra**, **✂️ tijeras** o **📄 papel**.

3. El gesto reconocido se mostrará en pantalla en tiempo real.

---

# Ejercicio 3

[![TP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FCEIA-AAII/lab5/blob/main/Ejercicio_3.ipynb)

---

## Fuente de datos

Puedes acceder a los datos necesarios para estos ejercicios desde el siguiente enlace:

[Fuente de datos](https://drive.google.com/drive/folders/1ll10p_9Kvsq_PGI7eKPBdEaxUL3fx8Eb?usp=sharing)

---
