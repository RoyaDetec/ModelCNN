import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import math
from google.colab import drive

drive.mount('/content/drive')

tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU configurada correctamente: {len(gpus)} GPU(s) disponible(s)")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
else:
    print("No se detectó GPU, usando CPU")

DATASET_PATH = "/content/drive/MyDrive/dataset"
IMG_SIZE = 224
BATCH_SIZE = 16
FINE_TUNE_BATCH_SIZE = 8
NUM_CLASSES = 5
EPOCHS_INITIAL = 50
EPOCHS_FINE_TUNE = 40
INITIAL_LR = 0.001
FINE_TUNE_LR = 1e-5

class_names = ['level0', 'level1', 'level2', 'level3', 'level4']
train_counts = [420, 228, 124, 242, 99]
val_counts = [90, 57, 30, 10, 5]
test_counts = [90, 57, 30, 10, 5]

print(f"Distribución del dataset:")
print(f"Train: {train_counts} (Total: {sum(train_counts)})")
print(f"Validation: {val_counts} (Total: {sum(val_counts)})")
print(f"Test: {test_counts} (Total: {sum(test_counts)})")

class_weights_sklearn = compute_class_weight(
    'balanced',
    classes=np.arange(NUM_CLASSES),
    y=np.repeat(np.arange(NUM_CLASSES), train_counts)
)

class_weights = {i: weight for i, weight in enumerate(class_weights_sklearn)}
class_weights[3] *= 1.8
class_weights[4] *= 2.5

print("Pesos de clase calculados:")
for i, weight in class_weights.items():
    print(f"  {class_names[i]}: {weight:.3f}")

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=True,
        seed=42
    )

    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'validation'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=False,
        seed=42
    )

    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=False,
        seed=42
    )

    return train_generator, validation_generator, test_generator

def create_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

def exponential_decay(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * math.exp(-0.1)

def create_callbacks(phase="initial"):
    if phase == "initial":
        checkpoint_path = "coffee_rust_mobilenet_initial.h5"
        patience = 12
        min_delta = 0.001
    else:
        checkpoint_path = "coffee_rust_mobilenet_final.h5"
        patience = 15
        min_delta = 0.0005

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            min_delta=min_delta,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1
        )
    ]

    if phase == "initial":
        callbacks.append(
            LearningRateScheduler(exponential_decay, verbose=1)
        )

    return callbacks

print("Creando generadores de datos...")
train_gen, val_gen, test_gen = create_data_generators()

print(f"Imágenes encontradas:")
print(f"  Train: {train_gen.samples}")
print(f"  Validation: {val_gen.samples}")
print(f"  Test: {test_gen.samples}")

print("Creando modelo...")
model, base_model = create_model()

print("Arquitectura del modelo:")
model.summary()

print("Compilando modelo para entrenamiento inicial...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Iniciando entrenamiento inicial (base congelada)...")
initial_callbacks = create_callbacks("initial")

history_initial = model.fit(
    train_gen,
    epochs=EPOCHS_INITIAL,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=initial_callbacks,
    verbose=1
)

print("Cargando el mejor modelo del entrenamiento inicial...")
model.load_weights("coffee_rust_mobilenet_initial.h5")

print("Evaluación después del entrenamiento inicial:")
initial_val_loss, initial_val_acc = model.evaluate(val_gen, verbose=0)
print(f"Validation Accuracy: {initial_val_acc:.4f}")
print(f"Validation Loss: {initial_val_loss:.4f}")

print("Preparando para fine-tuning...")
base_model.trainable = True

fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

total_layers = len(base_model.layers)
trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
print(f"Capas totales en base_model: {total_layers}")
print(f"Capas entrenables en fine-tuning: {trainable_layers}")

print("Recompilando modelo para fine-tuning...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_gen.batch_size = FINE_TUNE_BATCH_SIZE
val_gen.batch_size = FINE_TUNE_BATCH_SIZE

print("Iniciando fine-tuning...")
fine_tune_callbacks = create_callbacks("fine_tune")

history_fine_tune = model.fit(
    train_gen,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=fine_tune_callbacks,
    verbose=1
)

print("Cargando el mejor modelo del fine-tuning...")
model.load_weights("coffee_rust_mobilenet_final.h5")

print("Evaluación final en conjunto de validación:")
final_val_loss, final_val_acc = model.evaluate(val_gen, verbose=0)
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

print("Evaluación en conjunto de test:")
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

print("Guardando modelo final...")
model.save("coffee_rust_mobilenet_final.h5")
print("Modelo guardado como: coffee_rust_mobilenet_final.h5")

print("Convirtiendo a TensorFlow Lite...")

def representative_dataset():
    test_gen.reset()
    for i, (images, _) in enumerate(test_gen):
        if i >= 10:
            break
        for image in images:
            yield [np.expand_dims(image, axis=0).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

print("Aplicando cuantización INT8...")
tflite_model_quant = converter.convert()

tflite_filename = "coffee_rust_mobilenet_final.tflite"
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model_quant)

print(f"Modelo TFLite guardado como: {tflite_filename}")
print(f"Tamaño del modelo TFLite: {len(tflite_model_quant) / 1024:.2f} KB")

print("Verificando modelo TFLite...")
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Detalles del modelo TFLite:")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Input type: {input_details[0]['dtype']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output type: {output_details[0]['dtype']}")

print("\nResumen del entrenamiento:")
print(f"Accuracy inicial (solo cabeza): {initial_val_acc:.4f}")
print(f"Accuracy final (con fine-tuning): {final_val_acc:.4f}")
print(f"Mejora obtenida: {final_val_acc - initial_val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

if test_acc >= 0.90:
    print("🎉 ¡EXCELENTE! El modelo superó el 90% de accuracy objetivo!")
elif test_acc >= 0.80:
    print("✅ ¡ÉXITO! El modelo alcanzó el objetivo mínimo de 80% de accuracy!")
else:
    print("⚠️  El modelo no alcanzó el 80% objetivo. Considera:")
    print("   - Incrementar épocas de fine-tuning")
    print("   - Ajustar pesos de clase para clases minoritarias")
    print("   - Recolectar más datos para level3 y level4")

print("Entrenamiento y conversión completados exitosamente!")