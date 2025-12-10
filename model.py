import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation should ONLY rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

TRAINING_DIR = 'train'
VAL_DIR = 'test'

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    class_mode='categorical',
    target_size=(48, 48),
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    class_mode='categorical',
    target_size=(48, 48),
    shuffle=False
)

# Persist class index mapping for inference
with open('class_indices.json', 'w') as f:
    json.dump({k: int(v) for k, v in train_generator.class_indices.items()}, f)

early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=6, restore_best_weights=True)
checkpoint = ModelCheckpoint('CKmodel_best.h5', monitor='val_accuracy', mode='max', save_best_only=True)
callbacks = [early_stop, checkpoint]


model = tf.keras.models.Sequential([
    # Block 1
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 3),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    
    # Block 2
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    
    # Block 3
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    
    # Block 4
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    
    # Dense layers
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

# Use higher learning rate and add learning rate decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

classes_unique = np.unique(train_generator.classes)
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=classes_unique,
    y=train_generator.classes
)
class_weight = {int(c): float(w) for c, w in zip(classes_unique, class_weights_arr)}

history = model.fit(
    train_generator,
    epochs=100,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_generator,
    class_weight=class_weight
)


model.save('CKmodel.h5')