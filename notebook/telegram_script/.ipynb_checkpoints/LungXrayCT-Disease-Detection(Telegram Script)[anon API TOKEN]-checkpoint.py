
import asyncio
import numpy as np
import cv2
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import tensorflow as tf
from tensorflow.keras.models import load_model
import nest_asyncio
import time
from telegram.error import NetworkError

# Fix event loop issues in Jupyter Notebook
nest_asyncio.apply()

# Define custom loss function
class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weights, name="WeightedSparseCategoricalCrossentropy"):
        super().__init__(name=name)
        self.weights = tf.convert_to_tensor(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.nn.softmax(y_pred)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weight = tf.gather(self.weights, y_true)
        return loss * weight

    def get_config(self):
        return {"weights": self.weights.numpy().tolist()}

    @classmethod
    def from_config(cls, config):
        weights = np.array(config.get("weights", [1.0, 1.0, 1.0]), dtype=np.float32)
        return cls(weights=weights)

# Custom loss instance
custom_loss = WeightedSparseCategoricalCrossentropy(weights=[1.0, 1.5, 2.0])

# Model paths
PNEUMONIA_MODEL_PATH = r"path"

LUNG_CANCER_MODEL_PATH = r"path"

TUBERCULOSIS_MODEL_PATH = r"path"

# Load models
pneumonia_model = load_model(PNEUMONIA_MODEL_PATH)
lung_cancer_model = load_model(LUNG_CANCER_MODEL_PATH, custom_objects={"WeightedSparseCategoricalCrossentropy": lambda: custom_loss})
tuberculosis_model = load_model(TUBERCULOSIS_MODEL_PATH)

# Class labels
PNEUMONIA_LABELS = ["Normal", "Pneumonia"]
LUNG_CANCER_LABELS = ["Normal", "Benign", "Malignant"]
TUBERCULOSIS_LABELS = ["Normal", "Tuberculosis"]

# Default model (None until the user selects)
selected_model = None

def load_image_from_bytes(byte_image: BytesIO) -> np.ndarray:
    """Load and preprocess image from byte stream."""
    byte_image.seek(0)
    file_bytes = np.asarray(bytearray(byte_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

def predict_image(image: np.ndarray, model_name: str) -> str:
    """Predict disease based on selected model."""
    processed_image = np.expand_dims(image, axis=0)
    
    if model_name == "pneumonia":
        prediction = pneumonia_model.predict(processed_image)[0]
        prob_class_1 = float(prediction[1]) if len(prediction) > 1 else float(prediction)
        prob_class_0 = 1.0 - prob_class_1
        predicted_class = 1 if prob_class_1 >= 0.5 else 0
        confidence = round(prob_class_1 * 100, 2) if predicted_class == 1 else round(prob_class_0 * 100, 2)
        label = PNEUMONIA_LABELS[predicted_class]
    elif model_name == "lung_cancer":
        prediction = lung_cancer_model.predict(processed_image)[0]
        predicted_class = np.argmax(prediction)
        confidence = round(float(prediction[predicted_class]) * 100, 2)
        label = LUNG_CANCER_LABELS[predicted_class]
    else:
        prediction = tuberculosis_model.predict(processed_image)[0]
        prob_class_1 = float(prediction[1]) if len(prediction) > 1 else float(prediction)
        prob_class_0 = 1.0 - prob_class_1
        predicted_class = 1 if prob_class_1 >= 0.5 else 0
        confidence = round(prob_class_1 * 100, 2) if predicted_class == 1 else round(prob_class_0 * 100, 2)
        label = TUBERCULOSIS_LABELS[predicted_class]
    
    return f"🩺 Prediction: {label}\nConfidence: {confidence}%\n⚠️ This is NOT a medical diagnosis. Consult a doctor."

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "👋 Welcome! Please select a model before sending an image:\n\n"
        "🔹 /select_pneumonia - Detect Pneumonia\n"
        "🔹 /select_lung_cancer - Detect Lung Cancer\n"
        "🔹 /select_tuberculosis - Detect Tuberculosis\n"
        "🔹 /help - Get information about bot functions"
    )

async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "This bot analyzes lung X-ray images to detect diseases. Use the following commands:\n\n"
        "/select_pneumonia - Activate pneumonia detection\n"
        "/select_lung_cancer - Activate lung cancer detection\n"
        "/select_tuberculosis - Activate tuberculosis detection\n"
        "Send an X-ray image after selecting a model to get a prediction."
        "🩻🩻🩻"
    )

async def select_model(update: Update, context: CallbackContext, model_name: str) -> None:
    global selected_model
    selected_model = model_name
    await update.message.reply_text(f"✅ {model_name.replace('_', ' ').title()} detection model activated. Send an X-ray image now.")

async def handle_image(update: Update, context: CallbackContext) -> None:
    global selected_model
    if selected_model is None:
        await update.message.reply_text("⚠️ Please select a model first: /select_pneumonia, /select_lung_cancer, or /select_tuberculosis")
        return
    
    photo = update.message.photo[-1]
    retries = 3
    while retries > 0:
        try:
            photo_file = await context.bot.get_file(photo.file_id)
            byte_image = BytesIO()
            await photo_file.download_to_memory(out=byte_image)
            break
        except NetworkError:
            retries -= 1
            time.sleep(2)
    
    image = load_image_from_bytes(byte_image)
    response = predict_image(image, selected_model)
    await update.message.reply_text(response)

async def main():
    TOKEN = "API-TOKEN"
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("select_pneumonia", lambda u, c: select_model(u, c, "pneumonia")))
    application.add_handler(CommandHandler("select_lung_cancer", lambda u, c: select_model(u, c, "lung_cancer")))
    application.add_handler(CommandHandler("select_tuberculosis", lambda u, c: select_model(u, c, "tuberculosis")))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())


