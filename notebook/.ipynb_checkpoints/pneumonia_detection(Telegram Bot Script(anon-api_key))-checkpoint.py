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

# Load models
pneumonia_model = load_model(PNEUMONIA_MODEL_PATH)
lung_cancer_model = load_model(LUNG_CANCER_MODEL_PATH, custom_objects={"WeightedSparseCategoricalCrossentropy": lambda: custom_loss})

# Class labels
PNEUMONIA_LABELS = ["Normal", "Pneumonia"]
LUNG_CANCER_LABELS = ["Normal", "Benign", "Malignant"]

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
        # Fix for pneumonia model - handle both single value and array outputs
        if isinstance(prediction, np.ndarray) and len(prediction.shape) > 0 and prediction.shape[0] > 1:
            # If prediction is an array with multiple values
            prob_class_1 = float(prediction[1])
            prob_class_0 = float(prediction[0])
        else:
            # If prediction is a single value (e.g., sigmoid output)
            prob_class_1 = float(prediction)
            prob_class_0 = 1.0 - prob_class_1
            
        predicted_class = 1 if prob_class_1 >= 0.5 else 0
        confidence = round(prob_class_1 * 100, 2) if predicted_class == 1 else round(prob_class_0 * 100, 2)
        label = PNEUMONIA_LABELS[predicted_class]
    else:
        prediction = lung_cancer_model.predict(processed_image)[0]
        predicted_class = np.argmax(prediction)
        confidence = round(float(prediction[predicted_class]) * 100, 2)
        label = LUNG_CANCER_LABELS[predicted_class]

    return f"🩺 Prediction: {label}\nConfidence: {confidence}%\n⚠️ This is NOT a medical diagnosis. Consult a doctor."

async def start(update: Update, context: CallbackContext) -> None:
    """Send welcome message and prompt user to select a model."""
    await update.message.reply_text(
        "👋 Welcome! Please select a model before sending an X-ray image:\n\n"
        "🔹 /select_pneumonia - Detect Pneumonia\n"
        "🔹 /select_lung_cancer - Detect Lung Cancer\n"
        "🔹 /help - Get information about bot functions"
    )

async def help_command(update: Update, context: CallbackContext) -> None:
    """Send help message explaining bot functionality."""
    await update.message.reply_text(
        "ℹ️ This bot analyzes medical scans to predict potential conditions:\n\n"
        "✅ Pneumonia Detection (/select_pneumonia):\n  - Input: Chest X-ray\n  - Output: Normal or Pneumonia with confidence level\n\n"
        "✅ Lung Cancer Detection (/select_lung_cancer):\n  - Input: Lung CT scan\n  - Output: Normal, Benign, or Malignant with confidence level\n\n"
        "⚠️ Disclaimer: This bot does NOT provide medical advice. Always consult a doctor."
    )

async def select_pneumonia(update: Update, context: CallbackContext) -> None:
    """Set pneumonia detection model as active."""
    global selected_model
    selected_model = "pneumonia"
    await update.message.reply_text("✅ Pneumonia detection model activated. Send a Chest X-ray image now.")

async def select_lung_cancer(update: Update, context: CallbackContext) -> None:
    """Set lung cancer detection model as active."""
    global selected_model
    selected_model = "lung_cancer"
    await update.message.reply_text("✅ Lung cancer detection model activated. Send a Lung CT scan image now.")

async def handle_image(update: Update, context: CallbackContext) -> None:
    """Process received image and return prediction."""
    global selected_model
    if selected_model is None:
        await update.message.reply_text("⚠️ Please select a model first: /select_pneumonia or /select_lung_cancer")
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
            if retries == 0:
                await update.message.reply_text("Failed to download image. Please try again.")
                return
            time.sleep(2)
    
    try:
        image = load_image_from_bytes(byte_image)
        response = predict_image(image, selected_model)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error processing image: {str(e)}\nPlease try again with a different image.")

async def main():
    """Initialize and start the Telegram bot."""
    TOKEN = "API-KEY"
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("select_pneumonia", select_pneumonia))
    application.add_handler(CommandHandler("select_lung_cancer", select_lung_cancer))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    await application.run_polling()

# Check if running in an existing event loop (e.g., Jupyter)
if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(main())
    except RuntimeError:
        asyncio.run(main())