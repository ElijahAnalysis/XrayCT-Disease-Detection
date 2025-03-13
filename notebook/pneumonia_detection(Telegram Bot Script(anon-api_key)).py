import asyncio
import numpy as np
import cv2
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from tensorflow.keras.models import load_model
import nest_asyncio
import time
from telegram.error import NetworkError

# Fix event loop issues in Jupyter
nest_asyncio.apply()

# Load pre-trained model
MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\Pneumonia-Detection\models\lung_scans_sequential_neural_net_version2.keras"
model = load_model(MODEL_PATH)

# Class labels
CLASS_LABELS = ["Normal", "Pneumonia"]

# Command bar
COMMANDS = (
    "✅ *Available Commands:*",
    "/start - Begin interaction with the bot",
    "/help - Get instructions and warnings",
    "/howpredictionsmade - Learn about the CNN model",
    "Send an X-ray image to get a prediction"
)

COMMANDS_TEXT = "\n".join(COMMANDS)

def load_image_from_bytes(byte_image: BytesIO) -> np.ndarray:
    """Loads and preprocesses an image from a byte stream."""
    byte_image.seek(0)
    file_bytes = np.asarray(bytearray(byte_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (224, 224))  # Resize to model input
    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Expands dimensions for batch processing before model inference."""
    return np.expand_dims(image, axis=0)  # Model expects batch input

def predict_image(image: np.ndarray) -> str:
    """Runs model prediction on the processed image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]  # Get prediction
    probability_pneumonia = float(prediction[0])  # Probability of Pneumonia
    probability_normal = 1 - probability_pneumonia  # Probability of Normal
    
    predicted_class = int(round(probability_pneumonia))  # Convert to 0 or 1
    confidence = round(probability_pneumonia * 100, 2) if predicted_class == 1 else round(probability_normal * 100, 2)
    
    description = (
        "🩺 *Prediction Result:*\n"
        f"🔹 Condition: *{CLASS_LABELS[predicted_class]}*\n"
        f"🔹 Confidence Level: *{confidence}%*\n\n"
        "📌 *Important:* This is an AI-based analysis. Consult a medical professional for accurate diagnosis."
    )
    return description

async def start(update: Update, context: CallbackContext) -> None:
    """Handles the /start command."""
    await update.message.reply_text(f"Welcome! {COMMANDS_TEXT}", parse_mode="Markdown")

async def help_command(update: Update, context: CallbackContext) -> None:
    """Handles the /help command, providing instructions and warnings."""
    help_text = (
        "📌 *How to use this bot:*\n"
        "1️⃣ Send a chest X-ray image.\n"
        "2️⃣ The bot will analyze it and provide a prediction.\n\n"
        "⚠️ *Warnings:*\n"
        "- This is *not* a medical diagnosis. Always consult a doctor.\n"
        "- Image quality affects accuracy. Use clear scans.\n"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def how_predictions_made(update: Update, context: CallbackContext) -> None:
    """Handles the /howpredictionsmade command, describing the CNN architecture."""
    model_info = (
        "🧠 *Model Architecture:*\n"
        "- Conv2D (32 filters, 3x3) + MaxPooling2D\n"
        "- BatchNormalization\n"
        "- Conv2D (64 filters, 3x3) + MaxPooling2D\n"
        "- BatchNormalization\n"
        "- Conv2D (128 filters, 3x3) + MaxPooling2D\n"
        "- BatchNormalization\n"
        "- Conv2D (256 filters, 3x3) + MaxPooling2D\n"
        "- BatchNormalization\n"
        "- GlobalAveragePooling2D\n"
        "- Dense (128 units) + Dropout\n"
        "- Dense (256 units)\n"
        "- Dense (1 unit, sigmoid activation)\n"
    )
    await update.message.reply_text(model_info, parse_mode="Markdown")

async def handle_image(update: Update, context: CallbackContext) -> None:
    """Handles image messages and runs prediction."""
    photo = update.message.photo[-1]  # Get highest resolution image
    
    retries = 3
    while retries > 0:
        try:
            photo_file = await context.bot.get_file(photo.file_id)
            byte_image = BytesIO()
            await photo_file.download_to_memory(out=byte_image)
            break  # Download successful, exit loop
        except NetworkError:
            retries -= 1
            if retries == 0:
                await update.message.reply_text("Failed to download image due to network issues. Please try again.")
                return
            time.sleep(2)  # Wait before retrying

    image = load_image_from_bytes(byte_image)
    response = predict_image(image)
    await update.message.reply_text(response, parse_mode="Markdown")

async def main():
    """Runs the Telegram bot."""
    TOKEN = "API-KEY"  # Replace with your actual bot token
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers for commands and images
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("howpredictionsmade", how_predictions_made))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Start bot
    await application.run_polling()
    print('RUNING >>>...')

if __name__ == "__main__":
    asyncio.run(main())
