import asyncio
import numpy as np
import cv2
from io import BytesIO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, CallbackQueryHandler, filters, CallbackContext
import tensorflow as tf
from tensorflow.keras.models import load_model
import nest_asyncio
import time
import joblib
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
PNEUMONIA_MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\LungXrayCT-Disease-Detection\models\pneumonia\lung_xray_scan_cases_sequential_neural_net_version2.keras"

LUNG_CANCER_MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\LungXrayCT-Disease-Detection\models\lung_tumor\lung_ct_scan_cases_sequential_neural_net.keras"

TUBERCULOSIS_MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\LungXrayCT-Disease-Detection\models\tuberculosis\tuberculosis_xray_scans_sequential_neural_net.keras"

OBESITY_MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\LungXrayCT-Disease-Detection\models\obesity\obesity_stacking.joblib"  # Add path to your obesity model

# Load models
pneumonia_model = load_model(PNEUMONIA_MODEL_PATH)
lung_cancer_model = load_model(LUNG_CANCER_MODEL_PATH, custom_objects={"WeightedSparseCategoricalCrossentropy": lambda: custom_loss})
tuberculosis_model = load_model(TUBERCULOSIS_MODEL_PATH)
obesity_model = joblib.load(OBESITY_MODEL_PATH)  # Load the obesity model

# Class labels
PNEUMONIA_LABELS = ["Normal", "Pneumonia"]
LUNG_CANCER_LABELS = ["Normal", "Benign", "Malignant"]
TUBERCULOSIS_LABELS = ["Normal", "Tuberculosis"]
OBESITY_LABELS = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Obesity Type I",
    3: "Obesity Type II",
    4: "Obesity Type III",
    5: "Overweight Level I",
    6: "Overweight Level II"
}

# Default model (None until the user selects)
selected_model = None

# Conversation states for obesity prediction
GENDER, AGE, HEIGHT, WEIGHT, FAMILY_HISTORY, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS = range(16)

# Dictionary to store user data for obesity prediction
user_data = {}

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

def predict_obesity(user_inputs: dict) -> str:
    """Predict obesity category and return a readable result."""
    processed_inputs = np.array(list(user_inputs.values())).reshape(1, -1)
    prediction = obesity_model.predict(processed_inputs)[0]  # Assuming model output is an encoded integer
    
    obesity_advice = {
        "Insufficient Weight": "⚠️ You may need to consume more balanced meals with proteins and healthy fats. Consider consulting a nutritionist.",
        "Normal Weight": "✅ Great job! Maintain your current healthy habits and stay active.",
        "Obesity Type I": "⚠️ It's important to manage weight through a combination of diet, exercise, and medical advice.",
        "Obesity Type II": "⚠️ Consider structured weight management plans with guidance from healthcare professionals.",
        "Obesity Type III": "🚨 High-risk category! Consult a doctor to develop a personalized plan to improve health.",
        "Overweight Level I": "⚠️ Try incorporating more physical activity and balanced meals into your routine.",
        "Overweight Level II": "⚠️ A structured diet and exercise plan can help you move towards a healthier weight."
    }
    
    result_label = OBESITY_LABELS[prediction]
    result_advice = obesity_advice[result_label]
    
    return f"🏥 Prediction: {result_label}\n{result_advice}\n⚠️ This is NOT a medical diagnosis. Please consult a healthcare professional."

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "👋 Welcome! Please select a model or assessment:\n\n"
        "🔹 /select_pneumonia - Detect Pneumonia\n"
        "🔹 /select_lung_cancer - Detect Lung Cancer\n"
        "🔹 /select_tuberculosis - Detect Tuberculosis\n"
        "🔹 /assess_obesity - Obesity Assessment\n"
        "🔹 /help - Get information about bot functions"
    )

async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "This bot can analyze lung X-ray, CT-Scans and health information. Use the following commands:\n\n"
        "/select_pneumonia - Activate pneumonia detection\n"
        "/select_lung_cancer - Activate lung cancer detection\n"
        "/select_tuberculosis - Activate tuberculosis detection\n"
        "/assess_obesity - Start obesity assessment questionnaire\n"
        "Send an X-ray image after selecting a model to get a prediction."
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

# Obesity assessment conversation handlers
async def assess_obesity(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    user_data[user_id] = {}
    
    keyboard = [
        [InlineKeyboardButton("Female", callback_data="0"),
         InlineKeyboardButton("Male", callback_data="1")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🔍 Starting Obesity Assessment\n\n"
        "Please select your gender:",
        reply_markup=reply_markup
    )
    return GENDER

async def gender(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['gender'] = int(query.data)
    
    await query.edit_message_text("Enter your age:")
    return AGE

async def age(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    try:
        user_data[user_id]['age'] = float(update.message.text)
        await update.message.reply_text("Enter your height (in meters, e.g., 1.75):")
        return HEIGHT
    except ValueError:
        await update.message.reply_text("Please enter a valid number for age.")
        return AGE

async def height(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    try:
        user_data[user_id]['height'] = float(update.message.text)
        await update.message.reply_text("Enter your weight (in kg):")
        return WEIGHT
    except ValueError:
        await update.message.reply_text("Please enter a valid number for height.")
        return HEIGHT

async def weight(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    try:
        user_data[user_id]['weight'] = float(update.message.text)
        
        keyboard = [
            [InlineKeyboardButton("No", callback_data="0"),
             InlineKeyboardButton("Yes", callback_data="1")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Has a family member suffered from overweight?",
            reply_markup=reply_markup
        )
        return FAMILY_HISTORY
    except ValueError:
        await update.message.reply_text("Please enter a valid number for weight.")
        return WEIGHT

async def family_history(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['family_history_with_overweight'] = int(query.data)
    
    keyboard = [
        [InlineKeyboardButton("No", callback_data="0"),
         InlineKeyboardButton("Yes", callback_data="1")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "Do you eat high caloric food frequently?",
        reply_markup=reply_markup
    )
    return FAVC

async def favc(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['favc'] = int(query.data)
    
    await query.edit_message_text("How many times do you eat vegetables per meal? (Enter a number from 1-3):")
    return FCVC

async def fcvc(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    try:
        user_data[user_id]['fcvc'] = float(update.message.text)
        await update.message.reply_text("How many main meals do you have daily? (Enter a number from 1-4):")
        return NCP
    except ValueError:
        await update.message.reply_text("Please enter a valid number.")
        return FCVC

async def ncp(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    try:
        user_data[user_id]['ncp'] = float(update.message.text)
        
        keyboard = [
            [InlineKeyboardButton("Always", callback_data="0"),
             InlineKeyboardButton("Frequently", callback_data="1")],
            [InlineKeyboardButton("Sometimes", callback_data="2"),
             InlineKeyboardButton("No", callback_data="3")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Do you eat any food between meals?",
            reply_markup=reply_markup
        )
        return CAEC
    except ValueError:
        await update.message.reply_text("Please enter a valid number.")
        return NCP

async def caec(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['caec'] = int(query.data)
    
    keyboard = [
        [InlineKeyboardButton("No", callback_data="0"),
         InlineKeyboardButton("Yes", callback_data="1")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "Do you smoke?",
        reply_markup=reply_markup
    )
    return SMOKE

async def smoke(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['smoke'] = int(query.data)
    
    await query.edit_message_text("How much water do you drink daily? (Liters, e.g., 2.5):")
    return
async def smoke(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['smoke'] = int(query.data)
    
    await query.edit_message_text("How much water do you drink daily? (Liters, e.g., 2.5):")
    return CH2O

async def ch2o(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    try:
        user_data[user_id]['ch2o'] = float(update.message.text)
        
        keyboard = [
            [InlineKeyboardButton("No", callback_data="0"),
             InlineKeyboardButton("Yes", callback_data="1")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Do you monitor the calories you eat daily?",
            reply_markup=reply_markup
        )
        return SCC
    except ValueError:
        await update.message.reply_text("Please enter a valid number.")
        return CH2O

async def scc(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['scc'] = int(query.data)
    
    await query.edit_message_text("How often do you have physical activity? (Hours per week, e.g., 3):")
    return FAF

async def faf(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    try:
        user_data[user_id]['faf'] = float(update.message.text)
        await update.message.reply_text("How much time do you use technological devices daily? (Hours, e.g., 4):")
        return TUE
    except ValueError:
        await update.message.reply_text("Please enter a valid number.")
        return FAF

async def tue(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    try:
        user_data[user_id]['tue'] = float(update.message.text)
        
        keyboard = [
            [InlineKeyboardButton("Always", callback_data="0"),
             InlineKeyboardButton("Frequently", callback_data="1")],
            [InlineKeyboardButton("Sometimes", callback_data="2"),
             InlineKeyboardButton("Never or Almost Never", callback_data="3")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "How often do you drink alcohol?",
            reply_markup=reply_markup
        )
        return CALC
    except ValueError:
        await update.message.reply_text("Please enter a valid number.")
        return TUE

async def calc(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['calc'] = int(query.data)
    
    keyboard = [
        [InlineKeyboardButton("Automobile", callback_data="0"),
         InlineKeyboardButton("Bike", callback_data="1")],
        [InlineKeyboardButton("Motorbike", callback_data="2"),
         InlineKeyboardButton("Public Transportation", callback_data="3")],
        [InlineKeyboardButton("Walking", callback_data="4")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "Which transportation do you usually use?",
        reply_markup=reply_markup
    )
    return MTRANS

async def mtrans(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_data[user_id]['mtrans'] = int(query.data)
    
    # Ensure all required data is collected
    input_data = {
        'gender': user_data[user_id]['gender'],
        'age': user_data[user_id]['age'],
        'height': user_data[user_id]['height'],
        'weight': user_data[user_id]['weight'],
        'family_history_with_overweight': user_data[user_id]['family_history_with_overweight'],
        'favc': user_data[user_id]['favc'],
        'fcvc': user_data[user_id]['fcvc'],
        'ncp': user_data[user_id]['ncp'],
        'caec': user_data[user_id]['caec'],
        'smoke': user_data[user_id]['smoke'],
        'ch2o': user_data[user_id]['ch2o'],
        'scc': user_data[user_id]['scc'],
        'faf': user_data[user_id]['faf'],
        'tue': user_data[user_id]['tue'],
        'calc': user_data[user_id]['calc'],
        'mtrans': user_data[user_id]['mtrans']
    }
    
    # Get prediction
    result = predict_obesity(input_data)
    
    await query.edit_message_text(f"Assessment complete!\n\n{result}")
    
    # Clear user data 
    del user_data[user_id]
    
    return ConversationHandler.END

async def cancel(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    if user_id in user_data:
        del user_data[user_id]
    
    await update.message.reply_text("Assessment cancelled❌. You can start again with /assess_obesity.")
    return ConversationHandler.END

async def main():
    TOKEN = "7578001596:AAGbZIr9Zyh9sYpe4UA_WPEALRsedzef0i8"
    application = Application.builder().token(TOKEN).build()
    
    # Regular command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("select_pneumonia", lambda u, c: select_model(u, c, "pneumonia")))
    application.add_handler(CommandHandler("select_lung_cancer", lambda u, c: select_model(u, c, "lung_cancer")))
    application.add_handler(CommandHandler("select_tuberculosis", lambda u, c: select_model(u, c, "tuberculosis")))
    
    # Conversation handler for obesity assessment
    obesity_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("assess_obesity", assess_obesity)],
        states={
            GENDER: [CallbackQueryHandler(gender)],
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, age)],
            HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, height)],
            WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, weight)],
            FAMILY_HISTORY: [CallbackQueryHandler(family_history)],
            FAVC: [CallbackQueryHandler(favc)],
            FCVC: [MessageHandler(filters.TEXT & ~filters.COMMAND, fcvc)],
            NCP: [MessageHandler(filters.TEXT & ~filters.COMMAND, ncp)],
            CAEC: [CallbackQueryHandler(caec)],
            SMOKE: [CallbackQueryHandler(smoke)],
            CH2O: [MessageHandler(filters.TEXT & ~filters.COMMAND, ch2o)],
            SCC: [CallbackQueryHandler(scc)],
            FAF: [MessageHandler(filters.TEXT & ~filters.COMMAND, faf)],
            TUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tue)],
            CALC: [CallbackQueryHandler(calc)],
            MTRANS: [CallbackQueryHandler(mtrans)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    
    application.add_handler(obesity_conv_handler)
    
    # Image handler for X-ray analysis
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    
    # Start the bot
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())

