import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import time
from PIL import Image




# Title of the web app
st.markdown('<h1 class="title">Face Recognition App</h1>', unsafe_allow_html=True)

# Section 1: What is Face Recognition?
col1, col2 = st.columns(2)
with col1:
    st.markdown('<h2 class="header"> What is Face Recognition? </h2> ', unsafe_allow_html=True)
    st.write("Facial Recognition is a way of recognizing a human face using biometrics. "
            "It consists of comparing features of a person’s face with a database of known faces to find a match. "
            "When the match is found correctly, the system is said to have ‘recognized’ the face. "
            "Face Recognition is used for a variety of purposes, like unlocking phone screens, identifying criminals, "
            "and authorizing visitors.")
with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
    image1 = Image.open('media/Picture1.png')
    st.image(image1, caption='')

# Section 2: How do Computers Recognize Faces?
st.markdown('<h2 class="header"> How do Computers Recognize Faces?</h2> ', unsafe_allow_html=True)
st.write("The Face Recognition system uses Machine Learning to analyze and process facial features from images or videos. "
         "Features can include anything, from the distance between your eyes to the size of your nose. "
         "These features, which are unique to each person, are also known as Facial Landmarks. "
         "The machine learns patterns in these landmarks by training Artificial Neural Networks. "
         "The machine can then identify people faces by matching these learned patterns against new facial data.")

# Section 3: Teach the Computer to Recognize your Face
st.markdown('<h2 class="header"> Teach the Computer to Recognize your Face </h2> ', unsafe_allow_html=True)
image2 = Image.open('media/Screenshot 2023-11-09 151026.png')
st.image(image2, caption='First Step')

st.markdown('<h3 class="sub-header"> Step 1 – Collect Data </h3> ', unsafe_allow_html=True)
st.write("We want our model to learn how to recognize your face. We will need two kinds of images for this - images of you, and images of people who are not you. This way, the model will learn to recognize how you look and also recognize how you don’t look.")

# Move the definition of me_files and not_me_files outside the button click condition
me_files = st.file_uploader("Upload 'me' Class Images", type=["jpg", "png"], accept_multiple_files=True, key="me")
not_me_files = st.file_uploader("Upload 'not me' Class Images", type=["jpg", "png"], accept_multiple_files=True, key="not_me")



# Function to capture image from webcam using OpenCV
def capture_image(label, save_folder):
    cam_port = 0  # Change this if you have multiple cameras
    cam = cv2.VideoCapture(cam_port)
    result, img = cam.read()
    cam.release()


    if result:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        image_name = f"{label}_captured_{time.time()}.png"
        image_path = os.path.join(save_folder, image_name)
        cv2.imwrite(image_path, img)
        st.success(f"Image captured and saved in {save_folder}")
    else:
        st.error("Failed to capture image. Please try again.")

# Webcam capture for 'me' class images
# Section to capture images
st.markdown('<h2 class="header"> Capture Images </h2> ', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Capture 'me' Image")
    if st.button("Capture 'me' Image"):
        capture_image('me', os.path.abspath('captured_images/me'))

with col2:
    st.subheader("Capture 'not me' Image")
    if st.button("Capture 'not me' Image"):
        capture_image('not_me', os.path.abspath('captured_images/not_me'))



# Slider for number of epochs
st.markdown('<h2 class="header"> Train the Machine </h2> ', unsafe_allow_html=True)
st.write("""Now let us set up our Machine Learning model!  Enter the number of epochs for which you would like the model to train:""")
epochs_duplicate = st.slider("Number of Epochs", 10, 100, 10)
epochs = (epochs_duplicate // 10)
st.write("""Once your model is all set, you can start training your model - 
""")


# Train the model
if st.button("Train Model"):
    # Gather paths for uploaded and captured images
    me_folder = os.path.abspath('captured_images/me')
    not_me_folder = os.path.abspath('captured_images/not_me')

    # Process uploaded images
    processed_images = []
    labels = []

    for uploaded_file in me_files or []:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        processed_images.append(img)
        labels.append(1)  # 'me' class

    for uploaded_file in not_me_files or []:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        processed_images.append(img)
        labels.append(0)  # 'not me' class

    # Process captured images if the folders exist
    if os.path.exists(me_folder) and os.path.exists(not_me_folder):
        for img_filename in os.listdir(me_folder):
            img_path = os.path.join(me_folder, img_filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            processed_images.append(img)
            labels.append(1)

        for img_filename in os.listdir(not_me_folder):
            img_path = os.path.join(not_me_folder, img_filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            processed_images.append(img)
            labels.append(0)

    if processed_images:  # Check if any images are available for training
        X_train = np.vstack(processed_images)
        y_train = np.array(labels)

        # Rest of your training code...
    else:
        st.warning("No images available for training. Please capture or upload images.")

    # Rest of your training code remains unchanged...
    # Train the model, save the model, etc.
    st.write(f"Training with {epochs_duplicate} epochs...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)  # Training with user-defined epochs

    # Calculate training accuracy
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    st.write(f"Training complete! Training Accuracy: {train_acc * 100:.2f}%")

    # Save the model
    model.save('model.h5')




# Section 5: Test the Model
st.markdown('<h2 class="header"> Test the model </h2> ', unsafe_allow_html=True)

# Option to upload a test image
st.markdown("<h3 class='sub-header'>Upload Test Image</h3>", unsafe_allow_html=True)
test_image = st.file_uploader("Upload a test image...", type=["jpg", "png"])

# Option to capture a test image
st.markdown("<h3 class='sub-header'>Capture Test Image</h3>", unsafe_allow_html=True)
if st.button("Capture Test Image"):
    capture_image('test_capture', 'captured_images/test_capture')

# Process the uploaded or captured test image
if test_image or os.path.exists('captured_images/test_capture'):
    st.write("Processing test image...")
    
    if os.path.exists('captured_images/test_capture'):
        # Use the captured test image
        test_image_path = os.path.join('captured_images/test_capture', os.listdir('captured_images/test_capture')[0])
    else:
        # Use the uploaded test image
        test_image_path = 'uploaded_test_image.png'
        with open(test_image_path, "wb") as f:
            f.write(test_image.read())

    img = image.load_img(test_image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make prediction
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
    model = Model(inputs=model.input, outputs=predictions)

    model.load_weights('model.h5')
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Display result
    if predicted_class == 1:
        st.write("Result: This is you!")
    else:
        st.write("Result: This is not you.")
else:
    st.warning("Please upload a test image or capture one.")