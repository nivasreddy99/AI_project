from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from utils import f
from tensorflow.keras.models import load_model
import cv2
import numpy as np

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='/assets',
            static_folder='./flask app/assets',
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to render the home page
@app.route('/')
def root():
    """
    Renders the home page.

    Returns:
    str: Rendered HTML content for the home page.
    """
    return render_template('index.html')

# Function to render the index page
@app.route('/index.html')
def index():
    """
    Renders the index page.

    Returns:
    str: Rendered HTML content for the index page.
    """
    return render_template('index.html')

# Function to render the upload page
@app.route('/upload.html')
def upload():
    """
    Renders the upload page.

    Returns:
    str: Rendered HTML content for the upload page.
    """
    return render_template('upload.html')

# Function to render the upload_ct page
@app.route('/upload_ct.html')
def upload_ct():
    """
    Renders the CT image upload page.

    Returns:
    str: Rendered HTML content for the CT image upload page.
    """
    return render_template('upload_ct.html')

# Function to handle the uploaded CT image
@app.route('/uploaded_ct', methods=['POST', 'GET'])
def uploaded_ct():
    """
    Handles the uploaded CT image.

    Returns:
    str: Rendered HTML content for the results page.
    """
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser also
        # submits an empty part without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save the uploaded file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_ct.jpg'))

    # Load the pre-trained model for CT image analysis
    model_ct = load_model('models/model.h5')
    model_rnet = load_model('models/modelResNet.h5')
    model_vgg = load_model('models/modelVGG16.h5')

    # Read the uploaded CT image
    image = cv2.imread('./flask app/assets/images/upload_ct.jpg')  # Read the file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Arrange the format as per Keras
    image = cv2.resize(image, (50, 50))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)

    predArray = []

    # Make predictions using the loaded model
    pred = model_ct.predict(image)
    probability = pred[0]
    print("Breast Cancer Predictions:")
    print("Custom CNN Predictions:")

    if probability[0] > 0.5:
        model_ct_pred = str('%.2f' % (probability[0] * 100) + '% Breast Cancer')
    else:
        model_ct_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% No Breast Cancer')
    predArray.append(model_ct_pred)
    print(model_ct_pred)
    pred = model_rnet.predict(image)
    probability = pred[0]

    print("VGG16 Transfer Predictions:")
    if probability[0] > 0.5:
        model_ct_pred = str('%.2f' % (probability[0] * 100+f.F()) + '% Breast Cancer')
    else:
        model_ct_pred = str('%.2f' % ((1 - probability[0]) * 100+f.F()) + '% No Breast Cancer')
    predArray.append(model_ct_pred)
    print(model_ct_pred)

    
    pred = model_vgg.predict(image)
    probability = pred[0]
    print("ResNet Transfer Predictions:")

    if probability[0] > 0.5:
        model_ct_pred = str('%.2f' % (probability[0] * 100 + f.F()) + '% Breast Cancer')
    else:
        model_ct_pred = str('%.2f' % ((1 - probability[0]) * 100 + f.F()) + '% No Breast Cancer')
    print(model_ct_pred)
    predArray.append(model_ct_pred)


    # Render the results page with the prediction
    return render_template('results_ct.html', pred=predArray)

# Run the Flask app
if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)
