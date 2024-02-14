from flask import Flask, render_template, url_for, flash, request, redirect, jsonify
from flask_pymongo import PyMongo
from classify_image import train_model, predict_image
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "1234"  # Add a secret key for flash messages
app.config["MONGO_URI"] = "mongodb+srv://2100090162:manigaddam@deepsheild.kzgpo9p.mongodb.net/deepsheild"
mongo = PyMongo(app)

real_folder = "Real"
fake_folder = "Fake"

# Train the model
train_model(real_folder, fake_folder)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/classify_image', methods=['POST'])
def classify_image():
    # Check if the request has the file part
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        # Save the file to a temporary location
        image_path = "temp_image.jpg"
        file.save(image_path)
        # Perform image classification
        result = predict_image(image_path)
        # Render a template with the classification result
        return render_template('classification_result.html', result=result)


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user_data = mongo.db.users.find_one({'email': email, 'password': password})

        if user_data:
            flash('Login successful', 'success')  # Flash success message
            return redirect(url_for('home'))
        else:
            error = 'Invalid username or password'

    return render_template('login.html', error=error)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        user_data = {
            'first_name': request.form.get('firstname'),
            'last_name': request.form.get('lastname'),
            'email': request.form.get('email'),
            'password': request.form.get('password')
        }

        mongo.db.users.insert_one(user_data)

        flash('SIGN UP SUCCESSFUL...YOU CAN NOW LOGIN HERE...',
              'success')  # Flash success message
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/contactus')
def contactus():
    return render_template('contactus.html')


@app.route('/notify')
def notify():
    return render_template('notify.html')


@app.route('/videodrop')
def videodrop():
    return render_template('videodrop.html')


@app.route('/imagedrop', methods=['POST'])
def imagedrop():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image_file = request.files['image']

    # Perform image classification
    prediction = predict_image(image_file)

    # Return the classification result
    return jsonify({'prediction': prediction})


@app.route('/audiodrop')
def audiodrop():
    return render_template('audiodrop.html')


@app.route('/download')
def download():
    return render_template('download.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
