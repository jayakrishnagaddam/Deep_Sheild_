from flask import Flask, render_template, url_for, flash, request, redirect
from flask_pymongo import PyMongo
app = Flask(__name__)
app.config["SECRET_KEY"] = "1234"  # Add a secret key for flash messages
app.config["MONGO_URI"] = "mongodb://localhost:27017/DeepSheild"
mongo = PyMongo(app)


@app.route('/')
def index():
    return render_template('index.html')


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

        flash('SIGN UP SUCCESSFUL...YOU CAN NOW LOGIN HERE...', 'success')  # Flash success message
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/videodrop')
def videodrop():
    return render_template('videodrop.html')
 
@app.route('/download')
def download():
    return render_template('download.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
