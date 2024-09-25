from flask import Flask, render_template, url_for, flash, request, redirect, session, abort
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import google.auth.transport.requests
from flask_pymongo import PyMongo
import os
import pathlib
import requests
from pip._vendor import cachecontrol
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import zipfile
import base64
from io import BytesIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "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"
app.config["MONGO_URI"] = "mongodb+srv://2100090162:manigaddam@deepsheild.kzgpo9p.mongodb.net/deepsheild"
mongo = PyMongo(app)

GOOGLE_CLIENT_ID = "327025315574-urqtgpl60f11h2gv436pbajok3q0vtg9.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "API.json")

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'


flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    redirect_uri="http://localhost:5000/callback"
)


with zipfile.ZipFile("examples.zip", "r") as zip_ref:
    zip_ref.extractall(".")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()


def predict(input_image_path):
    
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    face = mtcnn(image)
    if face is None:
        return "No face detected", None, None, None

    face_bboxes, _ = mtcnn.detect(image)

    heatmap = np.zeros_like(image)

    for bbox in face_bboxes:
        x, y, w, h = bbox.astype(int)
        heatmap[y:y + h, x:x + w] = 1

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlayed_image = cv2.addWeighted(np.array(image), 0.7, np.array(heatmap), 0.3, 0, dtype=cv2.CV_8U)

    overlayed_image_pil = Image.fromarray(overlayed_image)

 
    face = F.interpolate(torch.tensor(face).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
    face = face.to(DEVICE, dtype=torch.float32) / 255.0

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "Real" if output.item() < 0.5 else "Fake"

    prediction_base64 = image_to_base64(overlayed_image_pil)

    original_img_base64 = image_to_base64(image)
    metadata = {'date': '2024-03-21', 'time': '12:00', 'location': 'Unknown'}
    probability_score = output.item()

    return prediction, prediction_base64, original_img_base64, metadata, probability_score


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' in request.files:
        image = request.files['image']
        image_path = "temp_image.jpg"
        image.save(image_path)
        prediction, heatmap_base64, original_img_base64, metadata, probability_score = predict(image_path)
        return render_template('prediction.html', prediction=prediction, heatmap_img=heatmap_base64, original_img=original_img_base64, metadata=metadata, probability_score=probability_score)
    else:
        return "No image found in request!"


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logout')
def logout():
    session.pop('email', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))


@app.route('/home')
def home():
    if 'email' in session:
        name = session.get('name')
        return render_template('home.html', name=name)
    else:
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if 'google_login' in request.form:
        authorization_url, state = flow.authorization_url()
        session["state"] = state
        return redirect(authorization_url)
    if request.method == 'POST' and 'manual_login' in request.form:
        email = request.form.get('email')
        password = request.form.get('password')

        user_data = mongo.db.users.find_one({'email': email, 'password': password})

        if user_data:
            firstname = user_data['first_name']
            session['email'] = email
            session['name'] = firstname
            return redirect(url_for('home'))
        else:
            error = 'Invalid username or password'

    return render_template('login.html', error=error)

@app.route('/callback')
def callback():
    flow.fetch_token(authorization_response=request.url)
    
   
    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )

    session["google_id"] = id_info.get("sub")
    session["name"] = id_info.get("name")
    session["email"] = id_info.get("email")

    return redirect(url_for('home'))

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

        flash('SIGN UP SUCCESSFUL...YOU CAN NOW LOGIN HERE...', 'success')
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


@app.route('/imagedrop')
def imagedrop():
    return render_template('imagedrop.html')


@app.route('/audiodrop')
def audiodrop():
    return render_template('audiodrop.html')


@app.route('/download')
def download():
    return render_template('download.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
