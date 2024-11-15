from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import cv2
import dlib
import numpy as np
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()

# Load models (age detection and face detection)
age_weights = "age_deploy.prototxt"
age_config = "age_net.caffemodel"
age_net = cv2.dnn.readNet(age_config, age_weights)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)
face_detector = dlib.get_frontal_face_detector()

@app.get("/")
async def main():
    content = """
    <html>
        <head>
            <title>Age Detection</title>
            <style>
                body {
                    margin: 0;
                    font-family: 'Arial', sans-serif;
                    background: url('https://yourbackgroundimageurl.com/image.jpg') no-repeat center center/cover;
                    color: white;
                    text-align: center;
                }
                .header {
                    background: rgba(0, 0, 0, 0.7);
                    padding: 20px;
                    font-size: 36px;
                    font-weight: bold;
                    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
                }
                .container {
                    max-width: 600px;
                    margin: 50px auto;
                    padding: 20px;
                    background: rgba(0, 0, 0, 0.6);
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                }
                input[type="file"] {
                    margin: 10px 0;
                    padding: 12px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                }
                input[type="submit"] {
                    margin: 10px 0;
                    padding: 12px 20px;
                    font-size: 16px;
                    border-radius: 5px;
                    border: none;
                    background-color: #28a745;
                    color: white;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #218838;
                }
                .result-message {
                    margin-top: 20px;
                    font-size: 18px;
                }
                img {
                    margin-top: 20px;
                    max-width: 100%;
                    border-radius: 5px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                }
            </style>
        </head>
        <body>
            <div class="header">Health is Our Priority</div>
            <div class="container">
                <h2>Upload Your Image for Age Detection</h2>
                <p>Detect your approximate age category from uploaded images using our advanced AI model.</p>
                <form id="upload-form" action="/upload/" enctype="multipart/form-data" method="post">
                    <input type="file" name="file" accept="image/*" required>
                    <input type="submit" value="Upload Image">
                </form>
                <div id="result"></div>
            </div>
            <script>
                document.getElementById('upload-form').onsubmit = async function(event) {
                    event.preventDefault();
                    const formData = new FormData(this);
                    const response = await fetch('/upload/', { method: 'POST', body: formData });
                    const content = await response.text();
                    document.getElementById('result').innerHTML = content;
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Image clarity check (blurriness detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return HTMLResponse(content="<p class='result-message'>The image is too blurry. Please upload a clearer image.</p>")

    # Face detection
    faces = face_detector(gray)
    if len(faces) == 0:
        return HTMLResponse(content="<p class='result-message'>No human face detected. Please upload a valid image.</p>")

    # Age estimation
    for face in faces:
        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y:y2, x:x2]
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), model_mean, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = ageList[age_preds[0].argmax()]

        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    _, encoded_img = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
    content = f"""
        <p class='result-message'>Age Prediction: {age}</p>
        <img src="data:image/png;base64,{img_base64}" alt="Processed Image">
    """
    return HTMLResponse(content=content)
