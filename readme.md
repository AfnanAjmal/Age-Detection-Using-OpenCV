# Age and Gender Detection

This project utilizes deep learning models to detect the age and gender of people from images and videos. The app allows users to upload images, which are processed to predict the approximate age and gender of detected faces. The application uses FastAPI for the backend and OpenCV for image processing.

## Model Files

Before running the project, download the required pre-trained model files. These models are used for face detection and age prediction. Since these files are too large, you can download them from the links provided below:

1. **Face Detection Model**  
Download the Face Detection Model files:  
- [**opencv_face_detector.pbtxt**](https://your-link-here)  
- [**opencv_face_detector_uint8.pb**](https://your-link-here)  
These files are used for detecting faces in images and videos.

2. **Age Detection Model**  
Download the Age Detection Model files:  
- [**age_deploy.prototxt**](https://your-link-here)  
- [**age_net.caffemodel**](https://your-link-here)  
These files are used for predicting the age of the detected faces.

Once downloaded, place these model files in the root directory of your project.

## How to Run the Project

### 1. Clone the Repository  
Clone this repository to your local machine using the following command:  
`git clone https://github.com/yourusername/age-gender-detection.git && cd age-gender-detection`

### 2. Install Dependencies  
To install the required dependencies, run the following command:  
`pip install -r requirements.txt`

The `requirements.txt` file includes:  
- `opencv-python`  
- `dlib`  
- `numpy`  
- `fastapi`  
- `uvicorn`  
- `Pillow`

### 3. Run the Application  
Run the FastAPI application with this command:  
`uvicorn app:app --reload`