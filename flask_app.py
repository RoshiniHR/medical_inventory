from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from train_model import create_model 
import tensorflow as tf
from predict import predict_pollution_level

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the model
model = None
try:
    model = tf.keras.models.load_model('waterpollution/project/pollution_level_model1.keras')
except:
    print("Model not found, please train the model first.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction (already returns text category)
            prediction_text = predict_pollution_level(model, filepath)
            return render_template('result.html', 
                                filename=filename, 
                                prediction=prediction_text)
    return render_template('upload.html')

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
def load_images_and_labels(img_dir, label_csv):
    df = pd.read_csv(label_csv)
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0
        images.append(img)
        labels.append(int(row['label']))

    return np.array(images), to_categorical(labels, num_classes=3)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(3, activation='softmax')  # 3 classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    images, labels = load_images_and_labels('images/', 'labels.csv')
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    model = create_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    model.save('pollution_level_model.keras')

if __name__ == '__main__':
    train()































sudo gedit /var/lib/jenkins/workspace/AnsiDeploy/hosts.ini

pipeline
{
    agent any
    stages
    {
        stage("Checkout")
        {
            steps
            {
                git branch: 'main', url : 'https://github.com/RoshiniHR/RoshDevop.git'
            }
        }
    
        stage("Build")
        {
            steps
            {
                sh 'mvn clean package'
            }       
        }
        stage("Test")
        {
            steps
            {
                sh 'mvn test'
            }
        }
        stage("Archive Artifacts")
        {
            steps
            {
                archiveArtifacts artifacts: '**/target/*.jar', allowEmptyArchive: true
            }   
        }
        stage("Deploy")
        {
            steps
            {
                sh """
                    export ANSIBLE_HOST_KEY_CHECKING=False
                    ansible-playbook -i hosts.ini mydeploy.yml --extra-vars='ansible_become_pass=exam@cse'
                    """
            }
        }
    }
}
