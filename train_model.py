import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = (128, 128)
EPOCHS = 30
BATCH_SIZE = 32

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










1) Create a Maven project by executing the following command

mvn archetype:generate –DgroupId=in.eg -DartifactId=maven4 -
DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false

2) Go inside the newly created project by executing the following command

cd maven4

3) Package the newly created project by executing the following command
mvn package

You will get an error because of java version on executing the above command.
For that,

4) Open pom.xml and type the following code (after version tag (or) after
name tag)

<properties>
<maven.compiler.source>7</maven.compiler.source>
<maven.compiler.target>1.7</maven.compiler.target>
</properties>

5) Again package the project using the command

mvn package

6) See the list of options that can be used with git by typing
git

7) Add your email by executing the following command
git config --global user.email “Your Github email”

8) Add your name by executing the following command
git config --global user.name “Your Github username”

9) Initialize the Git repository by executing the following command
git init

10) Add pom.xml file by executing the following command
git add pom.xml

11) Add the source folder by executing the following command
git add src

12) Generate one SSH key by executing the following command

ssh-keygen -t ed25519 -C “Your Github email”

For enter file, don’t give anything just press Enter
For enter passphrase, don’t give anything just press Enter
For enter same passphrase again, don’t give anything just press Enter

13) Display the contents of the public key file id_ed25519.pub by executing the
following command
sudo cat /home/student/.ssh/id_ed25519.pub

Select the contents of this public key from the terminal, copy and paste it in the
3
rd input by logging in to your Github account and in the dashboard, click on
the profile pic icon in the top right corner --> Settings --> SSH and GPG keys -
-> New SSH Key.
Also, you have to enter a key name. For example, mykey or key1

14) Give a commit message
git commit -m “Your commit message”

Create a repository by logging in to your Github account and in the dashboard
left side, click on New -> enter a repository name, Give a description
(optional) and tick the checkbox “Add a Readme file” and click on the green
button “Create repository”

15) Set the origin to your Github account and repository
git remote add origin git@github.com:<Your Github username>/<Your
repository name>

16) Set push to Github for saving code to Github repository
git config --global push.autoSetupRemote true

17) Set the branch to main
git branch -M main

18) Push the code to Github repository
git push origin main

19) Rebase your repository to point to the main branch
git pull --rebase origin main

20) Again execute Step# 18

21) In case you executed Step #15 more than once, then execute the command

git remote set-url origin git@github.com:<Your Github username>/<Your
repository name> and execute the commands from Step# 16 to Step #20

22) Create an empty jar file, for example t.jar in /home/student path by
executing the command

gedit t.jar (and enter any character. Then, save and close the file)

22) Then, see the status of Jenkins by typing the following command

sudo systemctl status jenkins
(If you get a green symbol, it means Jenkins is active/running)
