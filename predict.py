import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (128, 128)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img

def predict_pollution_level(model, image_path):
    img = preprocess_image(image_path)
    if img is None:
        return "Error processing image"
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    labels = ['Low', 'Medium', 'High']
    return labels[predicted_class]

























---
- name: Deploy Artifact to Localhost
  hosts: localhost
  tasks:
    - name: Copy the artifact to the target location
      become: true
      become_user: student
      become_method: su
      copy:
        src: "/var/lib/jenkins/workspace/AnsiVinay/target/my-app-1.0-SNAPSHOT.jar"
        dest: "/home/student/t.jar"



-------------------------------------------------------------------------------------------

- name: First Playbook
  hosts: local
  connection: local
  tasks:
    - name: My first task
      debug:
        msg: "Ansible is a config mgmt tool"

-----------------------------------------------------------------------------------------
#!/usr/bin/python

from ansible.module_utils.basic import AnsibleModule

def walk():
    module_args = dict(
        name=dict(type='str', required=True)
    )

    module = AnsibleModule(argument_spec=module_args)
   
    result = {
        "changed": False,
        "message": f"Welcome, {module.params['name']}!"
    }

    module.exit_json(**result)

if __name__ == '__main__':
    walk()

-----------------------------------------------------------------------------------------
---
- name: Ansible with one module
  hosts: localhost
  gather_facts: no
  tasks:
    - name: Welcome friends
      firstmodule:
        name: "Ansible"
      register: result

    - name: Show message
      debug:
        msg: "{{ result.message }}"
