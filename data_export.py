import os
import tensorflow as tf
from proto.grip_attempt_pb2 import GripAttempt
import PIL.Image
from cStringIO import StringIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import csv
from sklearn.model_selection import train_test_split

def print_img(grip_attempt, num, cnt):
    img = grip_attempt.gripper_out_position.image
    a = cv2.imdecode(np.fromstring(img, np.uint8), -1)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    f = StringIO()
    PIL.Image.fromarray(np.uint8(a)).save(f, 'jpeg')
    if not os.path.exists('pics2'):
      os.makedirs('pics2')
    with open('pics2/pic_{:02d}_{:02d}.jpeg'.format(num, cnt), 'w') as fi:
        fi.write(f.getvalue())
    
cords_list = []
for num in range(18):
  cnt = 0
  iterator = tf.python_io.tf_record_iterator('data/file-{}.tf'.format(num))
  for data in iterator:
    grip_attempt = GripAttempt()
    grip_attempt.ParseFromString(data)
    print_img(grip_attempt, num, cnt)
    cords = (grip_attempt.gripper_grip_position.pose.x, grip_attempt.gripper_grip_position.pose.y)
    cords_list.append(('{:02d}_{:02d}'.format(num, cnt), cords))
    cnt += 1

train, test = train_test_split(cords_list, test_size=0.2)
with open('train_list.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(train)

with open('test_list.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(test)