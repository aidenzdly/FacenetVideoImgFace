# Introduction
This project is to capture face from video, save face in the project directory, and upload a standard face, respectively, through the standard face and face capture face deployed on the server facenet model, return 128 dimensional face feature values, subtract and obtain Euclidean distance. Then judge the threshold value according to the test data, calculate the similarity according to the formula, judge the relationship between the intercepted face and the standard face.
# reg_img_face.py:
This file is created by canvas through MTCNN method, which will send a frame of picture in video to the back end, capture the face in the picture through MTCNN method, and save it to the project directory.
# detect_face.py and n_detect_face.py distinction：
The difference is that the two py-file det1.npy, det2.npy, det3.npy --> path problem.
