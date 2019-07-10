from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import n_detect_face
import cv2
import numpy as np
import os
from scipy.misc import imread
import json
import requests
import re

# 储存文件路径
UPLOAD_FOLDER = 'static/uploads'
# 允许上传文件扩展名集合
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
# detector = dlib.get_frontal_face_detector()
video = cv2.VideoCapture('./static/video/my.mp4')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.add_url_rule('/static/uploads/<filename>', 'uploaded_file',
                 build_only=True)

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor:比例因子
gpu_memory_fraction = 1.0  # 拿出GPU容量比例
print('Creating networks and loading parameters')

# 创建session,对session进行参数配置
with tf.Graph().as_default():
    # 指定了每个GPU进程中使用显存的上限，但它只能均匀地作用于所有GPU，无法对不同GPU设置不同的上限。
    # 1:每个GPU拿出全部容量给进程使用
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    # 默认是用GPU内存，不打印设备分配日志
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        # 创建mtcnn结构
        pnet, rnet, onet = n_detect_face.create_mtcnn(sess, None)


def get_euclidean_distance(emb_a, emb_b):
    """
    计算欧氏距离
    """
    distance = np.sqrt(np.sum(np.square(emb_a - emb_b), axis=-1))
    return distance


def get_input_data(image_path):
    """
    读取图片
    """
    image_data = cv2.imread(image_path)
    image_data = cv2.resize(image_data, (224, 224))
    # BGR to RGB
    image_data = image_data[:, :, ::-1]
    image_data = image_data.astype(np.float32)
    # 此时的image_data 类型为numpy 是无法直接包装到json请求信息中的，需要转换为数组
    image_data = image_data.tolist()  # list
    return image_data


def request_model(image_face_dir):
    """
    图片上传服务器，请求模型返回特征值
    """
    URL = "http://192.168.4.140:8501/v1/models/facenet1:predict"
    headers = {"content-type": "application/json"}
    image_face_paths = os.listdir(image_face_dir)
    # 将图片路径按顺序排列
    new_image_face_paths = sorted(image_face_paths, key=lambda i: int(re.match(r'(\d+)', i).group()))
    dirs = []
    for image_face in new_image_face_paths:
        complete_dirs = image_face_dir + image_face
        dirs.append(complete_dirs)
    instances = []
    for per_image_dir in dirs:
        # 保证一次上传一张图片至服务器
        del instances[:]
        per_image_instance = {"input_image": get_input_data(per_image_dir)}
        instances.append(per_image_instance)
        body = {
            "signature_name": "serving_default",
            "instances": instances
        }
        r = requests.post(URL, data=json.dumps(body), headers=headers, timeout=30)
        result = json.loads(r.text)
        for per_result in result['predictions']:
            face_eigenvalue = np.array(per_result)
    # 返回128维人脸特征值
    return face_eigenvalue


def allowed_file(filename):
    """
    规范文件拓展名
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST', 'GET'])
def compare():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        infname = request.form['fname']
        # 有文件且文件拓展名允许
        if file and allowed_file(infname):  # True
            # secure_filename()函数对文件名进行校验
            filename = secure_filename(infname)  # x.jpg
            arr = imread(file, mode='RGB')
            frame = arr
            draw_frame = frame.copy()
            # 识别人脸图片矩阵
            bounding_boxes, _ = n_detect_face.detect_face(draw_frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]  # 人脸数目
            print('找到人脸数目为：{}'.format(nrof_faces))
            crop_faces = []
            crop_index = 1
            for face_position in bounding_boxes:
                face_position = face_position.astype(int)  # int类型
                # 截取识别人脸，并相对扩大人脸图片
                crop = draw_frame[face_position[1] - 10:face_position[3] + 10,
                       face_position[0] - 5:face_position[2] + 5, ]
                # 未截取到人脸
                if len(crop) == 0:
                    return json.dumps(
                        {
                            'ok': False,
                            'msg': '未检测到人脸，请继续截取比对!'
                        }).encode('utf8')
                # 放大识别人脸图像:cv2.INTER_CUBIC(推荐)和cv2.INTER_LINEAR(默认)
                crop_image = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
                # BGR to RGB
                crop_image_rgb = cv2.cvtColor(crop_image, cv2.COLOR_RGBA2BGR)
                # 拼接保存文件路径
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), crop_image_rgb)
                crop_faces.append(crop)
                crop_index += 1
            # 标准别对人脸
            standard_face_dir = './static/regist_image_face/'
            face_eigenvalue_std = request_model(standard_face_dir)
            # while True:
            # 截取别对人脸
            image_face_dir = './static/uploads/'
            face_eigenvalue = request_model(image_face_dir)
            # 欧氏距离计算
            euclidean_distance = get_euclidean_distance(face_eigenvalue_std, face_eigenvalue)
            # euclidean_distance = '%.3f' % euclidean_distance  # str
            # 模型根据测试数据不同，阈值取值为0.9
            # 相似度计算
            euclidean_distance_min = 0.4  # 设置欧氏距离最小值
            euclidean_distance_max = 1.5  # 设置欧氏距离最大值
            similarity = (euclidean_distance_max - float(euclidean_distance)) \
                         / (euclidean_distance_max - euclidean_distance_min)
            similarity = '%.2f%%' % (similarity * 100)
            # 返回数据data
            return json.dumps(
                {
                    'ok': True,
                    'itemName': infname,
                    # 人脸图片路径拼接：反向解析得到储存文件路径
                    'itemUrl': 'http://localhost:5000' + url_for('uploaded_file',
                                                                 filename=filename),
                    'img_similarity': str(similarity)
                }).encode('utf8')
    return ''


if __name__ == '__main__':
    app.run()
