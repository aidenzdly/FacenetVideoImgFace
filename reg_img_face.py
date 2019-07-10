import tensorflow as tf
import cv2
import detect_face
from scipy import misc
import os


minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor:比例因子
gpu_memory_fraction = 1.0 # 拿出GPU容量比列

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
        pnet, rnet, onet = detect_face.create_mtcnn(sess,'detection_model')


image_face_dir = './static/regist_image/'
image_face_paths = [os.path.join(image_face_dir,i) for i in os.listdir(image_face_dir)]
image_index = 0
for image_face in image_face_paths:
    image_index += 1
    img = misc.imread(image_face)
# image_path = r'C:\Users\T470P\Desktop\test\faces.png'
# 读入图片

# 识别人脸，bounding_boxex为图像矩阵
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]  # 人脸数目
    print('第{}张图找到人脸数目为：{}'.format(image_index,nrof_faces))
    # print(bounding_boxes)

    # 在识别人脸上画框
    crop_faces = []
    crop_index = 1
    for face_position in bounding_boxes:
        # 变为int
        face_position = face_position.astype(int)
        # print(face_position[0:4])
        # 画框
        # cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        # 截取识别人脸
        crop = img[face_position[1]-10:face_position[3]+5,
               face_position[0]-4:face_position[2]+5, ]
        # 放大识别人脸图像:cv2.INTER_CUBIC(推荐)和cv2.INTER_LINEAR(默认)
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
        crop_rgb = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        # print(crop.shape)
        cv2.imwrite(r"static/regist_image_face/" + str(crop_index) + "_" + "face" + '.jpg', crop_rgb)
        # cv2.imwrite(r"static/uploads/"  + "1_img_face" + '.jpg', crop_rgb)
        crop_faces.append(crop_rgb)
        crop_index += 1