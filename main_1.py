import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os

from tools import label_map_util
from tools import visualization_utils as vis_util

#导入预训练好的权重和标签
PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_LABELS = 'ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

## 将模型权重载入内存
#tf.Graph() 表示实例化了一个类，一个用于 tensorflow 计算和表示用的数据流图
#tf.Graph().as_default() 表示将这个类实例化，也就是新生成的图作为整个 tensorflow 运行环境的默认图
detection_graph = tf.Graph()
with detection_graph.as_default():
    #新建GraphDef文件，用于临时载入模型中的图
    od_graph_def = tf.GraphDef()
    #tf.gfile.GFile(path, mode)
    #类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        #GraphDef加载模型中的图
        od_graph_def.ParseFromString(fid.read())
        #在空白图中加载GraphDef中的图
        tf.import_graph_def(od_graph_def, name='')

## 载入标签图
#标签图将索引映射到类名称，当我们的卷积预测5时，我们知道它对应飞机。这里我们使用内置函数，但是任何返回将整数映射到恰当字符标签的字典都适用。
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

## 将图片转化为数组
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


## 测试图片路径
PATH_TO_TEST_IMAGES_DIR = 'test_images'
#os.path.join()函数用于路径拼接文件路径
#'test{}.jpg'.format(i)) for i in range(1, 3) 循环输出test1.jpg, test2.jpg....
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test{}.jpg'.format(i)) for i in range(1, 3) ]
IMAGE_SIZE = (12, 8) #输出图片的大小

##目标检测
with detection_graph.as_default():
    #图在会话中才能操作
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # 这个array在之后会被用来准备为图片加上框和标签
            image_np = load_image_into_numpy_array(image)
            # 扩展维度，应为模型期待: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # 每个框代表一个物体被侦测到.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # 每个分值代表侦测到物体的可信度.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # 执行侦测任务.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # 图形化.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()
