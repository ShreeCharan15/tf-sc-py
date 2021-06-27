import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
category_index = {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'bicycle'}, 3: {'id': 3, 'name': 'car'}, 4: {'id': 4, 'name': 'motorcycle'}, 5: {'id': 5, 'name': 'airplane'}, 6: {'id': 6, 'name': 'bus'}, 7: {'id': 7, 'name': 'train'}, 8: {'id': 8, 'name': 'truck'}, 9: {'id': 9, 'name': 'boat'}, 10: {'id': 10, 'name': 'traffic light'}, 11: {'id': 11, 'name': 'fire hydrant'}, 13: {'id': 13, 'name': 'stop sign'}, 14: {'id': 14, 'name': 'parking meter'}, 15: {'id': 15, 'name': 'bench'}, 16: {'id': 16, 'name': 'bird'}, 17: {'id': 17, 'name': 'cat'}, 18: {'id': 18, 'name': 'dog'}, 19: {'id': 19, 'name': 'horse'}, 20: {'id': 20, 'name': 'sheep'}, 21: {'id': 21, 'name': 'cow'}, 22: {'id': 22, 'name': 'elephant'}, 23: {'id': 23, 'name': 'bear'}, 24: {'id': 24, 'name': 'zebra'}, 25: {'id': 25, 'name': 'giraffe'}, 27: {'id': 27, 'name': 'backpack'}, 28: {'id': 28, 'name': 'umbrella'}, 31: {'id': 31, 'name': 'handbag'}, 32: {'id': 32, 'name': 'tie'}, 33: {'id': 33, 'name': 'suitcase'}, 34: {'id': 34, 'name': 'frisbee'}, 35: {'id': 35, 'name': 'skis'}, 36: {'id': 36, 'name': 'snowboard'}, 37: {'id': 37, 'name': 'sports ball'}, 38: {'id': 38, 'name': 'kite'}, 39: {'id': 39, 'name': 'baseball bat'}, 40: {'id': 40, 'name': 'baseball glove'}, 41: {'id': 41, 'name': 'skateboard'}, 42: {'id': 42, 'name': 'surfboard'}, 43: {'id': 43, 'name': 'tennis racket'}, 44: {'id': 44, 'name': 'bottle'}, 46: {'id': 46, 'name': 'wine glass'}, 47: {'id': 47, 'name': 'cup'}, 48: {'id': 48, 'name': 'fork'}, 49: {'id': 49, 'name': 'knife'}, 50: {'id': 50, 'name': 'spoon'}, 51: {'id': 51, 'name': 'bowl'}, 52: {'id': 52, 'name': 'banana'}, 53: {'id': 53, 'name': 'apple'}, 54: {'id': 54, 'name': 'sandwich'}, 55: {'id': 55, 'name': 'orange'}, 56: {'id': 56, 'name': 'broccoli'}, 57: {'id': 57, 'name': 'carrot'}, 58: {'id': 58, 'name': 'hot dog'}, 59: {'id': 59, 'name': 'pizza'}, 60: {'id': 60, 'name': 'donut'}, 61: {'id': 61, 'name': 'cake'}, 62: {'id': 62, 'name': 'chair'}, 63: {'id': 63, 'name': 'couch'}, 64: {'id': 64, 'name': 'potted plant'}, 65: {'id': 65, 'name': 'bed'}, 67: {'id': 67, 'name': 'dining table'}, 70: {'id': 70, 'name': 'toilet'}, 72: {'id': 72, 'name': 'tv'}, 73: {'id': 73, 'name': 'laptop'}, 74: {'id': 74, 'name': 'mouse'}, 75: {'id': 75, 'name': 'remote'}, 76: {'id': 76, 'name': 'keyboard'}, 77: {'id': 77, 'name': 'cell phone'}, 78: {'id': 78, 'name': 'microwave'}, 79: {'id': 79, 'name': 'oven'}, 80: {'id': 80, 'name': 'toaster'}, 81: {'id': 81, 'name': 'sink'}, 82: {'id': 82, 'name': 'refrigerator'}, 84: {'id': 84, 'name': 'book'}, 85: {'id': 85, 'name': 'clock'}, 86: {'id': 86, 'name': 'vase'}, 87: {'id': 87, 'name': 'scissors'}, 88: {'id': 88, 'name': 'teddy bear'}, 89: {'id': 89, 'name': 'hair drier'}, 90: {'id': 90, 'name': 'toothbrush'}}

class detect:
    def downloadImage(self,url):
        import requests
        response = requests.get(url)
        file = open("image.png", "wb")
        file.write(response.content)
        file.close()
        img = tf.io.read_file("image.png")
        img = tf.image.decode_jpeg(img, channels=3)
        converted_img  = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
        return converted_img
    
    def detectImage(self,image):
        print("loading model: ")
        #download model each time
        det=hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1")

        #download the model from https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1 and save it locally then comment the above line and
        #uncomment the below line to use locally (the above approach taken as github would not let me upload the model as it was 104mb)
        #det=tf.saved_model.load("")

        print("detecting: ")
        detector_output = det(image)
        return detector_output

    def mapToLabel(self,detector_output):
        class_ids = detector_output["detection_classes"].numpy()[0].tolist()
        labels=[category_index[a]['name'] for a in class_ids]
        scores=detector_output["detection_scores"].numpy()[0].tolist()
        vals={}
        for i in range(len(class_ids)):
            if scores[i]>=0.30:
                vals[labels[i]]=0
        for i in range(len(class_ids)):
            if scores[i]>=0.30:
                vals[labels[i]]+=1
        return vals

    
    def visualise(self,detector_output,image):
        from object_detection.utils import visualization_utils as viz_utils
        im=image.numpy().astype(np.uint8)
        COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16)]
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detector_output:
            keypoints = detector_output['detection_keypoints'][0]
            keypoint_scores = detector_output['detection_keypoint_scores'][0]
        viz_utils.visualize_boxes_and_labels_on_image_array(
            im[0],
            detector_output['detection_boxes'][0].numpy(),
            (detector_output['detection_classes'][0].numpy().astype(int) ),
            detector_output['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS
            )

        plt.figure(figsize=(24,32))
        plt.imshow(im[0])
        plt.savefig("labelledimage.png")
