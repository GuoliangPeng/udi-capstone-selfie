from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np
import os
import rospy

class TLClassifier(object):
    def __init__(self):
        
        
        self.dg = tf.Graph()

        pwd = os.path.dirname(os.path.realpath(__file__))
        with self.dg.as_default():
            gdef = tf.GraphDef()
            with open(pwd+"/models/frozen_inference_graph.pb", 'rb') as f:
                gdef.ParseFromString(f.read())
                tf.import_graph_def(gdef, name="")

            self.session = tf.Session(graph=self.dg)
            # The name of the tensor is in the form tensor_name:tensor_index
            self.image_tensor = self.dg.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.dg.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.dg.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.dg.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.dg.get_tensor_by_name('num_detections:0')  

    def get_box(self, image):
        with self.dg.as_default():
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            tf_image_input = np.expand_dims(image, axis=0)

            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: tf_image_input})

            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)

            ret = []
            ret_scores = []
            detection_threshold = 0.3

            # Traffic signals are labelled 10 in COCO
            for idx, cl in enumerate(detection_classes.tolist()):
                if cl == 10:
                    if detection_scores[idx] < detection_threshold:
                        continue
                    dim = image.shape[0:2]
                    box = detection_boxes[idx]
                    box = [int(box[0] * dim[0]), int(box[1] * dim[1]), int(box[2] * dim[0]), int(box[3] * dim[1])]
                    box_h, box_w = (box[2] - box[0], box[3] - box[1])
                    #if box_h / box_w < 1.6:
                    #    continue
                    #print('detected bounding box: {} conf: {}'.format(box, detection_scores[idx]))
                    ret.append(box)
                    ret_scores.append(detection_scores[idx])
        return ret[np.argmax(ret_scores)] if ret else ret

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        box = self.get_box(image)
        #rospy.logerr("got image")
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image    
        if not box:
            #rospy.logerr("Couldn't locate lights")
            return TrafficLight.UNKNOWN
        i = 0
        class_image = img[box[0]:box[2], box[1]:box[3]]
            # The green needs to be checked first since red appears in many other components
            # For example traffice signs / other colors
        ret, thresh = cv2.threshold(class_image[:, :, 1], 150, 255, cv2.THRESH_BINARY)
        green_count = cv2.countNonZero(thresh)
        print('green_count', green_count)
        box_h, box_w = (box[2] - box[0], box[3] - box[1])
        img_size = box_w * box_h
        print('img_size', box_w * box_h)

        ret, thresh = cv2.threshold(class_image[:, :, 0], 150, 255, cv2.THRESH_BINARY)
        red_count = cv2.countNonZero(thresh)
        print('red_count', red_count)
        if green_count < 0.1 * img_size and red_count < 0.1 * img_size:
            rospy.logerr('YELLOW')
            return TrafficLight.YELLOW
        else:
            if red_count > green_count:
                rospy.logerr('RED')
                return TrafficLight.RED
            else:
                rospy.logerr('GREEN')
                return TrafficLight.GREEN
                
            
   
 




