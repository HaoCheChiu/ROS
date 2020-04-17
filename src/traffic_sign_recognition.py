import rospy
from vgg import *
import torch
from torchvision import transforms
from PIL import Image
from sensor_msgs.msg import Image
from std_msgs.msg import Int32


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

use_gpu = False
if torch.cuda.is_available():
    print("use_gpu")
    use_gpu = True

model = vgg11(num_classes=2)
model.load_state_dict(torch.load("vgg11.pth"))
if use_gpu:
    model = model.cuda()

def inference(img, model):
    #input img a W x H x 3 PIL image file
    #return label number, 1: has stop sign, 0: has no stop sign 
    model.eval()
    im = data_transforms(img)
    if use_gpu:
        im = im.cuda()    
    im = im.view(1,3,224,224)
    print(im.shape)
    score = model(im)
    label = torch.argmax(score, dim = 1)[0].item()
    return label

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "picture recieved: %s", data.data)

def main():
    rospy.init_node('cv_moduel')
    image_sub = rospy.Subscriber('image_converter', Image, callback)
    im = image_sub
    label = interface(im, model)
    label_pub = rospy.Publisher('turtlebot_control', Int32, queue_size = 2)
    while not rospy.is_shutdown():
        rospy.loginfo(label)
        label_pub.publish(label)
        rate.sleep()

if __name__ == '__main__'
    main()
    

    
        
        

