#include <ros/ros.h>
#include <signal.h>  
#include <geometry_msgs/Twist.h>  
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Int32.h>
int flag;

 void cvcallback (const std_msgs::Int32& label)
 {int i = label.data;
 if (i == 1)
 {::flag = 1;}
 else
 {::flag = 0;} 
 }
 void scanCallback (const sensor_msgs::LaserScan::ConstPtr& scan_msg)
 {
  ros::NodeHandle node;
  ros::Publisher cmdVelPub = node.advertise<geometry_msgs::Twist>("/mobile_base/commands/velocity", 1);
  geometry_msgs::Twist speed;
 // ros::Rate loopRate(0.2);
//  while (ros::ok())  
//  {
    if (::flag ==1 && ((scan_msg->ranges[scan_msg->ranges.size()/2] < 1 &&  scan_msg->ranges[scan_msg->ranges.size()/2] > 0) || (scan_msg->ranges[1+(scan_msg->ranges.size()/2)] < 1 &&  scan_msg->ranges[1+(scan_msg->ranges.size()/2)] > 0) || (scan_msg->ranges[(scan_msg->ranges.size()/2)-1] < 1 &&  scan_msg->ranges[(scan_msg->ranges.size()/2)-1] > 0))){
    speed.linear.x = 0;
    speed.angular.z = 0;
    ros::Time begintime = ros::Time::now();
    ros::Duration secondswanttostop = ros::Duration(3);
    ros::Time endtime = begintime + secondswanttostop; 
    while(ros::Time::now() < endtime)
    {
    cmdVelPub.publish(speed);
    }    
    ros::Duration(2).sleep();
    }
//  }  
 }


/*void shutdown(int sig)  
{ ros::NodeHandle node;
  ros::Publisher cmdVelPub = node.advertise<geometry_msgs::Twist>("/mobile_base/commands/velocity", 1); 
  cmdVelPub.publish(geometry_msgs::Twist());
  ROS_INFO("turtlebot_control ended!");  
  ros::shutdown();  
}*/

int main(int argc, char** argv)  
{ 
  ros::init(argc, argv, "turtlebot_control");
  ros::NodeHandle node;
  ros::Subscriber cv_sub = node.subscribe("cv_moduel", 10,cvcallback);
  ros::Subscriber scan_sub = node.subscribe("scan", 1, scanCallback);
  ros::Publisher cmdVelPub = node.advertise<geometry_msgs::Twist>("/mobile_base/commands/velocity", 1);
  /*signal(SIGINT, shutdown);  
  ROS_INFO("turtlebot_control start..."); */ 
  ros::spin();
  return 0;  
}  
