#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy import spatial
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

#LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5

class WaypointUpdater(object):
    def __init__(self):
        #rospy.init_node('waypoint_updater')
        rospy.init_node('waypoint_updater',log_level=rospy.DEBUG)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.waypoints_2d   = None
        self.waypoint_tree  = None
        self.pose           = None
        self.road_size      = -1
        

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # no need to add /obstacle_waypoint subscriber since there will be no traffic
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()


    def loop(self):

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        # co-ordinates of the car
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # query for 1 waypoint that is closest to the car
        closest_idx = self.waypoint_tree.query([x,y],1)[1]

        # check if the closest point in ahead or behind the vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # We have the closest and the previous to closest co-ordinates now
        # we will construct vectors out of these and decide whether the waypoint is
        # ahead or behind depending on the dot product being positive or negative
        cl_vec = np.array(closest_coord)
        prev_vec = np.array(prev_coord)
        pos_vec = np.array([x,y])

        val = np.dot(cl_vec - prev_vec, pos_vec - cl_vec)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx


    def publish_waypoints(self):
        lane = Lane()
        lane.header = self.base_waypoints.header

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        
        if farthest_idx < self.road_size:
            main_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        else:
            rospy.logfatal("WP lookahead passed end of track!!!")
            offset = farthest_idx - self.road_size
            farthest_idx = self.track_waypoinroad_size_count - 2
            main_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
            main_waypoints = main_waypoints + self.base_waypoints.waypoints[0:offset]
            
            
        lane.waypoints = main_waypoints
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        
        rospy.loginfo("pose_cb: enter")
        self.pose = msg
        rospy.loginfo("pose_cb: done")

    def waypoints_cb(self, waypoints):
        
        size = len(waypoints.waypoints)
        self.road_size = len(waypoints.waypoints)
        
        self.base_waypoints = waypoints
        
        #use scipi KDTree to get closes waypoint
        if not self.waypoints_2d:
             self.waypoints_2d = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
             self.waypoint_tree = spatial.KDTree(self.waypoints_2d)   
        else:
            rospy.logerr("self.waypoints_2d already assigned?: %s",self.waypoints_2d)


    def traffic_cb(self, msg):
        rospy.logerr("---------------------------traffic_cb got called")
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        
        rospy.logerr("---------------------------obstacle_cb got called")
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')