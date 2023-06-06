#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Python Imports
import numpy as np
import tf.transformations as tr
# ROS Imports
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics

class BaxterDMP(object):
    """
    Wrapper class to interface baxter python functions (possibly GRASP specific) with visual servoing needs.
    """
    def __init__(self,limb):
        self._arm=baxter_interface.limb.Limb(limb)
        self._name=self._arm.joint_names()
        self._kin = baxter_kinematics(limb)
        self._joint_angle = dict()
        self._filename1 = '/home/robot707/ros_ws/src/visual_servoing_pbvs/src/dmpResult1.csv'
        self._filename2 = '/home/robot707/ros_ws/src/visual_servoing_pbvs/src/recording1.csv'
        self._main()
        
    def set_joint_angles(self,angles):
        joints=dict(zip(self._name,(angles)))
        #self._arm.set_joint_positions(joints)
        self._arm.move_to_joint_positions(joints,timeout=15.0)
        
    def _trajectory(self,name,euler=True):
        tra = np.loadtxt(name,delimiter=",", dtype=float)
        Nt = len(tra)
        if euler:
            tra_euler = tra[0:Nt,3:6]
            tra_quar = []
            for i in range(Nt):
                temp = tr.quaternion_from_euler(tra_euler[i][0],tra_euler[i][1],tra_euler[i][2])
                tra_quar.append(temp.tolist())
            tra = np.concatenate((tra[0:Nt,0:3],tra_quar),axis = 1)
            return tra, Nt
        else:
            return tra, Nt

    def _main(self):

        tra,Nt = self._trajectory(self._filename1,euler=True)
        tra = tra.tolist()
        
        for i in range(Nt):
            position = tra[i][0:3]
            orientation = tra[i][3:7]
            test = self._kin.inverse_kinematics(position,orientation)
            self.set_joint_angles(test)
        
        
def main():
    rospy.init_node('Baxter_DMP')
    rospy.loginfo("Start node: Baxter DMP")
    limb='right'
    BDMP = BaxterDMP(limb)
    
if __name__ == '__main__':
    main()

