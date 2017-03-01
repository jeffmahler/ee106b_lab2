"""
Starter script for EE106B grasp planning lab
Author: Jeff Mahler
"""
import numpy as np

from core import RigidTransform
from meshpy import ObjFile

SPRAY_BOTTLE_MESH_FILENAME = 'data/spray.obj'

def contacts_to_baxter_hand_pose(contact1, contact2):
    c1 = np.array(contact1)
    c2 = np.array(contact2)

    # compute gripper center and axis
    center = 0.5 * (c1 + c2)
    y_axis = c2 - c1
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.array([y_axis[1], -y_axis[0], 0]) # the z axis will always be in the table plane for now
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)

    # convert to hand pose
    R_obj_gripper = np.array([x_axis, y_axis, z_axis]).T
    t_obj_gripper = center
    return RigidTransform(rotation=R_obj_gripper, 
                          translation=t_obj_gripper,
                          from_frame='gripper',
                          to_frame='obj')

if __name__ == '__main__':
    of = ObjFile(SPRAY_BOTTLE_MESH_FILENAME)
    mesh = of.read()

    vertices = mesh.vertices
    triangles = mesh.triangles
    normals = mesh.normals

    print 'Num vertices:', len(vertices)
    print 'Num triangles:', len(triangles)
    print 'Num normals:', len(normals)

    # 1. Generate candidate pairs of contact points

    # 2. Check for force closure

    # 3. Convert each grasp to a hand pose
    contact1 = vertices[0]
    contact2 = vertices[100]
    T_obj_gripper = contacts_to_baxter_hand_pose(contact1, contact2)
    print 'Translation', T_obj_gripper.translation
    print 'Rotation', T_obj_gripper.quaternion

    pose_msg = T_obj_gripper.pose_msg

    # 4. Execute on the actual robot
