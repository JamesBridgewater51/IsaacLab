import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import pyrfuniverse.attributes as attr
try:
    import open3d as o3d
except ImportError:
    print('This feature requires open3d, please install with `pip install open3d`')
    raise
import pyrfuniverse.utils.rfuniverse_utility as utility
import pyrfuniverse.utils.depth_processor as dp
from pyrfuniverse.utils.coordinate_system_converter import CoordinateSystemConverter as csc
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv

nd_main_intrinsic_matrix = np.array([[600, 0, 240],
                                     [0, 600, 240],
                                     [0, 0, 1]])

limit = np.array([[-0.489, 0.14],
                  [-0.698, 0.489],
                  [-0.349, 0.349],
                  [0, 1.571],
                  [0, 1.571],
                  [0, 1.571],
                  [-0.349, 0.349],
                  [0, 1.571],
                  [0, 1.571],
                  [0, 1.571],
                  [-0.349, 0.349],
                  [0, 1.571],
                  [0, 1.571],
                  [0, 1.571],
                  [0, 0.785],
                  [-0.349, 0.349],
                  [0, 1.571],
                  [0, 1.571],
                  [0, 1.571],
                  [-1.047, 1.047],
                  [0, 1.222],
                  [-0.209, 0.209],
                  [-0.524, 0.524],
                  [0, 1.571]])
limit = limit * 180/np.pi
print(limit)

env = RFUniverseBaseEnv(check_version=False)
env.SetTimeStep(0.02)
env.SetGroundActive(False)
shadow = env.InstanceObject(name='shadowhand', attr_type=attr.ControllerAttr)
shadow.SetPosition([0 - 0.01, 0.5-0.02, 0-0.05])
# shadow.SetPosition([0, 0.52, 0])
shadow.SetRotation([0, -90, 90])


#你可以取消注释下面的6行代码，启动后移动镜头到你想要的位置，然后窗口上面的End Pend，就可以得到当前视口的位置
# env.Pend()
# env.GetViewTransform()
# env.step()
# print(env.data["view_position"])
# print(env.data["view_rotation"])
# env.Pend()

#设置想要的视口位置
env.SetViewTransform([-0.4683650732040405, 0.8655611872673035, -0.6687851548194885], [30.669431686401367, 48.74113845825195, -9.926162647388992e-07])

cube = env.LoadMesh(path=r'D:/Downloads/zyb/zyb/cube.fbx')
cube.SetKinematic(True)
cube.SetScale([0.05, 0.05, 0.05])
target_cube = cube.Copy(123456)
target_cube.SetPosition([0.2, 0.6, -0.5])
cameras = []
for i in range(14):
    camera = env.InstanceObject(name='Camera', attr_type=attr.CameraAttr)
    if i == 0:
        camera.SetTransform([0, 0.5, -1])
    elif i == 1:
        camera.SetTransform([0, 0.5, 1])
    elif i == 2:
        camera.SetTransform([-1, 0.5, 0])
    elif i == 3:
        camera.SetTransform([1, 0.5, 0])
    elif i == 4:
        camera.SetTransform([0, -0.5, 0])
    elif i == 5:
        camera.SetTransform([0, 1.5, 0])
    elif i == 6:
        camera.SetTransform([-0.5, 1, -0.5])
    elif i == 7:
        camera.SetTransform([0.5, 1, 0.5])
    elif i == 8:
        camera.SetTransform([0.5, 1, -0.5])
    elif i == 9:
        camera.SetTransform([-0.5, 1, 0.5])
    elif i == 10:
        camera.SetTransform([-0.5, 0, -0.5])
    elif i == 11:
        camera.SetTransform([0.5, 0, 0.5])
    elif i == 12:
        camera.SetTransform([0.5, 0, -0.5])
    elif i == 13:
        camera.SetTransform([-0.5, 0, 0.5])
    camera.LookAt([0, 0.5, 0])
    cameras.append(camera)

main_path = r"D:\Downloads\zyb\zyb\record"
actions = os.listdir(os.path.join(main_path, "action"))
csc = csc(["right", "up", "forward"], ["right", "forward", "up"])
for frame in actions:
    action = np.load(os.path.join(main_path, "action", frame))
    obs = np.load(os.path.join(main_path, "obs", frame))
    for i in range(obs.shape[0]):
        action1 = action[i]
        obs1 = obs[i]
        position = obs1[72:75].tolist()
        rotation = obs1[75:79].tolist()
        joint_positions = obs1[0:24].tolist()
        joint_positions[2] = -joint_positions[2]
        joint_positions[6] = -joint_positions[6]
        joint_positions[22] = -joint_positions[22]
        joint_positions[23] = -joint_positions[23]

        for i in range(24):
            joint_positions[i] = ((joint_positions[i] + 1) / 2) * (limit[i, 1] - limit[i, 0]) + limit[i, 0]
        joins = joint_positions.copy()
        joins[2:7] = joint_positions[19:24]
        joins[7:12] = joint_positions[14:19]
        joins[12:16] = joint_positions[10:14]
        joins[16:20] = joint_positions[6:10]
        joins[20:24] = joint_positions[2:6]



        cube.SetPosition(csc.cs2_pos_to_cs1_pos(position))
        cube.SetRotationQuaternion(csc.cs2_quat_to_cs1_quat(rotation))
        shadow.SetJointPositionDirectly(joins)
        env.step()

        point_cloud = []
        # for cam in cameras:
        #     cam.GetRGB(intrinsic_matrix=nd_main_intrinsic_matrix)
        #     env.step(simulate=False)
        #     image_byte = cam.data["rgb"]
        #     image_rgb = np.frombuffer(image_byte, dtype=np.uint8)
        #     image_rgb = cv2.imdecode(image_rgb, cv2.IMREAD_COLOR)
        #     image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        #
        #     cam.GetID(intrinsic_matrix=nd_main_intrinsic_matrix)
        #     env.step(simulate=False)
        #     image_id = cam.data["id_map"]
        #     image_id = np.frombuffer(image_id, dtype=np.uint8)
        #     image_id = cv2.imdecode(image_id, cv2.IMREAD_COLOR)
        #     image_id = cv2.cvtColor(image_id, cv2.COLOR_BGR2RGB)
        #
        #     cam.GetDepthEXR(intrinsic_matrix=nd_main_intrinsic_matrix)
        #     env.step(simulate=False)
        #     image_depth_exr = cam.data["depth_exr"]
        #
        #     local_to_world_matrix = cam.data["local_to_world_matrix"]
        #
        #     point_cloud1 = dp.image_bytes_to_point_cloud_intrinsic_matrix(
        #         image_byte, image_depth_exr, nd_main_intrinsic_matrix, local_to_world_matrix
        #     )
        #     color = utility.EncodeIDAsColor(shadow.id)[0:3]
        #     mask_point_cloud_shadow = dp.mask_point_cloud_with_id_color(point_cloud1, image_id, color)
        #
        #     color = utility.EncodeIDAsColor(cube.id)[0:3]
        #     mask_point_cloud_cube = dp.mask_point_cloud_with_id_color(point_cloud1, image_id, color)
        #
        #     mask_point_cloud_shadow.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        #     mask_point_cloud_cube.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        #     point_cloud.append(mask_point_cloud_shadow)
        #     point_cloud.append(mask_point_cloud_cube)

        env.step(2, simulate=False, collect=False)
        # o3d.visualization.draw_geometries(point_cloud)

env.Pend()
