import open3d as o3d
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_rgbd


class PointcloudVisualizer() :
	def __init__(self) -> None:
		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window()
		# self.vis.register_key_callback(key, your_update_function)
	
	def add_geometry(self, cloud) :
		self.vis.add_geometry(cloud)

	def update(self, cloud):
		#Your update routine
		self.vis.update_geometry(cloud)
		self.vis.update_renderer()
		self.vis.poll_events()



def farthest_point_sampling(self, point_cloud, num_points):
        """
        使用Open3D库进行最远点采样

        参数：
            point_cloud (numpy.array): 表示点云的NumPy数组，每一行是一个点的坐标 [x, y, z]
            num_points (int): 采样点的数量

        返回：
            numpy.array: 采样后的点云数组，每一行是一个采样点的坐标 [x, y, z]
        """
        sampled_points = o3d.geometry.PointCloud.farthest_point_down_sample(point_cloud, num_points)


        return sampled_points
