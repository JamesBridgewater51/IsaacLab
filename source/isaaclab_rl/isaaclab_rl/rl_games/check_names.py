import json

json_path = "source/standalone/workflows/rl_games/PointData.json"
# Load the JSON file
with open(json_path) as json_file:
    data = json.load(json_file)

joint_set = set()

# turn the data to dict
data_dict = dict(data)

'''
在isaaclab里面,actuated names是不包括 RFJ0, FFJ0, 等等等等,由此可知 xxx_distal应该对应的是xxxJ0
对应地,middle表示中间指关节,proximal表示近端指关节,也就是靠近手掌的关节

所以palm对应的是robot0_palm, distal对应的是xxJ0, middle对应的是xxJ1, proximal对应的是xxJ2

那些有3的是因为有一个knuckle,有4的lf是因为有一个"掌骨", metacarpal

th 比较特殊, 从里到外的关节是 thumb_base, thumb_proximal, thumb, thumb_middle, thumb_distal, 其中base和mb都只是两个点,反正我们在这里只需要对应它的distal和 middle就行了

'''

json_names = ['robot0_ffdistal', 'robot0_ffmiddle', 'robot0_ffproximal', 'robot0_lfdistal', 'robot0_lfmiddle', 'robot0_lfproximal', 'robot0_mfdistal', 'robot0_mfmiddle', 'robot0_mfproximal', 'robot0_palm', 'robot0_rfdistal', 'robot0_rfmiddle', 'robot0_rfproximal', 'robot0_thdistal', 'robot0_thmiddle']

isaaclab_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_LFJ4', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_LFJ3', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ2', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ1', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']

    