import json

json_path = "source/standalone/workflows/rl_games/PointData.json"

# joint_names = [
#     "robot0_palm",
# ]


# Load the JSON file
with open(json_path) as json_file:
    data = json.load(json_file)

# turn the data to dict
data_dict = dict(data)
res_dict = {
    
}
# print(data_dict)
for key, val in data_dict.items():
    joint_name = val["Item2"]
    if res_dict.get(joint_name) is not None:
        res_dict[joint_name].append(eval(val["Item1"]) + list(val["Item3"].values()))
    else:
        res_dict[joint_name] = [eval(val["Item1"]) + list(val["Item3"].values())]
    
    print(f"joint_name: {joint_name}")
    print(f"list_items: {list(val['Item1']) + list(val['Item3'].values())}")


print(res_dict)
# save res_dict as joint_index.json
with open("source/standalone/workflows/rl_games/joint_index.json", "w") as json_file:
    json.dump(res_dict, json_file)