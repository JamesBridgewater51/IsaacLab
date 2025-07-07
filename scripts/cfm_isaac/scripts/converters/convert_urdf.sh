#!/bin/bash

# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <urdf_type>"
#     exit 1
# fi

# URDF_TYPE=$1


# python source/standalone/tools/convert_urdf.py \
#     assets/shape_variant/urdf_heavy/${URDF_TYPE}/model.urdf \
#     assets/shape_variant/usd_heavy/${URDF_TYPE}/model.usd \



python source/standalone/tools/convert_urdf.py \
    /home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/assets/Shadow_URDF/sr_hand.urdf \
    /home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/assets/Shadow_URDF/sr_hand.usd \
    --fix-base \