#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <urdf_type>"
    exit 1
fi

URDF_TYPE=$1

python source/standalone/tools/convert_urdf.py \
    assets/shape_variant/urdf_heavy/${URDF_TYPE}/model.urdf \
    assets/shape_variant/usd_heavy_instanceable/${URDF_TYPE}/model_instanceable.usd \
    --make-instanceable