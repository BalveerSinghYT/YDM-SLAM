#!/usr/bin/env bash

# Incremental build helper.
# Usage:
#   ./build.sh                 # default behavior (thirdparty + vocabulary + rgbd targets)
#   ./build.sh rgbd            # only build rgbd targets (incremental)
#   YDM_SKIP_THIRDPARTY=1 ./build.sh        # skip thirdparty (incremental)
#   YDM_SKIP_VOCAB=1 ./build.sh             # skip ORB vocabulary extraction

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_MODE="${1:-all}"
SKIP_THIRDPARTY="${YDM_SKIP_THIRDPARTY:-0}"
SKIP_VOCAB="${YDM_SKIP_VOCAB:-0}"
WITH_ROS="${YDM_WITH_ROS:-0}"

echo "Build mode: ${BUILD_MODE}"

build_thirdparty() {
	echo "Configuring and building Thirdparty/DBoW2 ..."

	cd "$ROOT_DIR/Thirdparty/DBoW2"
	mkdir -p build
	cd build
	cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build . -- -j

	echo "Configuring and building Thirdparty/g2o ..."

	cd "$ROOT_DIR/Thirdparty/g2o"
	mkdir -p build
	cd build
	cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build . -- -j

	echo "Configuring and building Thirdparty/Sophus ..."

	cd "$ROOT_DIR/Thirdparty/Sophus"
	mkdir -p build
	cd build
	cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build . -- -j
}

build_vocab() {
	if [[ "$SKIP_VOCAB" == "1" ]]; then
		echo "Skipping vocabulary extraction (YDM_SKIP_VOCAB=1)."
		return
	fi

	if [[ -f "$ROOT_DIR/Vocabulary/ORBvoc.txt" ]]; then
		echo "Vocabulary already extracted: Vocabulary/ORBvoc.txt"
		return
	fi

	echo "Uncompress vocabulary ..."
	cd "$ROOT_DIR/Vocabulary"
	tar -xf ORBvoc.txt.tar.gz
}

build_project() {
	echo "Configuring and building YDM-SLAM ..."
	mkdir -p "$ROOT_DIR/build"
	cd "$ROOT_DIR/build"
	ROS_FLAG=""
	if [[ "$WITH_ROS" == "1" ]]; then
		ROS_FLAG="-DYDM_WITH_ROS=ON"
		echo "ROS enabled: -DYDM_WITH_ROS=ON"
	fi
	cmake .. -DCMAKE_BUILD_TYPE=Release ${ROS_FLAG}
	cmake --build . --target rgbd_tum -- -j4
	cmake --build . --target flow_infer -- -j4
}

if [[ "$BUILD_MODE" == "rgbd" ]]; then
	# RGB-D only: keep it fast and incremental.
	SKIP_THIRDPARTY=1
fi

if [[ "$SKIP_THIRDPARTY" != "1" ]]; then
	build_thirdparty
else
	echo "Skipping thirdparty build (YDM_SKIP_THIRDPARTY=1)."
fi

build_vocab

build_project

exit 0

echo "Configuring and building Thirdparty/DBoW2 ..."


