git pull
docker build --build-arg USER_ID=$UID --build-arg GROUP_ID=$(id -g) -t spine-detr . && ./launch_docker.sh