git pull
docker build --build_arg USER_ID=$UID --build_arg USER_ID=$(id -g) -t spine-detr . && ./launch_docker.sh