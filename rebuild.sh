git pull
docker build --build_arg USER_ID=$UID -t spine-detr . && ./launch_docker.sh