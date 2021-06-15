export CUDA_VISIBLE_DEVICES=$1

python3 main.py --dataset_file 2d_spine --2d_spine_ann fake_coco --num_queries 15 --2d_spine_images "/home/data/Diagnostikbilanz/Fertig 20190503/" --rand_rot 1 --batch_size 8 --comment "$2_$3_180" --cross_val "0" --output_dir logs --rand_crop $3 --resize 180  $4

python3 main.py --dataset_file 2d_spine --2d_spine_ann fake_coco --num_queries 15 --2d_spine_images "/home/data/Diagnostikbilanz/Fertig 20190503/" --rand_rot 1 --batch_size 8 --comment "$2_$3_224" --cross_val "0" --output_dir logs --rand_crop $3 --resize 224  $4

python3 main.py --dataset_file 2d_spine --2d_spine_ann fake_coco --num_queries 15 --2d_spine_images "/home/data/Diagnostikbilanz/Fertig 20190503/" --rand_rot 1 --batch_size 8 --comment "$2_$3_256" --cross_val "0" --output_dir logs --rand_crop $3 --resize 256  $4

python3 main.py --dataset_file 2d_spine --2d_spine_ann fake_coco --num_queries 15 --2d_spine_images "/home/data/Diagnostikbilanz/Fertig 20190503/" --rand_rot 1 --batch_size 8 --comment "$2_$3_300" --cross_val "0" --output_dir logs --rand_crop $3 --resize 300  $4

python3 main.py --dataset_file 2d_spine --2d_spine_ann fake_coco --num_queries 15 --2d_spine_images "/home/data/Diagnostikbilanz/Fertig 20190503/" --rand_rot 1 --batch_size 8 --comment "$2_$3_360" --cross_val "0" --output_dir logs --rand_crop $3 --resize 360  $4

python3 main.py --dataset_file 2d_spine --2d_spine_ann fake_coco --num_queries 15 --2d_spine_images "/home/data/Diagnostikbilanz/Fertig 20190503/" --rand_rot 1 --batch_size 8 --comment "$2_$3_480" --cross_val "0" --output_dir logs --rand_crop $3 --resize 480  $4

python3 main.py --dataset_file 2d_spine --2d_spine_ann fake_coco --num_queries 15 --2d_spine_images "/home/data/Diagnostikbilanz/Fertig 20190503/" --rand_rot 1 --batch_size 8 --comment "$2_$3_512" --cross_val "0" --output_dir logs --rand_crop $3 --resize 512  $4

python3 main.py --dataset_file 2d_spine --2d_spine_ann fake_coco --num_queries 15 --2d_spine_images "/home/data/Diagnostikbilanz/Fertig 20190503/" --rand_rot 1 --batch_size 8 --comment "480_300_val0" --cross_val "0" --output_dir logs --rand_crop 480 --resize 300