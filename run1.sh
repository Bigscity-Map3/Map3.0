
# for dataset in 'cd' 'porto' 'sanfransico'
# do
#     for model in 'GMEL'
#     do
python run_model.py --task region_representation --model $1 --dataset $2  --gpu_id $3 --config dim128_region_100 --exp_id $1
#     done
# done

# python run_model.py --task road_representation --model SRN2Vec --dataset $1 --gpu_id 1 --config dim128_road_20 --exp_id SRN2Vec