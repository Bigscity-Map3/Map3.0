
# for dataset in 'porto'
# do
#     for model in 'CTLE' 'Teaser' 'Hier' 'POI2Vec' 'Tale' 'SkipGram'
#     do
#       python run_model.py --task poi_representation --model $model --dataset $dataset  --gpu_id 0 --config config --train false --exp_id $model
#     done
# done

python run_model.py --task road_representation --model $1 --dataset $2 --config dim128_road_100 --train false --exp_id $1