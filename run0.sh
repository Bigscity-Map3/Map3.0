
# for dataset in 'chicago' 'foursquare_tky' 'singapore' 'porto' 'sanfransico' 'nyc'
# do
#     for model in 'CTLE' 'Teaser' 'Hier'
#     do
#       python run_model.py --task poi_representation --model $model --dataset $dataset  --gpu_id 0 --config config --exp_id $model --train false
#     done
# done

python run_model.py --task road_representation --model HRNR --dataset bj --gpu_id 1 --config dim128_road_20 --exp_id hrnr
python run_model.py --task road_representation --model HRNR --dataset cd --gpu_id 1 --config dim128_road_20 --exp_id hrnr
python run_model.py --task road_representation --model HRNR --dataset singapore --gpu_id 1 --config dim128_road_20 --exp_id hrnr
python run_model.py --task road_representation --model HRNR --dataset porto --gpu_id 1 --config dim128_road_20 --exp_id hrnr
python run_model.py --task road_representation --model HRNR --dataset sanfransico --gpu_id 1 --config dim128_road_20 --exp_id hrnr