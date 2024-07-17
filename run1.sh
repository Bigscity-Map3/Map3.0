
# nohup sh run1.sh > run1.log 2>&1 &
for dataset in 'nyc' 'chicago' 'foursquare_tky' 'singapore' 'sanfransico'
do
    for model in 'CTLE'
    do
        python run_model.py --task poi_representation --model $model --dataset $dataset  --gpu_id 3 --config config --exp_id $model --train false
    done
done

# python run_model.py --task road_representation --model SRN2Vec --dataset $1 --gpu_id 1 --config dim128_road_20 --exp_id SRN2Vec