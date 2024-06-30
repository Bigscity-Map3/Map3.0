
for dataset in 'porto'
do
    for model in 'CTLE' 'Teaser' 'Hier' 'POI2Vec' 'Tale' 'SkipGram'
    do
      python run_model.py --task poi_representation --model $model --dataset $dataset  --gpu_id 0 --config config --train false --exp_id $model
    done
done