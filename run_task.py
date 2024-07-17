import os
import pdb
import torch
import argparse
import importlib
import numpy as np
import pandas as pd
from veccity.evaluator.utils import generate_road_representaion_downstream_data
from veccity.evaluator.road_representation_evaluator import RoadRepresentationEvaluator
from veccity.evaluator.representation_evaluator import RepresentationEvaluator
from veccity.evaluator.downstream_models.similarity_search_model import SimilaritySearchModel
from veccity.evaluator.hhgcl_evaluator import HHGCLEvaluator
from veccity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    # parser.add_argument('--model', type=str,
    #                     default='GRU', help='the name of model')
    representaition_object = 'road'
    # parser.add_argument('--dataset', type=str, default='METR_LA')
    dataset = 'porto'

    # parser.add_argument('--gpu_id', type=int, default=0)
    # args = vars(parser.parse_args())
    # model = args['model']
    # dataset = args['dataset']
    # gpu_id = args['gpu_id']
    gpu_id = 2
    output_dim = 256
    exp_id = 'tmp'
    
    concat = True
    config = {
        'dataset': dataset,
        'exp_id': exp_id,
        'model': 'UUKG',
        'embed_size': output_dim,
        'd_model':output_dim,
        'output_dim': output_dim,
        "device": torch.device(f"cuda:{gpu_id}"),
        "representation_object": representaition_object,
        "save_result": True,
        "region_clf_label": "type",
        "is_static": True
    }
    print(config)
    logger = get_logger(config)
    if concat:
        if representaition_object == 'region':
            data_feature = {}
            uukg_embedding_path = f'/home/zhangwt/tyb/tyb/remote/representation/raw_data/{dataset}/{representaition_object}_embedding.npy'
            evaluator = HHGCLEvaluator(config, data_feature)
            uukg_embedding = np.load(uukg_embedding_path)
            models = ['ZEMob', 'HDGE', 'MVURE', 'MGFN', 'ReMVC', 'HREP', 'Node2Vec', 'LINE']
            for model in models:
                model_embedding_path = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy' \
                            .format(16, model, dataset, 128)
                model_embedding = np.load(model_embedding_path)
                # pdb.set_trace()
                embedding = np.concatenate([uukg_embedding, model_embedding], axis=1)
                # 16_evaluate_ZEMob_porto_128.csv
                result = {}
                mae, rmse, r2, mape = evaluator._valid_flow(embedding)
                result['mae'] = [mae]
                result['rmse'] = [rmse]
                result['mape'] = [mape]
                result['r2'] = [r2]
                bilinear_mae,bilinear_rmse,bilinear_r2,bilinear_mape = evaluator._valid_flow_using_bilinear(embedding)
                result['bilinear_mae'] = [bilinear_mae]
                result['bilinear_rmse'] = [bilinear_rmse]
                result['bilinear_mape'] = [bilinear_mape]
                result['bilinear_r2'] = [bilinear_r2]
                y_truth,useful_index,micro_f1, macro_f1 = evaluator._valid_clf(embedding)
                result['clf_micro_f1'] = [micro_f1]
                result['clf_macro_f1'] = [macro_f1]
                df = pd.DataFrame(result, index=[0])
                result_path = './veccity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.csv'. \
                    format(exp_id, exp_id, model, dataset, str(output_dim))
                df.to_csv(result_path, index=False)
        elif representaition_object == 'road':
            data_feature = {}
            uukg_embedding_path = f'/home/zhangwt/tyb/tyb/remote/representation/raw_data/{dataset}/{representaition_object}_embedding.npy'
            evaluator = HHGCLEvaluator(config, data_feature)
            uukg_embedding = np.load(uukg_embedding_path)
            models = ['LINE', 'SRN2Vec','Toast', 'JCLRNT','SARN', 'START','HyperRoad']
            for model in models:
                if model in ['SRN2Vec']:
                    continue 
                evaluator.model_name=model
                print(evaluator.model_name)
                model_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy' \
                        .format(15, model, dataset, 128)
                model_embedding = np.load(model_embedding_path)
                if model == 'LINE':
                    road_offset=np.load(f'/home/zhangwt/tyb/tyb/remote/representation/raw_data/{dataset}/region_embedding.npy').shape[0]
                    model_embedding = model_embedding[road_offset:road_offset+uukg_embedding.shape[0], :]
                embedding = np.concatenate([uukg_embedding, model_embedding], axis=1)
                print(embedding.shape)
                evaluator.evaluate(embedding)
                  
    else:
        uukg_embedding_path = f'./raw_data/data/{dataset}/{representaition_object}_embedding.npy'
        data_feature = {}
        embedding = np.load(uukg_embedding_path)
        if representaition_object == 'region':
            evaluator = HHGCLEvaluator(config, data_feature)
            evaluator._valid_flow(embedding)
            evaluator._valid_flow_using_bilinear(embedding)
            evaluator._valid_clf(embedding)
        elif representaition_object == 'road':
            evaluator = HHGCLEvaluator(config, data_feature)
            tte_model = evaluator.get_downstream_model('TravelTimeEstimationModel')
            result = tte_model.run(embedding, evaluator.data_label['tte'])
            print(result)
            tsi_model = evaluator.get_downstream_model('SpeedInferenceModel')
            result = tsi_model.run(embedding, evaluator.data_label['tsi'])
            print(result)
            sts_model = evaluator.get_downstream_model('SimilaritySearchModel')
            sts_model.embedding = embedding
            new_row = np.zeros((1, sts_model.embedding.shape[1]))
            sts_model.embedding = np.concatenate((sts_model.embedding, new_row), axis=0)
            result = sts_model.run()
            print(result)
        elif representaition_object == 'poi':
            pass


