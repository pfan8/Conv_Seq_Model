from core.solver import CaptioningSolver
from core.model import CaptionGenerator
import pickle
import numpy as np


def main():
    # load train dataset
    with open("/home/luban/cas/train_features.pkl","rb") as f:
        train_features = pickle.load(f)
    with open("/home/luban/cas/train_labels.pkl","rb") as f:
        train_labels = pickle.load(f)
    # with open("/home/luban/cas/train_features.pkl","rb") as f:
    #     train_features = pickle.load(f)
    # with open("/home/luban/cas/train_labels.pkl","rb") as f:
    #     train_labels = pickle.load(f)
    word_to_idx = {"<START>":-3,"<END>":-2,"<NULL>":-1}
    dim_feature = train_features.shape[1]
    n_time_step = train_labels.shape[1] - 1
    print "n_time_step:%d"  % n_time_step

    model = CaptionGenerator(word_to_idx, V=np.max(train_features)+1, dim_feature=dim_feature, dim_embed=128,
                                       dim_hidden=128, n_time_step=n_time_step, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    data = {"features":train_features, "labels":train_labels}
    solver = CaptioningSolver(model, data, data, n_epochs=50000, batch_size=100, update_rule='adam',
                                          learning_rate=1e-6, print_every=100, save_every=10, image_path='./image/',
                                    pretrained_model=None, model_path='./model/cnn/', test_model='/ais/gobi5/linghuan/basic-attention/model/lstm/lstm/model-19',
                                     print_bleu=True, log_path='./log/')

    solver.train()

if __name__ == "__main__":
    main()