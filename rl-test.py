from core.rl_solver import CaptioningSolver
from core.rl_model import CaptionGenerator
import pickle
import numpy as np

def main():
   # load train dataset
    with open("/nfs/private/cas/test_features.pkl","rb") as f:
        test_features = pickle.load(f)
    with open("/nfs/private/cas/test_labels.pkl","rb") as f:
        test_labels = pickle.load(f)
    with open("./data/baseline/sample_labels.pkl","rb") as f:
        sample_labels = pickle.load(f)
    word_to_idx = {"<START>":0,"<END>":-2,"<NULL>":-1}
    # load val dataset to print out bleu scores every epoch
    dim_feature = test_features.shape[1]
    n_time_step = test_labels.shape[1]-1
    print "n_time_step:%d"  % n_time_step

    model = CaptionGenerator(word_to_idx, dim_feature=dim_feature, dim_embed=128,
                                       dim_hidden=128, n_time_step=n_time_step, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    data = {"features":test_features, "labels":test_labels, "sample_labels":sample_labels}
    solver = CaptioningSolver(model, data, data, n_epochs=50000, batch_size=100, update_rule='adam',
                                          learning_rate=1e-7, print_every=100, save_every=10, image_path='./image/',
                                    test_model='./model/rl/model_0.90_F1-508', model_path='./model/rl/',
                                     print_bleu=True, log_path='./log/')

    solver.test(data, split='val')


if __name__ == "__main__":
    main()