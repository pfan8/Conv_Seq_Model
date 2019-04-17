from core.solver import CaptioningSolver
from core.model import CaptionGenerator
import pickle
import numpy as np

def main():
    # load train dataset
    with open("/home/luban/cas/test_features.pkl","rb") as f:
        test_features = pickle.load(f)
    with open("/home/luban/cas/test_labels.pkl","rb") as f:
        test_labels = pickle.load(f)
    # with open("/home/luban/cas/seq_lengths.p","rb") as f:
    #     seq_lengths = pickle.load(f)
    word_to_idx = {"<START>":-3,"<END>":-2,"<NULL>":-1}
    dim_feature = test_features.shape[1]
    n_time_step = test_labels.shape[1] - 1
    print "n_time_step:%d"  % n_time_step


    model = CaptionGenerator(word_to_idx, V=np.max(test_features)+1, dim_feature=dim_feature, dim_embed=128,
                                       dim_hidden=128, n_time_step=n_time_step, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    data = {"features":test_features, "labels":test_labels}
    
    solver = CaptioningSolver(model, data, data, n_epochs=50, batch_size=100, update_rule='adam',
                                          learning_rate=0.001, print_every=500, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='./model/cnn/', test_model='./model/cnn/model-305',
                                     print_bleu=True, log_path='./log/')

    solver.test(data, split='val')


if __name__ == "__main__":
    main()