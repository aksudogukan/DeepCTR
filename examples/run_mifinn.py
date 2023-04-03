import copy

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import FiBiNET, MiFiNN

if __name__ == "__main__":
    """data = pd.read_csv('./avazu_sample.txt')
    data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
    data['hour'] = data['hour'].apply(lambda x: str(x)[6:])

    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_model', 'device_type', 'device_conn_type',  # 'device_ip',
                       'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    target = ['click']"""

    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        # print("data[feat]:", data[feat])
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        # print("data[feat]:", data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    #mi score
    from sklearn.feature_selection import mutual_info_classif
    print(data.head(5))
    print(data.shape)
    print(data.columns)
    data_copy = copy.deepcopy(data)
    label = data_copy.pop(target[0])
    print("label", label.head(2))
    print("sparse_features", sparse_features)
    score = mutual_info_classif(data_copy[sparse_features], label)
    print(score)

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = MiFiNN(score, linear_feature_columns, dnn_feature_columns, task='binary')
    # model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
