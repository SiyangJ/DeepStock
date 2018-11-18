# Data Folder

The files are shown in the order of creating/editting.

| File Name         | Description                                                     | Info                                                   |
|-------------------|-----------------------------------------------------------------|--------------------------------------------------------|
| concat.csv        | The raw data.                                                   | Collected by Jiyu Xu, Nov 5 - Nov 9                    |
| named.csv/h5      | Add names of stocks and features to raw data.                   |                                                        |
| clean.h5          | Fill missing NaN's.                                             | Cubic interpolation and clamping is used.              |
| return.h5         | Convert data to return.                                         | $r_{t+1}=\frac{p_{t+1}-p_t}{p_t}$                      |
| preprocessed_1.h5 | Normalize the volume data by each stock.                        | $v_{n,t}'=\frac{v_{n,t}-m_n}{\sigma_{n}}$              |
| vol_norm.h5       | Stores the volume normalization parameters.                     | The parameters are also stored in preprocessed_1.h5    |
| preprocessed_2.h5 | Normalize everything.                                           | Probably doesn't make sense.                           |
| XY_sequence.h5    | Formatted as supervised learning problem - X and Y.             | Corresponding to preprocessed_1.h5                     |
| XY_sequence_2.h5  | Formatted as supervised learning problem - X and Y.             | Corresponding to preprocessed_2.h5                     |
