import numpy as np
from numpy.lib.stride_tricks import as_strided


class AirQualityWindow(object):
    
    @staticmethod
    def gen_sequence(id_df, seq_length, horizon, seq_cols):
        """
        Generate input sequences considering forecast horizon.
        """
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]

        max_t = num_elements - horizon

        for start in range(0, max_t - seq_length + 1):
            stop = start + seq_length
            yield data_matrix[start:stop, :]

    @staticmethod
    def gen_labels(id_df, seq_length, horizon, label):
        """
        Generate labels for multi-step forecasting.

        Parameters
        ----------
        id_df : pandas.DataFrame
            DataFrame for a single campaign / id.
        seq_length : int
            Input sequence length.
        horizon : int
            Forecast horizon.
        label : list[str]
            Target column name.

        Returns
        -------
        numpy.ndarray
            Shape: [samples, horizon]
        """
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]

        labels = []

        for start in range(seq_length, num_elements - horizon + 1):
            labels.append(data_matrix[start:start + horizon, 0])

        return np.array(labels)

    def seq_generation(
        self,
        train_FD_norm,
        cols_sensors,
        sequence_length,
        horizon,
        campaign_name
    ):
        seq_gen = (
            list(
                AirQualityWindow.gen_sequence(
                    train_FD_norm[train_FD_norm[campaign_name] == id],
                    sequence_length,
                    horizon,
                    cols_sensors
                )
            )
            for id in train_FD_norm[campaign_name].unique()
        )
    
        seq_array_train = np.concatenate(list(seq_gen)).astype(np.float32)
    
        return seq_array_train

    def label_generation(
        self,
        train_FD_norm,
        sequence_length,
        horizon,
        campaign_name,
        target_name
    ):
        label_gen = [
            AirQualityWindow.gen_labels(
                train_FD_norm[train_FD_norm[campaign_name] == id],
                sequence_length,
                horizon,
                [target_name]
            )
            for id in train_FD_norm[campaign_name].unique()
        ]

        label_array_train = np.concatenate(label_gen).astype(np.float32)

        return label_array_train

    def networkinput_generation(self, seq_array_train, stride, n_window, window_length):
        """
        Versão VETORIZADA: Remove loops Python e usa manipulação de memória direta do NumPy.
        Input: (samples, sequence_length, sensors)
        """
        N, L, S = seq_array_train.shape
        s0, s1, s2 = seq_array_train.strides

        view = as_strided(
            seq_array_train,
            shape=(N, n_window, window_length, S),
            strides=(s0, stride * s1, s1, s2)
        )

        train_FD_sensor = []
        for i in range(S):
            # Extrai o sensor i, shape (N, n_window, window_length) -> adiciona dim (..., 1)
            sensor_data = view[..., i].reshape(N, n_window, window_length, 1).copy()
            train_FD_sensor.append(sensor_data)

        return train_FD_sensor