import torch
import pickle


def load_prepared_coco_data(
    path="np_dataset/vocab2000_feat_cap_id_train_test_andvocab.pickle",
):
    with open(path, "rb") as file:
        readout = pickle.load(file)
    datasets = readout[0]
    vocab = readout[1]
    train_ds = datasets[0]
    test_ds = datasets[1]

    return (train_ds, test_ds), vocab


def create_data_loader(
    dataset,
    batch_size,
    num_distractors,
    num_workers,
    prefetch=4,
    max_seq_len=45,
    seq_padding=0.0,
    device='cpu'
):

    # include the target
    num_elems_per_datapoint = num_distractors + 1

    # need num_elems_per_datapoint elems for each batch element
    num_elemens_per_batch = num_elems_per_datapoint * batch_size

    def prep_fn(elem):
        # elem is a list of batch_size * num_elems elements
        datapoints_by_batch_elem = [
            elem[i * num_elems_per_datapoint : (i + 1) * num_elems_per_datapoint]
            for i in range(batch_size)
        ]

        def create_batch_elem(batch_elem_datapoints):
            features = [torch.as_tensor(elem[0]) for elem in batch_elem_datapoints]
            captions = [torch.as_tensor(elem[1]) for elem in batch_elem_datapoints]
            ids = [torch.as_tensor(elem[2]) for elem in batch_elem_datapoints]

            target_idx = torch.randint(high=num_elems_per_datapoint, size=(1,))

            features = torch.stack(features)
            target_feature = features[target_idx, :]
            # captions = torch.nn.utils.rnn.pad_sequence(
            #    captions, batch_first=True, padding_value=seq_padding
            # )
            target_caption = captions[target_idx]
            ids = torch.stack(ids)
            return features, target_feature, target_caption, target_idx, ids

        datapoints_by_batch_elem = [
            create_batch_elem(elem) for elem in datapoints_by_batch_elem
        ]
        all_features_batch = torch.stack([elem[0] for elem in datapoints_by_batch_elem])
        target_features_batch = torch.stack(
            [elem[1] for elem in datapoints_by_batch_elem]
        )
        target_captions_stack = torch.nn.utils.rnn.pad_sequence(
            [elem[2] for elem in datapoints_by_batch_elem],
            batch_first=True,
            padding_value=seq_padding,
        )
        target_idx_batch = torch.stack([elem[3] for elem in datapoints_by_batch_elem])
        ids_batch = torch.stack([elem[4] for elem in datapoints_by_batch_elem])

        return (
            all_features_batch,
            target_features_batch,
            target_captions_stack,
            target_idx_batch,
            ids_batch,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=num_elemens_per_batch,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=prep_fn,
        drop_last=True,
        prefetch_factor=prefetch,
        pin_memory=True,
    )
    return data_loader


if __name__ == "__main__":
    (train_ds, test_ds), vocab = load_prepared_coco_data()
    data_loader_train = create_data_loader(
        train_ds, batch_size=16, num_distractors=7, num_workers=4
    )
    data_loader_test = create_data_loader(
        test_ds, batch_size=16, num_distractors=7, num_workers=4
    )
    for elem in data_loader_train:
        print([e.shape for e in elem])
