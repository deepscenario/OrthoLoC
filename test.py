

from ortholoc.dataset import OrthoLoC

dataset = OrthoLoC(
    dataset_dir=None,  # path to the dataset, if empty, the dataset will be downloaded automatically
    sample_paths=None,  # path to the samples (cannot be specified when dataset_dir is set), e.g. ["demo/samples/highway_rural.npz", "demo/samples/urban_residential_xDOPDSM.npz"]
    set_name='all',  # name of the set (all, train, val, test_inPlace, test_outPlace), if None, all samples will be used
    start=0.,  # start of the dataset
    end=.1,  # end of the dataset
    mode=0,  # mode 0 for all samples, 1 for samples with same UAV imagery and geodata domain, 2 for samples with DOP domain shift, 3 for samples with DOP and DSM domains shift
    new_size=None,  # new size of the images (useful for training)
    limit_size=None,  # limit size of the images (useful for debugging)
    shuffle=True,  # shuffle the dataset
    scale_query_image=1.0,  # scale of the query image (1.0 keep the original size)
    scale_dop_dsm=1.0,  # scale of the DOP and DSM images
    gt_matching_confidences_decay=1.0,  # decay of the matching confidences (the larger the less confident will be the GT for non-unique points like points on facades)
    covisibility_ratio=1.0,  # ratio of the covisibility (0.0 to 1.0, 0.0 exclusive, the larger the more area in the geodata will be visible for the UAV)
    return_tensor=False,  # return the samples while iterating over the dataset as dict of torch tensors
    predownload=False,  # if True, it will download the dataset while constructing the dataset object, otherwise it will download while iterating over the dataset
)

a = 1