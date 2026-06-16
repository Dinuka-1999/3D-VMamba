import numpy as np
import SimpleITK as sitk
import shutil
import multiprocessing

from time import sleep
from tqdm import tqdm
from os.path import join, isfile, isdir
from typing import Union
from batchgenerators.utilities.file_and_folder_operations import *
from skimage.transform import resize

import nnunetv2
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.ssl_paths import ssl_preprocessed, ssl_raw
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO as rw 
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


class SSL_preprocessor(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = "ZScoreNormalization"
            normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnunetv2.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
            normalizer = normalizer_class(use_mask_for_norm=False,
                                          intensityproperties={})
            data[c] = normalizer.run(data[c], None)
        return data
    
    def run_case_npy(self, data: np.ndarray, properties: dict):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy

        # apply transpose_forward, this also needs to be applied to the spacing!
        # data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        # if seg is not None:
            # seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        # original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, None)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]


        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data)

       
        old_shape = data.shape[1:]
        data = resize(data, (128,128,128),order =3, mode='edge',anti_aliasing=True)
        
        return data, properties

    def run_case(self, image_files: List[str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        # load image(s)
        data, data_properties = rw().read_images([image_files])

        data, data_properties = self.run_case_npy(data, data_properties)
        return data, data_properties
    
    def run_case_save(self, output_filename_truncated: str, image_files: List[str]):
        data, properties = self.run_case(image_files)
        data = data.astype(np.float32, copy=False)
        
        # print('dtypes', data.dtype, seg.dtype)
        block_size_data, chunk_size_data = nnUNetDatasetBlosc2.comp_blosc2_params(
            data.shape,
            tuple((128,128,128)),
            data.itemsize)

        nnUNetDatasetBlosc2.save_case(data,None, properties, output_filename_truncated,
                                      chunks=chunk_size_data, blocks=block_size_data,
                                      chunks_seg=None, blocks_seg=None)
    
    def get_file_names(self, dataset_folder):
        identifiers = []
        for f in subfiles(join(dataset_folder, 'imagesTr'), suffix='.nii.gz', join=False, sort=True):
            identifiers.append(f[:-7])  # Remove '.nii.gz' suffix
        return identifiers
    
    def run(self, dataset_ID: Union[str, int], num_processes: int=4):
        
        dataset_name = maybe_convert_to_dataset_name(dataset_ID)
        assert isdir(join(ssl_raw, dataset_name)), "The requested dataset is not found in this folder"
    
        output_folder = join(ssl_preprocessed, dataset_name)

        if isdir(output_folder):
            shutil.rmtree(output_folder)
        maybe_mkdir_p(output_folder)
        
        dataset = self.get_file_names(join(ssl_raw, dataset_name))
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.        
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            for k in dataset:
                r.append(p.starmap_async(self.run_case_save,[(
            join(output_folder, k),
            join(ssl_raw, dataset_name, "imagesTr", k + '.nii.gz')
        )]))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    # get done so that errors can be raised
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        r[_].get()  # allows triggering errors
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

if __name__ == "__main__":
    preprocessor = SSL_preprocessor(verbose=False)
    preprocessor.run(dataset_ID="Dataset120_Heart", num_processes=4)