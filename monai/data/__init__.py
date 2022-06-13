# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .box_utils import (
    box_area,
    box_centers,
    box_giou,
    box_iou,
    box_pair_giou,
    boxes_center_distance,
    centers_in_boxes,
    convert_box_mode,
    convert_box_to_standard_mode,
)
from .csv_saver import CSVSaver
from .dataloader import DataLoader
from .dataset import (
    ArrayDataset,
    CacheDataset,
    CacheNTransDataset,
    CSVDataset,
    Dataset,
    DatasetFunc,
    LMDBDataset,
    NPZDictItemDataset,
    PersistentDataset,
    SmartCacheDataset,
    ZipDataset,
)
from .dataset_summary import DatasetSummary
from .decathlon_datalist import (
    check_missing_files,
    create_cross_validation_datalist,
    load_decathlon_datalist,
    load_decathlon_properties,
)
from .folder_layout import FolderLayout
from .grid_dataset import GridPatchDataset, PatchDataset, PatchIter, PatchIterd
from .image_dataset import ImageDataset
from .image_reader import ImageReader, ITKReader, NibabelReader, NrrdReader, NumpyReader, PILReader
from .image_writer import (
    SUPPORTED_WRITERS,
    ImageWriter,
    ITKWriter,
    NibabelWriter,
    PILWriter,
    logger,
    register_writer,
    resolve_writer,
)
from .iterable_dataset import CSVIterableDataset, IterableDataset, ShuffleBuffer
from .meta_obj import MetaObj, get_track_meta, set_track_meta
from .meta_tensor import MetaTensor
from .nifti_saver import NiftiSaver
from .nifti_writer import write_nifti
from .png_saver import PNGSaver
from .png_writer import write_png
from .samplers import DistributedSampler, DistributedWeightedRandomSampler
from .synthetic import create_test_image_2d, create_test_image_3d
from .test_time_augmentation import TestTimeAugmentation
from .thread_buffer import ThreadBuffer, ThreadDataLoader
from .torchscript_utils import load_net_with_metadata, save_net_with_metadata
from .utils import (
    compute_importance_map,
    compute_shape_offset,
    convert_tables_to_dicts,
    correct_nifti_header_if_necessary,
    create_file_basename,
    decollate_batch,
    dense_patch_slices,
    get_extra_metadata_keys,
    get_random_patch,
    get_valid_patch_size,
    is_supported_format,
    iter_patch,
    iter_patch_slices,
    json_hashing,
    list_data_collate,
    orientation_ras_lps,
    pad_list_data_collate,
    partition_dataset,
    partition_dataset_classes,
    pickle_hashing,
    rectify_header_sform_qform,
    remove_extra_metadata,
    remove_keys,
    reorient_spatial_axes,
    resample_datalist,
    select_cross_validation_folds,
    set_rnd,
    sorted_dict,
    to_affine_nd,
    worker_init_fn,
    zoom_affine,
)
from .wsi_datasets import PatchWSIDataset
from .wsi_reader import BaseWSIReader, CuCIMWSIReader, OpenSlideWSIReader, WSIReader
