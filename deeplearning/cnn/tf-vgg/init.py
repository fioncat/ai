#!/usr/bin/python

# Check if tensorflow-vgg, vgg parameter, flower data exist
# if vgg parameter do not exist, download it.
from tqdm import tqdm


class DownloadProgress(tqdm):

    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


if __name__ == '__main__':

    from urllib.request import urlretrieve
    from os.path import isfile, isdir

    # Check tensorflow-vgg
    vgg_dir = 'tensorflow-vgg/'
    # make sure tensorflow-vgg exit
    if not isdir(vgg_dir):
        raise Exception('tensorflow vgg dir not exist!')

    # Check VGG parameters, if not exist, download.
    if not isfile(vgg_dir + 'vgg16.npy'):
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='VGG16 Parameters') as pbar:
            urlretrieve(
                'https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy',
                vgg_dir + 'vgg16.npy',
                pbar.hook)
    else:
        print('vgg parameter exits.')

    # Check flower data, if not exist, download.
    if not isfile('data/flower_photos.tar.gz'):
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='Flowers Dataset') as pbar:
            urlretrieve(
                'http://download.tensorflow.org/example_images/flower_photos.tgz',
                'data/flower_photos.tar.gz',
                pbar.hook)
    else:
        print('flower data exist.')

    # extract flower data
    import tarfile
    dataset_folder_path = 'data/flower_photos'
    if not isdir(dataset_folder_path):
        with tarfile.open('data/flower_photos.tar.gz') as tar:
            tar.extractall()
            tar.close()
