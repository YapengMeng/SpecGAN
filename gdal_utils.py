'''李汶原 20180804
gdal读取图像的辅助工具库
'''
'''
    Auxiliary tools for reading images by gdal
    @author: Wenyuan Li
    @date: 2018.08.04
'''
# import gdal
import numpy as np
import os

from osgeo import osr

from osgeo import gdal
from pyproj import Proj
from bs4 import BeautifulSoup


def convt_geo(x, y, proj='utm', zone_id=42, ellps='WGS84', south=False, inverse=True, **kwargs):
    p = Proj(proj=proj, zone=zone_id, ellps=ellps, south=south, preserve_units=False, **kwargs)
    lon, lat = p(x, y, inverse=inverse)
    return lon, lat


def get_image_shape(img_path):
    '''
    get image shape
    :param img_path: 
    :return: (height，width，bands)
    '''

    dataset = gdal.Open(img_path)
    if dataset is None:
        print("can't open file %s" % img_path)
        exit(-1)
    im_width = dataset.RasterXSize  # 图像的列数
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    del dataset
    return im_height, im_width, im_bands


def save_full_image(img_path, img, geoTranfsorm=None, proj=None, data_format='GDAL_FORMAT', **kwargs):
    '''
    :param img_path: save path
    :param img: 
    :param geoTranfsorm: 
    :param proj: 
    :return: 
    '''
    if data_format not in ['GDAL_FORMAT', 'NUMPY_FORMAT']:
        raise Exception('data_format parameter error')
    if 'uint8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_CInt16
    elif 'uint16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(img.shape) == 3:
        if data_format == 'NUMPY_FORMAT':
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 1)
        im_bands, im_height, im_width = img.shape
    elif len(img.shape) == 2:
        img = np.array([img])
        im_bands, im_height, im_width = img.shape
    else:
        im_bands, (im_height, im_width) = 1, img.shape

    driver = gdal.GetDriverByName("GTIFF")
    dataset = driver.Create(img_path, im_width, im_height, im_bands, datatype, **kwargs)
    if geoTranfsorm:
        dataset.SetGeoTransform(geoTranfsorm)
    if proj:
        dataset.SetProjection(proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(img[i])


def gdal_copy(src_file, dst_file, options=None):
    driver = gdal.GetDriverByName("GTIFF")
    dataset = gdal.Open(src_file)
    driver.CreateCopy(dst_file, dataset, strict=1, options=options)


def read_full_image(img_path, scale_factor=1, as_rgb=True,
                    data_format='GDAL_FORMAT', normalize=True, normalize_factor=16, band_idx=None):
    '''
    :param img_path: 
    :param scale_factor: 
    :param as_rgb: 
    :param data_format: 
    :return: 
    '''
    im_height, im_width, _ = get_image_shape(img_path)
    img = read_image(img_path, 0, 0, im_width, im_height, scale_factor, as_rgb, data_format, normalize=normalize,
                     normalize_factor=normalize_factor, band_idx=band_idx)
    return img


def read_image(img_path, width_offset, height_offset, width, height, scale_factor=1, as_rgb=True,
               data_format='GDAL_FORMAT', normalize=True, normalize_factor=16, band_idx=None):
    '''
        Read the image, support block read, if the read size exceeds the actual size of the image, the boundary is 0
        :param img_path: path of the image to be read
        :param width_offset: indicates the offset in the x direction
        :param height_offset: indicates the offset in the y direction
        :param width: Fast width of the image to be read
        :param height: height of the image to be read
        :param scale_factor: scale ratio
        :param as_rgb: specifies whether to convert a grayscale image to an rgb image
        :param data_format: format of the returned result. There are two possible values: 'GDAL_FORMAT','NUMPY_FORMAT'
        'GDAL_FORMAT': Returns image shape of '(bands,height,width)'
        'NUMPY_FORMAT': Returns image size of (height,width,bands)
        The shape length of the returned image in each format is 3
        :return:
    '''
    if data_format not in ['GDAL_FORMAT', 'NUMPY_FORMAT']:
        raise Exception('data_format parameters error')
    dataset = gdal.Open(img_path)
    if dataset is None:
        print("can't open file %s" % img_path)
        exit(-1)

    im_width = dataset.RasterXSize  # image rows
    im_height = dataset.RasterYSize
    if band_idx is None:
        im_bands = dataset.RasterCount
        band_idx = list(range(1, im_bands + 1))
    else:
        im_bands = len(band_idx)
    scale_width = int(width / scale_factor)
    scale_height = int(height / scale_factor)
    # To determine whether the index is out of bounds,
    # only the image that is not out of bounds is read,
    # and the rest of the image is filled with 0
    block_width = width
    block_height = height
    if width_offset + width > im_width:
        block_width = im_width - width_offset
    if height_offset + height > im_height:
        block_height = im_height - height_offset
    scale_block_width = int(block_width / scale_factor)
    scale_block_height = int(block_height / scale_factor)
    im_data = np.zeros((im_bands, scale_block_height, scale_block_width), dtype=np.uint16)
    for i, idx in enumerate(band_idx):
        band = dataset.GetRasterBand(idx)
        im_data[i] = band.ReadAsArray(width_offset, height_offset, block_width, block_height,
                                      scale_block_width, scale_block_height)
    if im_bands == 1 and as_rgb:
        im_data = np.tile(im_data, (3, 1, 1))
    elif im_bands >= 4 and as_rgb:
        im_data = im_data[0:3, :, :]

    if normalize:
        if isinstance(normalize_factor, int):
            im_data = im_data.astype(np.float32) / normalize_factor
        elif len(normalize_factor) == 2:
            im_data = im_data.astype(np.float32)
            im_data = (im_data - normalize_factor[0]) / (normalize_factor[1] - normalize_factor[0])
            im_data = np.clip(im_data, 0., 1.)
            im_data = im_data * 255
        else:
            raise NotImplementedError
        im_data = np.array(im_data, np.uint8)  # shape: (bands,height,width)

    if width != block_width or height != block_height:
        im_data = np.pad(im_data,
                         ((0, 0), (0, scale_height - scale_block_height), (0, scale_width - scale_block_width)),
                         mode='constant')

    if data_format == 'NUMPY_FORMAT':
        im_data = np.swapaxes(im_data, 0, 1)
        im_data = np.swapaxes(im_data, 1, 2)
    del dataset
    return im_data


def get_geoTransform(img_path, ):
    dataset = gdal.Open(img_path)
    if dataset is None:
        print("can't open file %s" % img_path)
        exit(-1)
    geotransform = dataset.GetGeoTransform()
    return geotransform


def get_transforms_xml(img_file, xml_file):
    im_height, im_width, _ = get_image_shape(img_file)
    file = open(xml_file, encoding='utf-8').read()
    soup = BeautifulSoup(file, 'xml')
    get_transforms = [-1.] * 6
    get_transforms[0] = float(soup.find('TopLeftLongitude').text)
    get_transforms[3] = float(soup.find('TopLeftLatitude').text)
    xmin = float(soup.find('TopLeftLongitude').text)
    ymin = float(soup.find('TopLeftLatitude').text)

    x = float(soup.find('TopRightLongitude').text)
    y = float(soup.find('TopRightLatitude').text)
    get_transforms[1] = (x - xmin) / im_width
    get_transforms[4] = (y - ymin) / im_width

    x = float(soup.find('BottomLeftLongitude').text)
    y = float(soup.find('BottomLeftLatitude').text)

    get_transforms[2] = (x - xmin) / im_height
    get_transforms[5] = (y - ymin) / im_height

    return tuple(get_transforms)


def get_transforms_rpb(img_file, result_file, exe_path):
    im_height, im_width, _ = get_image_shape(img_file)
    img_file = os.path.abspath(img_file)
    exe_path = os.path.abspath(exe_path)
    result_file = os.path.abspath(result_file)
    get_transforms = [-1.] * 6
    # read Upper left coordinate
    offset_height, offset_width = 0, 0
    cmd_line = '%s %s %s %d %d' % (exe_path, img_file, result_file, offset_width, offset_height)
    os.system(cmd_line)
    # print(cmd_line)
    lonlat_list = np.loadtxt(result_file)
    if lonlat_list[-1] == 0:
        raise Exception("Upper left coordinate resolution error")
    get_transforms[0] = lonlat_list[0]
    get_transforms[3] = lonlat_list[1]
    xmin = lonlat_list[0]
    ymin = lonlat_list[1]

    # read Upper right coordinate
    for i in np.arange(0, 1, 0.1, dtype=np.float):
        offset_width, offset_height = int(im_width * (1 - i)), 0
        cmd_line = '%s %s %s %d %d' % (exe_path, img_file, result_file, offset_width, offset_height)
        os.system(cmd_line)
        lonlat_list = np.loadtxt(result_file)
        if lonlat_list[-1] > 0:
            x = lonlat_list[0]
            y = lonlat_list[1]
            get_transforms[1] = (x - xmin) / offset_width
            get_transforms[4] = (y - ymin) / offset_width
            break
    else:
        raise Exception("can't convert")

    # read down left coordinate
    for i in np.arange(0, 1, 0.1, dtype=np.float):
        offset_width, offset_height = 0, int(im_height * (1 - i))
        cmd_line = '%s %s %s %d %d' % (exe_path, img_file, result_file, offset_width, offset_height)
        os.system(cmd_line)
        lonlat_list = np.loadtxt(result_file)
        if lonlat_list[-1] > 0:
            x = lonlat_list[0]
            y = lonlat_list[1]
            get_transforms[2] = (x - xmin) / offset_height
            get_transforms[5] = (y - ymin) / offset_height
            break
    else:
        raise Exception("can't convert")

    return tuple(get_transforms)


def get_projection(img_path):
    dataset = gdal.Open(img_path)
    if dataset is None:
        print("can't open file %s" % img_path)
        exit(-1)
    projection = dataset.GetProjection()
    return projection


def get_dataset(img_file):
    dataset = gdal.Open(img_file)
    if dataset is None:
        print("can't open file %s" % img_file)
        exit(-1)
    return dataset

